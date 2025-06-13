import torch,torchaudio
import os,sys,json
from tqdm import tqdm
import numpy as np

#from codeclm_song_v1.codeclm.semantic_extractor.SpeechDecoder_v01.generate import Tango
from generate_septoken import Tango as Tango_sep
from generate_2rvq import Tango as Tango_1x2
import kaldiio
from kaldiio import WriteHelper
from audio import AudioFile

from demucs.models.pretrained import get_model_from_yaml
from filelock import FileLock

# os.path.join(args.model_dir, "htdemucs.pth"), os.path.join(args.model_dir, "htdemucs.yaml")
class Separator:
    def __init__(self, dm_model_path='demucs/ckpt/htdemucs.pth', dm_config_path='demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)

    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if (fs != 48000):
            a = torchaudio.functional.resample(a, fs, 48000)
        # if a.shape[-1] >= 48000*10:
        #     a = a[..., :48000*10]
        # else:
        #     a = torch.cat([a, a], -1)
        # return a[:, 0:48000*10]
        return a
    
    def run(self, audio_path, output_dir='demucs/test_output', ext=".flac"):
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []
        # lock_path = os.path.join(output_dir, f"{name}.lock")
        # with FileLock(lock_path):  # 加一个避免多卡访问时死锁
        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            # drums_path, bass_path, other_path, vocal_path = output_paths
            vocal_path = output_paths[0]
        else:
            lock_path = os.path.join(output_dir, f"{name}_separate.lock")
            with FileLock(lock_path):
                drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        minlen = min(full_audio.shape[-1], vocal_audio.shape[-1])
        # bgm_audio = full_audio[:, 0:minlen] - vocal_audio[:, 0:minlen]
        bgm_audio = self.load_audio(drums_path) + self.load_audio(bass_path) + self.load_audio(other_path)
        for path in [drums_path, bass_path, other_path, vocal_path]:
            os.remove(path)
        return full_audio, vocal_audio, bgm_audio

def read_wav(fname, sample_rate=48_000):
    try:
        orig_samples, fs = torchaudio.load(fname)
    except:
        af = AudioFile(fname)
        orig_samples = af.read()
        fs = af.samplerate()
        orig_samples = orig_samples[0]
    if(fs!=sample_rate):
        orig_samples = torchaudio.functional.resample(orig_samples, fs, sample_rate)
        fs = sample_rate
    if orig_samples.shape[0] == 1:
        orig_samples = torch.cat([orig_samples, orig_samples], 0)
    return orig_samples

if __name__ == "__main__":
    # Define Model
    json_path = sys.argv[1]
    
    mus_infos = []
    with open(json_path) as f:
        for line in f:
            item = json.loads(line)
            mus_infos.append(item)

    tango_sep = Tango_sep(model_path="./saved/model_septoken/model_2.safetensors")
    tango_1x2 = Tango_1x2(model_path = './saved/model_2rvq/model_2_fixed.safetensors', rvq_num=2)
    separator = Separator()

    # Feature extraction loop
    # for i in tqdm(range(2000)):
    first_time = True
    for item in tqdm(mus_infos):
        if(os.path.exists(item['path'])):
            full_path = item['path']
        else:
            full_path = '/mnt/share/' + item['path']

        full_tensor, vocal_tensor, bgm_tensor = separator.run(full_path)

        # full_tensor = read_wav(full_path)
        # vocal_tensor = read_wav(vocal_path)
        # length = min(full_tensor.shape[-1], vocal_tensor.shape[-1])
        # full_tensor, vocal_tensor = full_tensor[:, 0:length], vocal_tensor[:, 0:length]
        # bgm_tensor = full_tensor - vocal_tensor
        codes_1x2 = tango_1x2.sound2code(full_tensor)
        codes_vocal, codes_bgm = tango_sep.sound2code(vocal_tensor, bgm_tensor)
        codes = torch.cat([codes_1x2[:,[0],:], codes_vocal, codes_bgm], 1).cpu().numpy()
        save_path = full_path.replace('.wav', '.1x1_and_sep.npy').replace('.mp3', '.1x1_and_sep.npy').replace('.flac', '.1x1_and_sep.npy').replace('.ogg', '.1x1_and_sep.npy')
        assert save_path != full_path, (save_path, full_path)
        np.save(save_path, codes)

        if(first_time):
            first_time = False
            print(codes_vocal.shape, codes_bgm.shape)

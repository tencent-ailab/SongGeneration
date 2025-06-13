import torch,torchaudio
import os,sys,json
from tqdm import tqdm

#from codeclm_song_v1.codeclm.semantic_extractor.SpeechDecoder_v01.generate import Tango
from generate_septoken import Tango
import kaldiio
from kaldiio import WriteHelper
from audio import AudioFile

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
    outdir = sys.argv[2]
    
    mus_infos = []
    with open(json_path) as f:
        for line in f:
            item = json.loads(line)
            mus_infos.append(item)

    tango = Tango(model_path="./saved/model_septoken/model_2.safetensors")
    
    
    # Feature extraction loop
    # for i in tqdm(range(2000)):
    first_time = True
    with WriteHelper('ark,scp:{}/token_vocal.ark,{}/token_vocal.scp'.format(outdir, outdir), write_function="pickle") as writer_vocal,  WriteHelper('ark,scp:{}/token_bgm.ark,{}/token_bgm.scp'.format(outdir, outdir), write_function="pickle") as writer_bgm:
        print('ark,scp:{}/token_vocal.ark,{}/token_vocal.scp'.format(outdir, outdir))
        print('ark,scp:{}/token_bgm.ark,{}/token_bgm.scp'.format(outdir, outdir))
        for item in tqdm(mus_infos):
            try:
            # if True:
                idx = item['idx']
                # print(idx)
                if(os.path.exists(item['path'])):
                    full_path = item['path']
                else:
                    full_path = '/mnt/share/' + item['path']
                if(os.path.exists(item['vocal_path'])):
                    vocal_path = item['vocal_path']
                    bgm_paths = item['bgm_path']
                else:
                    vocal_path = '/mnt/share/' + item['vocal_path']
                    bgm_paths = ['/mnt/share/' + p for p in item['bgm_path']]
                vocal_tensor = read_wav(vocal_path)
                # full_tensor = read_wav(full_path)
                # length = min(full_tensor.shape[-1], vocal_tensor.shape[-1])
                # full_tensor, vocal_tensor = full_tensor[:, 0:length], vocal_tensor[:, 0:length]
                # bgm_tensor = full_tensor - vocal_tensor
                bgm_tensor = sum([read_wav(p) for p in bgm_paths])
                codes_vocal, codes_bgm = tango.sound2code(vocal_tensor, bgm_tensor)
                writer_vocal(str(idx), codes_vocal.cpu())
                writer_bgm(str(idx), codes_bgm.cpu())
                if(first_time):
                    first_time = False
                    print(codes_vocal.shape, codes_bgm.shape)
            except:
                print(item['vocal_path'])
                print(item['bgm_path'])
                continue
            
            # idx = item['idx']
            # # print(idx)
            # full_path = item['path']
            # vocal_path = item['vocal_path']
            # bgm_paths = item['bgm_path']
            # full_tensor = read_wav(full_path)
            # vocal_tensor = read_wav(vocal_path)
            # length = min(full_tensor.shape[-1], vocal_tensor.shape[-1])
            # full_tensor, vocal_tensor = full_tensor[:, 0:length], vocal_tensor[:, 0:length]
            # bgm_tensor = full_tensor - vocal_tensor
            # codes_vocal, codes_bgm = tango.sound2code(vocal_tensor, bgm_tensor)
            # writer_vocal(str(idx), codes_vocal.cpu())
            # writer_bgm(str(idx), codes_bgm.cpu())
            # if(first_time):
            #     first_time = False
            #     print(codes_vocal.shape, codes_bgm.shape)


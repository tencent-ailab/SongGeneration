import json
import torch
from tqdm import tqdm
from model_septoken import PromptCondAudioDiffusion
from diffusers import DDIMScheduler, DDPMScheduler
import torchaudio
import librosa
import os
import math
import numpy as np
# from tools.get_mulan import get_mulan
from tools.get_1dvae_large import get_model
import tools.torch_tools as torch_tools
from safetensors.torch import load_file
from third_party.demucs.models.pretrained import get_model_from_yaml
from filelock import FileLock


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

class Tango:
    def __init__(self, \
        model_path, \
        vae_config,
        vae_model,
        layer_vocal=7,\
        layer_bgm=3,\
        device="cuda:0"):
        
        self.sample_rate = 48000
        scheduler_name = "configs/scheduler/stable_diffusion_2.1_largenoise_sample.json"
        self.device = device

        self.vae = get_model(vae_config, vae_model)
        self.vae = self.vae.to(device)
        self.vae=self.vae.eval()
        self.layer_vocal=layer_vocal
        self.layer_bgm=layer_bgm

        self.MAX_DURATION = 360
        main_config = {
            "num_channels":32,
            "unet_model_name":None,
            "unet_model_config_path":"configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
            "snr_gamma":None,
        }
        self.model = PromptCondAudioDiffusion(**main_config).to(device)
        if model_path.endswith(".safetensors"):
            main_weights = load_file(model_path)
        else:
            main_weights = torch.load(model_path, map_location=device)
        self.model.load_state_dict(main_weights, strict=False)
        print ("Successfully loaded checkpoint from:", model_path)
        
        self.model.eval()
        self.model.init_device_dtype(torch.device(device), torch.float32)
        
        # self.scheduler = DDIMScheduler.from_pretrained( \
        #     scheduler_name, subfolder="scheduler")
        # self.scheduler = DDPMScheduler.from_pretrained( \
        #     scheduler_name, subfolder="scheduler")
        print("Successfully loaded inference scheduler from {}".format(scheduler_name))


    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code(self, orig_vocal, orig_bgm, batch_size=8):
        if(orig_vocal.ndim == 2):
            audios_vocal = orig_vocal.unsqueeze(0).to(self.device)
        elif(orig_vocal.ndim == 3):
            audios_vocal = orig_vocal.to(self.device)
        else:
            assert orig_vocal.ndim in (2,3), orig_vocal.shape
        
        if(orig_bgm.ndim == 2):
            audios_bgm = orig_bgm.unsqueeze(0).to(self.device)
        elif(orig_bgm.ndim == 3):
            audios_bgm = orig_bgm.to(self.device)
        else:
            assert orig_bgm.ndim in (2,3), orig_bgm.shape

        
        audios_vocal = self.preprocess_audio(audios_vocal)
        audios_vocal = audios_vocal.squeeze(0)
        audios_bgm = self.preprocess_audio(audios_bgm)
        audios_bgm = audios_bgm.squeeze(0)
        if audios_vocal.shape[-1] > audios_bgm.shape[-1]:
            audios_vocal = audios_vocal[:,:audios_bgm.shape[-1]]
        else:
            audios_bgm = audios_bgm[:,:audios_vocal.shape[-1]]


        orig_length = audios_vocal.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1

        while(audios_vocal.shape[-1] < min_samples):
            audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)
            audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)
        int_max_len=audios_vocal.shape[-1]//min_samples+1
        audios_vocal = torch.cat([audios_vocal, audios_vocal], -1)
        audios_bgm = torch.cat([audios_bgm, audios_bgm], -1)
        audios_vocal=audios_vocal[:,:int(int_max_len*(min_samples))]
        audios_bgm=audios_bgm[:,:int(int_max_len*(min_samples))]
        codes_vocal_list=[]
        codes_bgm_list=[]

    

        audio_vocal_input = audios_vocal.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)
        audio_bgm_input = audios_bgm.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_vocal_input.shape[0], batch_size):
            [codes_vocal,codes_bgm], _, spk_embeds = self.model.fetch_codes_batch((audio_vocal_input[audio_inx:audio_inx+batch_size]), (audio_bgm_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer_vocal=self.layer_vocal,layer_bgm=self.layer_bgm)
            codes_vocal_list.append(codes_vocal)
            codes_bgm_list.append(codes_bgm)

        codes_vocal = torch.cat(codes_vocal_list, 0).permute(1,0,2).reshape(1, -1)[None]
        codes_bgm = torch.cat(codes_bgm_list, 0).permute(1,0,2).reshape(1, -1)[None]
        codes_vocal=codes_vocal[:,:,:output_len]
        codes_bgm=codes_bgm[:,:,:output_len]

        return codes_vocal, codes_bgm

    @torch.no_grad()
    def code2sound(self, codes, prompt_vocal=None, prompt_bgm=None, duration=40, guidance_scale=1.5, num_steps=20, disable_progress=False, chunked=False, chunk_size=128):
        codes_vocal,codes_bgm = codes
        codes_vocal = codes_vocal.to(self.device)
        codes_bgm = codes_bgm.to(self.device)

        min_samples = duration * 25 # 40ms per frame
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples
        ovlp_frames = ovlp_samples
        first_latent = torch.randn(codes_vocal.shape[0], min_samples, 64).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0


        if(isinstance(prompt_vocal, torch.Tensor) and isinstance(prompt_bgm, torch.Tensor)):
            # prepare prompt
            prompt_vocal = prompt_vocal.to(self.device)
            prompt_bgm = prompt_bgm.to(self.device)
            if(prompt_vocal.ndim == 3):
                assert prompt_vocal.shape[0] == 1, prompt_vocal.shape
                prompt_vocal = prompt_vocal[0]
                prompt_bgm = prompt_bgm[0]
            elif(prompt_vocal.ndim == 1):
                prompt_vocal = prompt_vocal.unsqueeze(0).repeat(2,1)
                prompt_bgm = prompt_bgm.unsqueeze(0).repeat(2,1)
            elif(prompt_vocal.ndim == 2):
                if(prompt_vocal.shape[0] == 1):
                    prompt_vocal = prompt_vocal.repeat(2,1)
                    prompt_bgm = prompt_bgm.repeat(2,1)

            if(prompt_vocal.shape[-1] < int(30 * self.sample_rate)):
                # if less than 30s, just choose the first 10s
                prompt_vocal = prompt_vocal[:,:int(10*self.sample_rate)] # limit max length to 10.24
                prompt_bgm = prompt_bgm[:,:int(10*self.sample_rate)] # limit max length to 10.24
            else:
                # else choose from 20.48s which might includes verse or chorus
                prompt_vocal = prompt_vocal[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
                prompt_bgm = prompt_bgm[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
            
            true_latent = self.vae.encode_audio(prompt_vocal+prompt_bgm).permute(0,2,1)
            
            first_latent[:,0:true_latent.shape[1],:] = true_latent
            first_latent_length = true_latent.shape[1]
            first_latent_codes = self.sound2code(prompt_vocal, prompt_bgm)
            first_latent_codes_vocal = first_latent_codes[0]
            first_latent_codes_bgm = first_latent_codes[1]
            first_latent_codes_length = first_latent_codes_vocal.shape[-1]
            codes_vocal = torch.cat([first_latent_codes_vocal, codes_vocal], -1)
            codes_bgm = torch.cat([first_latent_codes_bgm, codes_bgm], -1)
            

        codes_len= codes_vocal.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * self.sample_rate)
        # target_len = int(codes_len / 100 * 4 * self.sample_rate)
        # code repeat
        if(codes_len < min_samples):
            while(codes_vocal.shape[-1] < min_samples):
                codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
                codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)

            codes_vocal = codes_vocal[:,:,0:min_samples]
            codes_bgm = codes_bgm[:,:,0:min_samples]
        codes_len = codes_vocal.shape[-1]
        if((codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(codes_vocal.shape[-1] < len_codes):
                codes_vocal = torch.cat([codes_vocal, codes_vocal], -1)
                codes_bgm = torch.cat([codes_bgm, codes_bgm], -1)
            codes_vocal = codes_vocal[:,:,0:len_codes]
            codes_bgm = codes_bgm[:,:,0:len_codes]
        latent_length = min_samples
        latent_list = []
        spk_embeds = torch.zeros([1, 32, 1, 32], device=codes_vocal.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for sinx in range(0, codes_vocal.shape[-1]-hop_samples, hop_samples):
                codes_vocal_input=codes_vocal[:,:,sinx:sinx+min_samples]
                codes_bgm_input=codes_bgm[:,:,sinx:sinx+min_samples]
                if(sinx == 0):
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes([codes_vocal_input,codes_bgm_input], spk_embeds, first_latent, latent_length, incontext_length=incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    true_latent = latent_list[-1][:,:,-ovlp_frames:].permute(0,2,1)
                    len_add_to_1000 = min_samples - true_latent.shape[-2]
                    incontext_length = true_latent.shape[-2]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0],  len_add_to_1000, true_latent.shape[-1]).to(self.device)], -2)
                    latents = self.model.inference_codes([codes_vocal_input,codes_bgm_input], spk_embeds, true_latent, latent_length, incontext_length=incontext_length,  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,:,first_latent_length:]
        min_samples =  int(min_samples * self.sample_rate // 1000 * 40)
        hop_samples = int(hop_samples * self.sample_rate // 1000 * 40)
        ovlp_samples = min_samples - hop_samples
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                cur_output = self.vae.decode_audio(latent, chunked=chunked, chunk_size=chunk_size)[0].detach().cpu()

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    @torch.no_grad()
    def preprocess_audio(self, input_audios_vocal, threshold=0.8):
        assert len(input_audios_vocal.shape) == 3, input_audios_vocal.shape
        nchan = input_audios_vocal.shape[1]
        input_audios_vocal = input_audios_vocal.reshape(input_audios_vocal.shape[0], -1)
        norm_value = torch.ones_like(input_audios_vocal[:,0])
        max_volume = input_audios_vocal.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios_vocal.reshape(input_audios_vocal.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)
    
    @torch.no_grad()
    def sound2sound(self, orig_vocal,orig_bgm, prompt_vocal=None,prompt_bgm=None, steps=50, disable_progress=False):
        codes_vocal, codes_bgm = self.sound2code(orig_vocal,orig_bgm)
        codes=[codes_vocal, codes_bgm]
        wave = self.code2sound(codes, prompt_vocal,prompt_bgm, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        return wave
    
    def to(self, device=None, dtype=None, non_blocking=False):
        if device is not None:
            self.device = device
            self.model.device = device
        self.vae = self.vae.to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self

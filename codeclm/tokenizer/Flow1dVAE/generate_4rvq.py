import json
import torch
from tqdm import tqdm
from model_4rvq import PromptCondAudioDiffusion
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
from audio import AudioFile

class Tango:
    def __init__(self, \
        model_path, \
        layer_num=6, \
        rvq_num=1, \
        device="cuda:0"):
        
        self.sample_rate = 48000
        scheduler_name = "configs/scheduler/stable_diffusion_2.1_largenoise_sample.json"
        self.device = device

        self.vae = get_model()
        self.vae = self.vae.to(device)
        self.vae=self.vae.eval()
        self.layer_num = layer_num

        self.MAX_DURATION = 360
        main_config = {
            "num_channels":32,
            "unet_model_name":None,
            "unet_model_config_path":"configs/models/transformer2D_wocross_inch112_1x4_multi_large.json",
            "snr_gamma":None,
        }
        self.rvq_num = rvq_num
        # print("rvq_num: ", self.rvq_num)
        # exit()
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
    def sound2code(self, orig_samples, batch_size=8):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        audios = self.preprocess_audio(audios)
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        # print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes, _, spk_embeds = self.model.fetch_codes_batch((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num, rvq_num=self.rvq_num)
            # print("codes",codes[0].shape)

            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(self.rvq_num, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def sound2code_ds(self, orig_samples, ds, batch_size=6):
        if(orig_samples.ndim == 2):
            audios = orig_samples.unsqueeze(0).to(self.device)
        elif(orig_samples.ndim == 3):
            audios = orig_samples.to(self.device)
        else:
            assert orig_samples.ndim in (2,3), orig_samples.shape
        audios = self.preprocess_audio(audios)
        audios = audios.squeeze(0)
        orig_length = audios.shape[-1]
        min_samples = int(40 * self.sample_rate)
        # 40秒对应10个token
        output_len = int(orig_length / float(self.sample_rate) * 25) + 1
        # print("output_len: ", output_len)

        while(audios.shape[-1] < min_samples):
            audios = torch.cat([audios, audios], -1)
        int_max_len=audios.shape[-1]//min_samples+1
        audios = torch.cat([audios, audios], -1)
        audios=audios[:,:int(int_max_len*(min_samples))]
        codes_list=[]

        audio_input = audios.reshape(2, -1, min_samples).permute(1, 0, 2).reshape(-1, 2, min_samples)

        for audio_inx in range(0, audio_input.shape[0], batch_size):
            # import pdb; pdb.set_trace()
            codes, _, spk_embeds = self.model.fetch_codes_batch_ds((audio_input[audio_inx:audio_inx+batch_size]), additional_feats=[],layer=self.layer_num, rvq_num=self.rvq_num, ds=ds)
            # print("codes",codes[0].shape)

            codes_list.append(torch.cat(codes, 1))
            # print("codes_list",codes_list[0].shape)

        codes = torch.cat(codes_list, 0).permute(1,0,2).reshape(self.rvq_num, -1)[None] # B 3 T -> 3 B T
        codes=codes[:,:,:output_len]

        return codes

    @torch.no_grad()
    def code2sound(self, codes, prompt=None, duration=40, guidance_scale=1.5, num_steps=20, disable_progress=False):
        codes = codes.to(self.device)

        min_samples = duration * 25 # 40ms per frame
        hop_samples = min_samples // 4 * 3
        ovlp_samples = min_samples - hop_samples
        hop_frames = hop_samples
        ovlp_frames = ovlp_samples
        first_latent = torch.randn(codes.shape[0], min_samples, 64).to(self.device)
        first_latent_length = 0
        first_latent_codes_length = 0

        if(isinstance(prompt, torch.Tensor)):
            # prepare prompt
            prompt = prompt.to(self.device)
            if(prompt.ndim == 3):
                assert prompt.shape[0] == 1, prompt.shape
                prompt = prompt[0]
            elif(prompt.ndim == 1):
                prompt = prompt.unsqueeze(0).repeat(2,1)
            elif(prompt.ndim == 2):
                if(prompt.shape[0] == 1):
                    prompt = prompt.repeat(2,1)

            if(prompt.shape[-1] < int(30 * self.sample_rate)):
                # if less than 30s, just choose the first 10s
                prompt = prompt[:,:int(10*self.sample_rate)] # limit max length to 10.24
            else:
                # else choose from 20.48s which might includes verse or chorus
                prompt = prompt[:,int(20*self.sample_rate):int(30*self.sample_rate)] # limit max length to 10.24
            
            true_latent = self.vae.encode_audio(prompt).permute(0,2,1)
            # print("true_latent.shape", true_latent.shape)
            # print("first_latent.shape", first_latent.shape)
            #true_latent.shape torch.Size([1, 250, 64])
            # first_latent.shape torch.Size([1, 1000, 64])
            
            first_latent[:,0:true_latent.shape[1],:] = true_latent
            first_latent_length = true_latent.shape[1]
            first_latent_codes = self.sound2code(prompt)
            first_latent_codes_length = first_latent_codes.shape[-1]
            codes = torch.cat([first_latent_codes, codes], -1)

        codes_len= codes.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 100 * 4 * self.sample_rate)
        # target_len = int(codes_len / 100 * 4 * self.sample_rate)
        # code repeat
        if(codes_len < min_samples):
            while(codes.shape[-1] < min_samples):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:min_samples]
        codes_len = codes.shape[-1]
        if((codes_len - ovlp_samples) % hop_samples > 0):
            len_codes=math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples + ovlp_samples
            while(codes.shape[-1] < len_codes):
                codes = torch.cat([codes, codes], -1)
            codes = codes[:,:,0:len_codes]
        latent_length = min_samples
        latent_list = []
        spk_embeds = torch.zeros([1, 32, 1, 32], device=codes.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for sinx in range(0, codes.shape[-1]-hop_samples, hop_samples):
                codes_input=[]
                codes_input.append(codes[:,:,sinx:sinx+min_samples])
                if(sinx == 0):
                    # print("Processing {} to {}".format(sinx/self.sample_rate, (sinx + min_samples)/self.sample_rate))
                    incontext_length = first_latent_length
                    latents = self.model.inference_codes(codes_input, spk_embeds, first_latent, latent_length, incontext_length=incontext_length, additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)
                else:
                    # print("Processing {} to {}".format(sinx/self.sample_rate, (sinx + min_samples)/self.sample_rate))
                    true_latent = latent_list[-1][:,:,-ovlp_frames:].permute(0,2,1)
                    print("true_latent.shape", true_latent.shape)
                    len_add_to_1000 = 1000 - true_latent.shape[-2]
                    # print("len_add_to_1000", len_add_to_1000)
                    # exit()
                    incontext_length = true_latent.shape[-2]
                    true_latent = torch.cat([true_latent, torch.randn(true_latent.shape[0],  len_add_to_1000, true_latent.shape[-1]).to(self.device)], -2)
                    latents = self.model.inference_codes(codes_input, spk_embeds, true_latent, latent_length, incontext_length=incontext_length,  additional_feats=[], guidance_scale=1.5, num_steps = num_steps, disable_progress=disable_progress, scenario='other_seg')
                    latent_list.append(latents)

        latent_list = [l.float() for l in latent_list]
        latent_list[0] = latent_list[0][:,:,first_latent_length:]
        min_samples =  int(min_samples * self.sample_rate // 1000 * 40)
        hop_samples = int(hop_samples * self.sample_rate // 1000 * 40)
        ovlp_samples = min_samples - hop_samples
        with torch.no_grad():
            output = None
            for i in range(len(latent_list)):
                latent = latent_list[i]
                cur_output = self.vae.decode_audio(latent)[0].detach().cpu()

                if output is None:
                    output = cur_output
                else:
                    ov_win = torch.from_numpy(np.linspace(0, 1, ovlp_samples)[None, :])
                    ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                    print("output.shape", output.shape)
                    print("ov_win.shape", ov_win.shape)
                    output[:, -ovlp_samples:] = output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:] + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                    output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
            output = output[:, 0:target_len]
        return output

    @torch.no_grad()
    def preprocess_audio(self, input_audios, threshold=0.8):
        assert len(input_audios.shape) == 3, input_audios.shape
        nchan = input_audios.shape[1]
        input_audios = input_audios.reshape(input_audios.shape[0], -1)
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios.reshape(input_audios.shape[0], nchan, -1)/norm_value.unsqueeze(-1).unsqueeze(-1)
    
    @torch.no_grad()
    def sound2sound(self, sound, prompt=None, steps=50, disable_progress=False):
        codes = self.sound2code(sound)
        # print(codes.shape)
        # exit()
        wave = self.code2sound(codes, prompt, guidance_scale=1.5, num_steps=steps, disable_progress=disable_progress)
        # print(fname, wave.shape)
        return wave
    
    def file2code(self, fname):
        try:
            orig_samples, fs = torchaudio.load(fname)
        except:
            af = AudioFile(fname)
            orig_samples = af.read()
            fs = af.samplerate()
            orig_samples = orig_samples[0]
        if(fs!=self.sample_rate):
            orig_samples = torchaudio.functional.resample(orig_samples, fs, self.sample_rate)
            fs = self.sample_rate
        if orig_samples.shape[0] == 1:
            orig_samples = torch.cat([orig_samples, orig_samples], 0)
        return self.sound2code(orig_samples)

    def file2code_ds(self, fname, ds):
        try:
            orig_samples, fs = torchaudio.load(fname)
        except:
            af = AudioFile(fname)
            orig_samples = af.read()
            fs = af.samplerate()
            orig_samples = orig_samples[0]
        if(fs!=self.sample_rate):
            orig_samples = torchaudio.functional.resample(orig_samples, fs, self.sample_rate)
            fs = self.sample_rate
        if orig_samples.shape[0] == 1:
            orig_samples = torch.cat([orig_samples, orig_samples], 0)
        return self.sound2code_ds(orig_samples, ds)

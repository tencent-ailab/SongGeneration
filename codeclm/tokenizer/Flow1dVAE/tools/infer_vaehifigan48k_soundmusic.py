import json
import torch
from tqdm import tqdm
import torchaudio
import librosa
import os
import math
import numpy as np
from get_melvaehifigan48k import build_pretrained_models
import tools.torch_tools as torch_tools

class Tango:
    def __init__(self, \
        device="cuda:0"):
        
        self.sample_rate = 48000
        self.device = device

        self.vae, self.stft = build_pretrained_models()
        self.vae, self.stft = self.vae.eval().to(device), self.stft.eval().to(device)

        # print(sum(p.numel() for p in self.vae.parameters()));exit()

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    def sound2sound_generate_longterm(self, fname, batch_size=1, duration=10.24, steps=200, disable_progress=False):
        """ Genrate audio without condition. """
        num_frames = math.ceil(duration * 100. / 8)
        with torch.no_grad():
            orig_samples, fs = torchaudio.load(fname)
            if(orig_samples.shape[-1]<int(duration*48000)):
                orig_samples = orig_samples.repeat(1,math.ceil(int(duration*48000)/float(orig_samples.shape[-1])))
            # orig_samples = torch.cat([torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device), orig_samples, torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device)], -1).to(self.device)
            orig_samples = torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device)], -1).to(self.device)
            if(fs!=48000):orig_samples = torchaudio.functional.resample(orig_samples, fs, 48000)
            # resampled_audios = orig_samples[[0],int(4.64*48000):int(35.36*48000)+480].clamp(-1,1)
            resampled_audios = orig_samples[[0],int(0*48000):int(duration*48000)+480].clamp(-1,1)
            orig_samples = orig_samples[[0],:]

            mel, _, _ = torch_tools.wav_to_fbank2(resampled_audios, -1, fn_STFT=self.stft)
            mel = mel.unsqueeze(1).to(self.device)
            latents = torch.cat([self.vae.get_first_stage_encoding(self.vae.encode_first_stage(mel[[m]])) for m in range(mel.shape[0])],0)

            mel = self.vae.decode_first_stage(latents)
            audio = self.vae.decode_to_waveform(mel)
            audio = torch.from_numpy(audio)

            orig_samples = orig_samples[...,0:int(duration * 48000)]
            if(orig_samples.shape[-1]<audio.shape[-1]):
                orig_samples = torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], audio.shape[-1]-orig_samples.shape[-1], dtype=orig_samples.dtype, device=orig_samples.device)],-1)
            else:
                orig_samples = orig_samples[:,0:audio.shape[-1]]
            output = torch.cat([orig_samples.detach().cpu(),audio.detach().cpu()],0)
        return output

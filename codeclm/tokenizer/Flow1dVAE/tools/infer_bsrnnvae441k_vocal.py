import json
import torch
from tqdm import tqdm
import torchaudio
import librosa
import os
import math
import numpy as np
from tools.get_bsrnnvae import get_bsrnnvae
import tools.torch_tools as torch_tools

class Tango:
    def __init__(self, \
        device="cuda:0"):
        
        self.sample_rate = 44100
        self.device = device

        self.vae = get_bsrnnvae()
        self.vae = self.vae.eval().to(device)

    def sound2sound_generate_longterm(self, fname, batch_size=1, duration=20.48, steps=200, disable_progress=False):
        """ Genrate audio without condition. """
        num_frames = math.ceil(duration * 100. / 8)
        with torch.no_grad():
            orig_samples, fs = torchaudio.load(fname)
            if(fs!=44100):
                orig_samples = torchaudio.functional.resample(orig_samples, fs, 44100)
                fs = 44100
            if(orig_samples.shape[-1]<int(duration*44100*2)):
                orig_samples =  torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], int(duration*44100*2+480)-orig_samples.shape[-1], \
                    dtype=orig_samples.dtype, device=orig_samples.device)], -1)
            # orig_samples = torch.cat([torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device), orig_samples, torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device)], -1).to(self.device)
            orig_samples = torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device)], -1).to(self.device)
            if(fs!=44100):orig_samples = torchaudio.functional.resample(orig_samples, fs, 44100)
            # resampled_audios = orig_samples[[0],int(4.64*44100):int(35.36*48000)+480].clamp(-1,1)
            resampled_audios = orig_samples[[0],0:int(duration*2*44100)+480].clamp(-1,1)
            orig_samples = orig_samples[[0],0:int(duration*2*44100)]

            audio = self.vae(orig_samples[:,None,:])[:,0,:]

            if(orig_samples.shape[-1]<audio.shape[-1]):
                orig_samples = torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], audio.shape[-1]-orig_samples.shape[-1], dtype=orig_samples.dtype, device=orig_samples.device)],-1)
            else:
                orig_samples = orig_samples[:,0:audio.shape[-1]]
            output = torch.cat([orig_samples.detach().cpu(),audio.detach().cpu()],0)
        return output

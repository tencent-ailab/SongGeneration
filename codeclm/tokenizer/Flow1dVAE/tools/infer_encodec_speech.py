import json
import torch
from tqdm import tqdm
from audiocraft.models.loaders import load_compression_model
import torchaudio
import librosa
import os
import math
import numpy as np

class Tango:
    def __init__(self, \
        device="cuda:0"):
        
        self.sample_rate = 48000
        self.rsp48to32 = torchaudio.transforms.Resample(48000, 32000).to(device)
        self.rsp32to48 = torchaudio.transforms.Resample(32000, 48000).to(device)

        encodec = load_compression_model('compression_state_dict.bin', device='cpu').eval()
        encodec.set_num_codebooks(1)
        self.encodec = encodec.eval().to(device)
        self.device = torch.device(device)
        print ("Successfully loaded encodec model")

    @torch.no_grad()
    def remix(self, filename, duration=10.24, start_step=1000, steps=999, disable_progress=False):
        """ Genrate audio without condition. """
        orig_samples, fs = torchaudio.load(filename)
        if(orig_samples.shape[-1]<int(duration*48000)):
            orig_samples = orig_samples.repeat(1,math.ceil(int(duration*48000)/float(orig_samples.shape[-1])))
        orig_samples = torch.cat([orig_samples, torch.zeros(orig_samples.shape[0], int(duration * fs)//2, dtype=orig_samples.dtype, device=orig_samples.device)], -1).to(self.device)
        if(fs!=48000):orig_samples = torchaudio.functional.resample(orig_samples, fs, 48000)
        init_audio = orig_samples[[0],None,0:int(duration*48000)]

        rsped_audios = self.rsp48to32(init_audio)
        codes_rspd = self.encodec.encode(rsped_audios)[0]
        codec_audios = self.encodec.decode(codes_rspd, None)
        codec_audios = self.rsp32to48(codec_audios)
        rsped_audios = self.rsp32to48(rsped_audios)

        minlen = min(rsped_audios.shape[-1], codec_audios.shape[-1])
        output = torch.cat([rsped_audios.detach().cpu()[:,0,0:minlen],codec_audios.detach().cpu()[:,0,0:minlen]],0)
        return output

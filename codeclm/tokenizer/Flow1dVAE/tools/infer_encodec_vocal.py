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
        encodec.set_num_codebooks(4)
        self.encodec = encodec.eval().to(device)
        self.device = torch.device(device)
        print ("Successfully loaded encodec model")

    def set_num_codebooks(self, num):
        self.encodec.set_num_codebooks(num)

    @torch.no_grad()
    def remix(self, filename, start_step=1000, steps=999, disable_progress=False):
        """ Genrate audio without condition. """
        init_audio, _ = librosa.load(filename, sr=self.sample_rate, mono=False)
        if(len(init_audio.shape)>1):init_audio = init_audio[0]
        init_audio = torch.from_numpy(init_audio)[None,None,:].to(self.device)
        init_audio = init_audio[:,:,0:int(10.24*2*self.sample_rate)]
        if(init_audio.shape[-1]<int(10.24*2*self.sample_rate)):
            init_audio = torch.cat([init_audio, torch.zeros([1,1,int(10.24*2*self.sample_rate)-init_audio.shape[-1]], device=self.device)],-1)

        rsped_audios = self.rsp48to32(init_audio)
        codes_rspd = self.encodec.encode(rsped_audios)[0]
        codec_audios = self.encodec.decode(codes_rspd, None)
        codec_audios = self.rsp32to48(codec_audios)
        rsped_audios = self.rsp32to48(rsped_audios)

        minlen = min(rsped_audios.shape[-1], codec_audios.shape[-1])
        output = torch.cat([rsped_audios.detach().cpu()[:,0,0:minlen],codec_audios.detach().cpu()[:,0,0:minlen]],0)
        return output

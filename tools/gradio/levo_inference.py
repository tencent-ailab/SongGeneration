import os
import sys

import torch

import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM

from separator import Separator


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        torch.backends.cudnn.enabled = False 
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        OmegaConf.register_new_resolver("get_fname", lambda: 'default')
        OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        pt_path = os.path.join(ckpt_path, 'model.pt')

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        # Define model or load pretrained model
        model_light = CodecLM_PL(self.cfg, pt_path)

        model_light = model_light.eval().cuda()
        model_light.audiolm.cfg = self.cfg

        self.model_lm = model_light.audiolm
        self.model_audio_tokenizer = model_light.audio_tokenizer
        self.model_seperate_tokenizer = model_light.seperate_tokenizer

        self.model = CodecLM(name = "tmp",
            lm = self.model_lm,
            audiotokenizer = self.model_audio_tokenizer,
            max_duration = self.max_duration,
            seperate_tokenizer = self.model_seperate_tokenizer,
        )
        self.separator = Separator()


        self.default_params = dict(
            cfg_coef = 1.5,
            temperature = 1.0,
            top_k = 50,
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )

        self.model.set_generation_params(**self.default_params)

    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict()):
        params = {**self.default_params, **params}
        self.model.set_generation_params(**params)

        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            pmt_wav, vocal_wav, bgm_wav = self.separator.run(prompt_audio_path)
            melody_is_wav = True
        elif genre is not None and auto_prompt_path is not None:
            auto_prompt = torch.load(auto_prompt_path)
            merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
            if genre == "Auto": 
                prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
            else:
                prompt_token = auto_prompt[genre][np.random.randint(0, len(auto_prompt[genre]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = self.model.generate(**generate_inp, return_tokens=True)
            
        with torch.no_grad():
            if melody_is_wav:
                wav_seperate = self.model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type)
            else:
                wav_seperate = self.model.generate_audio(tokens, gen_type=gen_type)

        return wav_seperate[0]

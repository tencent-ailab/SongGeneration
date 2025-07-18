import os
import sys

import torch

import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from codeclm.models import builders

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
        self.pt_path = os.path.join(ckpt_path, 'model.pt')

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        self.default_params = dict(
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )


    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "all", params = dict()):
        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            separator = Separator()
            audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()
            seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()
            pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
            pmt_wav = pmt_wav.cuda()
            vocal_wav = vocal_wav.cuda()
            bgm_wav = bgm_wav.cuda()
            pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
            vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
            melody_is_wav = False
            melody_is_wav = False
            del audio_tokenizer
            del seperate_tokenizer
            del separator
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

        model_light = CodecLM_PL(self.cfg, self.pt_path)
        model_light = model_light.eval()
        model_light.audiolm.cfg = self.cfg
        model = CodecLM(name = "tmp",
            lm = model_light.audiolm,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = None,
        )
        del model_light
        model.lm = model.lm.cuda().to(torch.float16)
        params = {**self.default_params, **params}
        model.set_generation_params(**params)

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = model.generate(**generate_inp, return_tokens=True)

        del model
        torch.cuda.empty_cache()

        seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
        seperate_tokenizer = seperate_tokenizer.eval().cuda()
        model = CodecLM(name = "tmp",
            lm = None,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = seperate_tokenizer,
        )
            
        with torch.no_grad():
            if melody_is_wav:
                wav_seperate = model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type)
            else:
                wav_seperate = model.generate_audio(tokens, gen_type=gen_type)

        del seperate_tokenizer
        del model
        torch.cuda.empty_cache()

        return wav_seperate[0]

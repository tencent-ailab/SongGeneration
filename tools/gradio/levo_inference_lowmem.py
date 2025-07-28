import os
import gc
import sys

import torch

import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from codeclm.models import builders

from separator import Separator
from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse


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


    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict()):
        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            separator = Separator()
            audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            audio_tokenizer = audio_tokenizer.eval().cuda()
            pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
            pmt_wav = pmt_wav.cuda()
            vocal_wav = vocal_wav.cuda()
            bgm_wav = bgm_wav.cuda()
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
            del audio_tokenizer
            del separator
            torch.cuda.empty_cache()

            seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
            del seperate_tokenizer
            melody_is_wav = False
            torch.cuda.empty_cache()
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

        audiolm = builders.get_lm_model(self.cfg)
        checkpoint = torch.load(self.pt_path, map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()

        offload_audiolm = True if 'offload' in self.cfg.keys() and 'audiolm' in self.cfg.offload else False
        if offload_audiolm:
            audiolm_offload_param = OffloadParamParse.parse_config(audiolm, self.cfg.offload.audiolm)
            audiolm_offload_param.show()
            offload_profiler = OffloadProfiler(device_index=0, **(audiolm_offload_param.init_param_dict()))
            offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
            offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
        else:
            audiolm = audiolm.cuda().to(torch.float16)

        model = CodecLM(name = "tmp",
            lm = audiolm,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = None,
        )
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
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
                if offload_audiolm:
                    offload_profiler.reset_empty_cache_mem_line()
        offload_profiler.stop()
        del offload_profiler
        del audiolm_offload_param
        del model
        audiolm = audiolm.cpu()
        del audiolm
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
        device = "cuda:0"
        seperate_tokenizer.model.device = device
        seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
        seperate_tokenizer.model.model.device = torch.device(device)
        seperate_tokenizer = seperate_tokenizer.eval()

        offload_wav_tokenizer_diffusion =  True if 'offload' in self.cfg.keys() and 'wav_tokenizer_diffusion' in self.cfg.offload else False
        if offload_wav_tokenizer_diffusion:
            sep_offload_param = OffloadParamParse.parse_config(seperate_tokenizer, self.cfg.offload.wav_tokenizer_diffusion)
            sep_offload_param.show()
            sep_offload_profiler = OffloadProfiler(device_index=0, **(sep_offload_param.init_param_dict()))
            sep_offload_profiler.offload_layer(**(sep_offload_param.offload_layer_param_dict()))
            sep_offload_profiler.clean_cache_wrapper(**(sep_offload_param.clean_cache_param_dict()))
        else:
            seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

        model = CodecLM(name = "tmp",
            lm = None,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = seperate_tokenizer,
        )

        with torch.no_grad():
            if melody_is_wav:
                wav_seperate = model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type, chunked=True)
            else:
                wav_seperate = model.generate_audio(tokens, gen_type=gen_type, chunked=True)

        if offload_wav_tokenizer_diffusion:
            sep_offload_profiler.reset_empty_cache_mem_line()
            sep_offload_profiler.stop()
        torch.cuda.empty_cache()

        return wav_seperate[0]

import os
import time

import torch
import numpy as np
from omegaconf import OmegaConf
from vllm import LLM, SamplingParams

from codeclm.models import builders
from codeclm.models.codeclm_gen import CodecLM_gen
from generate import check_language_by_text, load_audio


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        torch.backends.cudnn.enabled = False 
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        OmegaConf.register_new_resolver("get_fname", lambda: 'default')
        OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
        if audio_tokenizer is not None:
            for param in audio_tokenizer.parameters():
                param.requires_grad = False
        print("Audio tokenizer successfully loaded!")
        audio_tokenizer = audio_tokenizer.eval().cuda()
        self.model_condition = CodecLM_gen(cfg=self.cfg,name = "tmp",audiotokenizer = audio_tokenizer,max_duration = self.max_duration)
        self.model_condition.condition_provider.conditioners.load_state_dict(torch.load(self.cfg.lm_checkpoint+"/conditioners_weights.pth"))
        self.embeded_eosp1 = torch.load(self.cfg.lm_checkpoint+'/embeded_eosp1.pt')
        print('Conditioner successfully loaded!')
        self.llm = LLM(
            model=self.cfg.lm_checkpoint,
            trust_remote_code=True,
            tensor_parallel_size=self.cfg.vllm.device_num,
            enforce_eager=True,
            dtype="bfloat16",  
            gpu_memory_utilization=0.65,
            max_num_seqs=8,
            tokenizer=None,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            enable_chunked_prefill=True,
        )

        self.default_params = dict(
            cfg_coef = 1.8,
            temperature = 0.8,
            top_k = 5000,
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )

    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict()):
        params = {**self.default_params, **params}

        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            pmt_wav = load_audio(prompt_audio_path)
            melody_is_wav = True
        elif genre is not None and auto_prompt_path is not None:
            auto_prompt = torch.load(auto_prompt_path)
            lang = check_language_by_text(lyric)
            prompt_token = auto_prompt[genre][lang][np.random.randint(0, len(auto_prompt[genre][lang]))]
            pmt_wav = prompt_token[:,[0],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            melody_is_wav = True

        description = description.lower() if description else '.'
        description = '[Musicality-very-high]' + ', ' + description
        generate_inp = {
            'descriptions': [lyric.replace("  ", " ")],
            'type_info': [description],
            'melody_wavs': pmt_wav,
            'melody_is_wav': melody_is_wav,
            'embeded_eosp1': self.embeded_eosp1,
        }
        fused_input, audio_qt_embs = self.model_condition.generate_condition(**generate_inp, return_tokens=True)
        prompt_token = audio_qt_embs[0][0].tolist() if audio_qt_embs else []
        allowed_token_ids = [x for x in range(self.cfg.lm.code_size+1) if x not in prompt_token]
        sampling_params = SamplingParams(
            max_tokens=self.cfg.audio_tokenizer_frame_rate*self.max_duration,
            temperature=params["temperature"],
            stop_token_ids=[self.cfg.lm.code_size],
            top_k=params["top_k"],
            frequency_penalty=0.2,
            seed=int(time.time() * 1000000) % (2**32) if self.cfg.vllm.cfg else -1,
            allowed_token_ids=allowed_token_ids,
            guidance_scale=params["cfg_coef"]
        )
        # 拆成现支持的batch 3 CFG形式
        prompts = [{"prompt_embeds": embed} for embed in fused_input]
        condi, uncondi = prompts[0], prompts[1]
        promptss = [condi, condi, uncondi]
        outputs = self.llm.generate(promptss, sampling_params=sampling_params)
        token_ids_CFG = torch.tensor(outputs[1].outputs[0].token_ids)
        token_ids_CFG = token_ids_CFG[:-1].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            if melody_is_wav:
                wav_cfg = self.model_condition.generate_audio(token_ids_CFG, pmt_wav, chunked=True)
            else:
                wav_cfg = self.model_condition.generate_audio(token_ids_CFG, chunked=True)

        return wav_cfg[0]

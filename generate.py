from hmac import new
import sys
import os
import argparse

import time
import json
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf
from codeclm.models import builders
import gc
from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from third_party.demucs.models.pretrained import get_model_from_yaml


auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']

class Separator:
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0) -> None:
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
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        return a[:, 0:48000*10]
    
    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name, _ = os.path.splitext(os.path.split(audio_path)[-1])
        output_paths = []

        for stem in self.demucs_model.sources:
            output_path = os.path.join(output_dir, f"{name}_{stem}{ext}")
            if os.path.exists(output_path):
                output_paths.append(output_path)
        if len(output_paths) == 1:  # 4
            vocal_path = output_paths[0]
        else:
            drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(audio_path, output_dir, device=self.device)
            for path in [drums_path, bass_path, other_path]:
                os.remove(path)
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        return full_audio, vocal_audio, bgm_audio


def parse_args():
    parser = argparse.ArgumentParser(description='Song Generation Script')
    
    # 必需参数
    parser.add_argument('--ckpt_path', type=str, required=True,
                      help='Path to the checkpoint directory containing config.yaml and model.pt')
    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='Path to input JSONL file containing generation tasks')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save generated audio files and results')
    # 可选参数
    parser.add_argument('--generate_type', type=str, default='mixed',
                      help='Type of generation: "vocal" or "bgm" or "separate" or "mixed" (default: "mixed")')
    parser.add_argument('--use_flash_attn', action='store_true',
                      help='Whether to use flash attention (default: False)')
    parser.add_argument('--low_mem', action='store_true',
                      help='Whether to use low memory mode (default: False)')
    return parser.parse_args()

def generate(args):
    ckpt_path = args.ckpt_path
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    ckpt_path = os.path.join(ckpt_path, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg.lm.use_flash_attn_2 = args.use_flash_attn
    print(f"use_flash_attn: {args.use_flash_attn}")
    cfg.mode = 'inference'
    max_duration = cfg.max_dur
    gen_type = args.generate_type
    

    separator = Separator()
    auto_prompt = torch.load('ckpt/prompt.pt')
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    audio_tokenizer = audio_tokenizer.eval().cuda()
    merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()

        
    new_items = []
    for line in lines:
        item = json.loads(line)
        target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"
        # get prompt audio
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = separator.run(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            pmt_wav = pmt_wav
            vocal_wav = vocal_wav
            bgm_wav = bgm_wav
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            if item["auto_prompt_audio_type"] == "Auto": 
                prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
            else:
                prompt_token = auto_prompt[item["auto_prompt_audio_type"]][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
        new_items.append(item)

    del audio_tokenizer
    del separator

    torch.cuda.empty_cache()

    if "audio_tokenizer_checkpoint_sep" in cfg.keys():
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg) 
    else:
        seperate_tokenizer = None
    
    if seperate_tokenizer is not None:
        seperate_tokenizer = seperate_tokenizer.eval().cuda()

    for item in new_items:
        if "prompt_audio_path" in item:
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav

    torch.cuda.empty_cache()
    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()
    audiolm = audiolm.cuda().to(torch.float16)

    model = CodecLM(name = "tmp",
        lm = audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = seperate_tokenizer,
    )

    cfg_coef = 1.5 #25
    temp = 0.9
    top_k = 50
    top_p = 0.0
    record_tokens = True
    record_window = 50

    model.set_generation_params(duration=max_duration, extend_stride=5, temperature=temp, cfg_coef=cfg_coef,
                                top_k=top_k, top_p=top_p, record_tokens=record_tokens, record_window=record_window)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)

    for item in new_items:
        lyric = item["gt_lyric"]
        descriptions = item["descriptions"] if "descriptions" in item else None
        pmt_wav = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav = item['bgm_wav']
        melody_is_wav = item['melody_is_wav']
        target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"


        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [descriptions],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }
        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
        mid_time = time.time()

        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if gen_type == 'separate':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'], chunked=True, gen_type='bgm')
                elif gen_type == 'mixed':
                    wav_seperate = model.generate_audio(tokens, item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'],chunked=True, gen_type=gen_type)
                else:
                    wav_seperate = model.generate_audio(tokens,chunked=True, gen_type=gen_type)
                del item['raw_pmt_wav']
                del item['raw_vocal_wav']
                del item['raw_bgm_wav']
            else:
                if gen_type == 'separate':
                    wav_vocal = model.generate_audio(tokens, chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(tokens, chunked=True, gen_type='bgm')
                    wav_seperate = model.generate_audio(tokens, chunked=True, gen_type='mixed')
                else:
                    wav_seperate = model.generate_audio(tokens, chunked=True, gen_type=gen_type)
        del item['pmt_wav']
        del item['vocal_wav']
        del item['bgm_wav']
        del item['melody_is_wav']
        end_time = time.time()
        if gen_type == 'separate':
            torchaudio.save(target_wav_name.replace('.flac', '_vocal.flac'), wav_vocal[0].cpu().float(), cfg.sample_rate)
            torchaudio.save(target_wav_name.replace('.flac', '_bgm.flac'), wav_bgm[0].cpu().float(), cfg.sample_rate)
            torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), cfg.sample_rate)
        else:
            torchaudio.save(target_wav_name, wav_seperate[0].cpu().float(), cfg.sample_rate)

        print(f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}")
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
    
    src_jsonl_name = os.path.split(input_jsonl)[-1]
    with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False)+"\n")

def generate_lowmem(args):
    ckpt_path = args.ckpt_path
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    ckpt_path = os.path.join(ckpt_path, 'model.pt')
    cfg = OmegaConf.load(cfg_path)
    cfg.lm.use_flash_attn_2 = args.use_flash_attn
    print(f"use_flash_attn: {args.use_flash_attn}")
    cfg.mode = 'inference'
    max_duration = cfg.max_dur
    gen_type = args.generate_type
    chunk_size = 128
    use_audio_tokenizer = False
    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()
    for line in lines:
        item = json.loads(line)
        if "prompt_audio_path" in item:
            use_audio_tokenizer = True
            break
    if use_audio_tokenizer:
        separator = Separator()
        audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
        audio_tokenizer = audio_tokenizer.eval().cuda()
    auto_prompt = torch.load('ckpt/prompt.pt')
    merge_prompt = [item for sublist in auto_prompt.values() for item in sublist]
    new_items = []
    for line in lines:
        item = json.loads(line)
        target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"
        # get prompt audio
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav, vocal_wav, bgm_wav = separator.run(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            item['raw_vocal_wav'] = vocal_wav
            item['raw_bgm_wav'] = bgm_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if vocal_wav.dim() == 2:
                vocal_wav = vocal_wav[None]
            if vocal_wav.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wav = list(vocal_wav)
            if bgm_wav.dim() == 2:
                bgm_wav = bgm_wav[None]
            if bgm_wav.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wav = list(bgm_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            if type(vocal_wav) == list:
                vocal_wav = torch.stack(vocal_wav, dim=0)
            if type(bgm_wav) == list:
                bgm_wav = torch.stack(bgm_wav, dim=0)
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            if item["auto_prompt_audio_type"] == "Auto": 
                prompt_token = merge_prompt[np.random.randint(0, len(merge_prompt))]
            else:
                prompt_token = auto_prompt[item["auto_prompt_audio_type"]][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True
        item['pmt_wav'] = pmt_wav
        item['vocal_wav'] = vocal_wav
        item['bgm_wav'] = bgm_wav
        item['melody_is_wav'] = melody_is_wav
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
        new_items.append(item)

    if use_audio_tokenizer:
        del audio_tokenizer
        del separator

    torch.cuda.empty_cache()
    
    if "audio_tokenizer_checkpoint_sep" in cfg.keys() and use_audio_tokenizer:
        seperate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg) 
    else:
        seperate_tokenizer = None
    
    if seperate_tokenizer is not None:
        seperate_tokenizer = seperate_tokenizer.eval().cuda()

    for item in new_items:
        if "prompt_audio_path" in item:
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(item['vocal_wav'].cuda(), item['bgm_wav'].cuda())
            item['vocal_wav'] = vocal_wav
            item['bgm_wav'] = bgm_wav

    if use_audio_tokenizer:
        del seperate_tokenizer

    torch.cuda.empty_cache()

    # Define model or load pretrained model
    audiolm = builders.get_lm_model(cfg)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    audiolm = audiolm.eval()

    offload_audiolm = True if 'offload' in cfg.keys() and 'audiolm' in cfg.offload else False
    if offload_audiolm:
        audiolm_offload_param = OffloadParamParse.parse_config(audiolm, cfg.offload.audiolm)
        audiolm_offload_param.show()
        offload_profiler = OffloadProfiler(device_index=0, **(audiolm_offload_param.init_param_dict()))
        offload_profiler.offload_layer(**(audiolm_offload_param.offload_layer_param_dict()))
        offload_profiler.clean_cache_wrapper(**(audiolm_offload_param.clean_cache_param_dict()))
    else:
        audiolm = audiolm.cuda().to(torch.float16)

    model = CodecLM(name = "tmp",
        lm = audiolm,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = None,
    )
    
    cfg_coef = 1.5 #25
    temp = 0.9
    top_k = 50
    top_p = 0.0
    record_tokens = True
    record_window = 50
    

    model.set_generation_params(duration=max_duration, extend_stride=5, temperature=temp, cfg_coef=cfg_coef,
                                top_k=top_k, top_p=top_p, record_tokens=record_tokens, record_window=record_window)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)

    
    for item in new_items:
        lyric = item["gt_lyric"]
        descriptions = item["descriptions"] if "descriptions" in item else None
        pmt_wav = item['pmt_wav']
        vocal_wav = item['vocal_wav']
        bgm_wav = item['bgm_wav']
        melody_is_wav = item['melody_is_wav']
            
        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [descriptions],
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
        item['tokens'] = tokens
    if offload_audiolm:
        offload_profiler.stop()
        del offload_profiler
        del audiolm_offload_param
    del model
    audiolm = audiolm.cpu()
    del audiolm
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    device = "cuda:0"
    seperate_tokenizer.model.device = device
    seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
    seperate_tokenizer.model.model.device = torch.device(device)
    seperate_tokenizer = seperate_tokenizer.eval()

    offload_wav_tokenizer_diffusion =  True if 'offload' in cfg.keys() and 'wav_tokenizer_diffusion' in cfg.offload else False
    if offload_wav_tokenizer_diffusion:
        sep_offload_param = OffloadParamParse.parse_config(seperate_tokenizer, cfg.offload.wav_tokenizer_diffusion)
        sep_offload_param.show()
        sep_offload_profiler = OffloadProfiler(device_index=0, **(sep_offload_param.init_param_dict()))
        sep_offload_profiler.offload_layer(**(sep_offload_param.offload_layer_param_dict()))
        sep_offload_profiler.clean_cache_wrapper(**(sep_offload_param.clean_cache_param_dict()))
    else:
        seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

    model = CodecLM(name = "tmp",
        lm = None,
        audiotokenizer = None,
        max_duration = max_duration,
        seperate_tokenizer = seperate_tokenizer,
    )

    for item in new_items:
        with torch.no_grad():
            if 'raw_pmt_wav' in item:
                if gen_type == 'separate':
                    wav_seperate = model.generate_audio(item['tokens'], item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'],chunked=True, gen_type='mixed')
                    wav_vocal = model.generate_audio(item['tokens'],chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(item['tokens'], chunked=True, gen_type='bgm')
                elif gen_type == 'mixed':
                    wav_seperate = model.generate_audio(item['tokens'], item['raw_pmt_wav'], item['raw_vocal_wav'], item['raw_bgm_wav'],chunked=True, gen_type=gen_type)
                else:
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, gen_type=gen_type)
                del item['raw_pmt_wav']
                del item['raw_vocal_wav']
                del item['raw_bgm_wav']
            else:
                if gen_type == 'separate':
                    wav_vocal = model.generate_audio(item['tokens'], chunked=True, gen_type='vocal')
                    wav_bgm = model.generate_audio(item['tokens'], chunked=True, gen_type='bgm')
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, gen_type='mixed')
                else:
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True, gen_type=gen_type)
        if gen_type == 'separate':
            torchaudio.save(item['wav_path'].replace('.flac', '_vocal.flac'), wav_vocal[0].cpu().float(), cfg.sample_rate)
            torchaudio.save(item['wav_path'].replace('.flac', '_bgm.flac'), wav_bgm[0].cpu().float(), cfg.sample_rate)
            torchaudio.save(item['wav_path'], wav_seperate[0].cpu().float(), cfg.sample_rate)
        else:
            torchaudio.save(item['wav_path'], wav_seperate[0].cpu().float(), cfg.sample_rate)
        del item['tokens']
        del item['pmt_wav']
        del item['vocal_wav']
        del item['bgm_wav']
        del item['melody_is_wav']
        if offload_wav_tokenizer_diffusion:
            sep_offload_profiler.reset_empty_cache_mem_line()
    
    if offload_wav_tokenizer_diffusion:
        sep_offload_profiler.stop()
    torch.cuda.empty_cache()
    src_jsonl_name = os.path.split(input_jsonl)[-1]
    with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    np.random.seed(int(time.time()))
    # 解析命令行参数
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        res_mem = (total - reserved) / 1024 / 1024 / 1024
        print(f"reserved memory: {res_mem}GB")

        model_name = args.ckpt_path.split("/")[-1]
        assert model_name in ['songgeneration_base'], f'{model_name} is not supported, currently only songgeneration_base is supported'
        if model_name == 'songgeneration_base':
            if res_mem > 24 and not args.low_mem:
                print("use generate")
                generate(args)
            else:
                from codeclm.utils.offload_profiler import OffloadProfiler, OffloadParamParse
                print("use generate_lowmem")
                generate_lowmem(args)

    else:
        print("CUDA is not available")
        exit()
    
import glob
import time
import torch
from codeclm.models.codeclm_gen import CodecLM_gen
from codeclm.models import builders
import sys
import os
import torchaudio
import numpy as np
import json
from vllm import LLM, SamplingParams
import re
import argparse
import librosa
auto_prompt_type = ['Pop', 'Latin', 'Rock', 'Electronic', 'Metal', 'Country', 'R&B/Soul', 'Ballad', 'Jazz', 'World', 'Hip-Hop', 'Funk', 'Soundtrack','Auto']


def check_language_by_text(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    english_pattern = re.compile(r'[a-zA-Z]')
    chinese_count = len(re.findall(chinese_pattern, text))
    english_count = len(re.findall(english_pattern, text))
    chinese_ratio = chinese_count / len(text)
    english_ratio = english_count / len(text)
    if chinese_ratio >= 0.2:
        return "zh"
    elif english_ratio >= 0.5:
        return "en"
    else:
        return "en"


def load_audio(f):
    a, fs= librosa.load(f, sr=48000)
    a = torch.tensor(a).unsqueeze(0)
    if (fs != 48000):
        a = torchaudio.functional.resample(a, fs, 48000)
    if a.shape[-1] >= 48000*10:
        a = a[..., :48000*10]
    return a[:, 0:48000*10]


def parse_args():
    parser = argparse.ArgumentParser(description='Song Generation Script')
    
    # 必需参数
    parser.add_argument('--input_jsonl', type=str, required=True,
                      help='Path to input JSONL file containing generation tasks')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory to save generated audio files and results')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                      help='Directory for ckpt')
    return parser.parse_args()


def main():
    torch.set_num_threads(1)
    torch.backends.cudnn.enabled = False #taiji的某些傻呗node会报奇奇怪怪的错
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
    OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(sys.argv[1]))[0])
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
    args = parse_args()
    input_jsonl = args.input_jsonl
    save_dir = args.save_dir
    ckpt_dir = args.ckpt_dir

    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    cfg.mode = 'inference'
    max_duration = cfg.max_dur
    
    audio_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint, cfg)
    if audio_tokenizer is not None:
        for param in audio_tokenizer.parameters():
            param.requires_grad = False
    print("Audio tokenizer successfully loaded!")
    audio_tokenizer = audio_tokenizer.eval().cuda()
    model_condition = CodecLM_gen(cfg=cfg,name = "tmp",audiotokenizer = audio_tokenizer,max_duration = max_duration)
    model_condition.condition_provider.conditioners.load_state_dict(torch.load(cfg.lm_checkpoint+"/conditioners_weights.pth"))
    print('Conditioner successfully loaded!')
    llm = LLM(
        model=cfg.lm_checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=cfg.vllm.device_num,
        enforce_eager=False,              
        dtype="bfloat16",  
        gpu_memory_utilization=cfg.vllm.gpu_memory_utilization, 
        tokenizer=None,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        enable_chunked_prefill=True,
    )
    print("LLM 初始化成功")
    auto_prompt = torch.load('tools/new_auto_prompt.pt')

    guidance_scale = cfg.vllm.guidance_scale
    temp = cfg.vllm.temp
    top_k = cfg.vllm.top_k
    sum_time = 0
    sum_wav_len = 0
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir + "/audios", exist_ok=True)
    os.makedirs(save_dir + "/jsonl", exist_ok=True)
    with open(input_jsonl, "r") as fp:
        lines = fp.readlines()
    new_items = []
    for line in lines:
        item = json.loads(line)
        lyric = item["gt_lyric"]
        descriptions = item["descriptions"].lower() if "descriptions" in item else '.'
        descriptions = '[Musicality-very-high]' + ', ' + descriptions
        target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"
        if os.path.exists(target_wav_name):
            continue
        if "prompt_audio_path" in item:
            assert os.path.exists(item['prompt_audio_path']), f"prompt_audio_path {item['prompt_audio_path']} not found"
            assert 'auto_prompt_audio_type' not in item, f"auto_prompt_audio_type and prompt_audio_path cannot be used together"
            with torch.no_grad():
                pmt_wav = load_audio(item['prompt_audio_path'])
            item['raw_pmt_wav'] = pmt_wav
            if pmt_wav.dim() == 2:
                pmt_wav = pmt_wav[None]
            if pmt_wav.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            pmt_wav = list(pmt_wav)
            if type(pmt_wav) == list:
                pmt_wav = torch.stack(pmt_wav, dim=0)
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav.cuda())
                print(pmt_wav.shape)
            melody_is_wav = False
        elif "auto_prompt_audio_type" in item:
            assert item["auto_prompt_audio_type"] in auto_prompt_type, f"auto_prompt_audio_type {item['auto_prompt_audio_type']} not found"
            lang = check_language_by_text(item['gt_lyric'])
            prompt_token = auto_prompt[item["auto_prompt_audio_type"]][lang][np.random.randint(0, len(auto_prompt[item["auto_prompt_audio_type"]][lang]))]
            pmt_wav = prompt_token[:,[0],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            melody_is_wav = True
        item["idx"] = f"{item['idx']}"
        item["wav_path"] = target_wav_name
        embeded_eosp1 = torch.load(cfg.lm_checkpoint+'/embeded_eosp1.pt')
        generate_inp = {
            'descriptions': [lyric.replace("  ", " ")],
            'type_info': [descriptions],
            'melody_wavs': pmt_wav,
            'melody_is_wav': melody_is_wav,
            'embeded_eosp1': embeded_eosp1,
        }
        fused_input, audio_qt_embs = model_condition.generate_condition(**generate_inp, return_tokens=True)
        prompt_token = audio_qt_embs[0][0].tolist() if audio_qt_embs else []
        allowed_token_ids = [x for x in range(cfg.lm.code_size+1) if x not in prompt_token]
        sampling_params = SamplingParams(
            max_tokens=cfg.audio_tokenizer_frame_rate*cfg.max_dur,
            temperature=temp,
            stop_token_ids=[cfg.lm.code_size],
            top_k=top_k,
            frequency_penalty=0.2,
            seed=int(time.time() * 1000000) % (2**32) if cfg.vllm.cfg else -1,
            allowed_token_ids=allowed_token_ids,
            guidance_scale=guidance_scale
        )
        # 拆成现支持的batch 3 CFG形式
        prompts = [{"prompt_embeds": embed} for embed in fused_input]
        promptss = []
        for _ in range(2):
            promptss+=prompts
        uncondi = prompts[1]
        promptss = promptss[::2] + [uncondi]
        start_time = time.time()
        outputs = llm.generate(promptss, sampling_params=sampling_params)
        mid_time = time.time()
        token_ids_CFG = torch.tensor(outputs[1].outputs[0].token_ids)
        token_ids_CFG = token_ids_CFG[:-1].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # wav_nocfg = model_condition.generate_audio(token_ids)
            if 'raw_pmt_wav' in item:
                wav_cfg = model_condition.generate_audio(token_ids_CFG, item['raw_pmt_wav'])
                del item['raw_pmt_wav']
            else:
                wav_cfg = model_condition.generate_audio(token_ids_CFG)
            end_time = time.time()
            torchaudio.save(target_wav_name, wav_cfg[0].cpu().float(), cfg.sample_rate)
        sum_time += end_time - start_time
        sum_wav_len += (token_ids_CFG.shape[-1] / 25)
        print(f"process{item['idx']}, lm cost {mid_time - start_time}s, diffusion cost {end_time - mid_time}, rtf {(end_time - start_time) / token_ids_CFG.shape[-1] * 25:.2f}")
        new_items.append(item)
    print(f"Total time: {sum_time:.4f} seconds, total wav length: {sum_wav_len:.4f} seconds, rtf {sum_time/sum_wav_len:.2f}")
    
    src_jsonl_name = os.path.split(input_jsonl)[-1]
    with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
        for item in new_items:
            fw.writelines(json.dumps(item, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
    

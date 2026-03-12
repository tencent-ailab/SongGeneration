import sys
import gradio as gr
import json
from datetime import datetime
import yaml
import time
import re
import os.path as op
import torch
import soundfile as sf
import numpy as np
import tempfile
from levo_inference import LeVoInference

EXAMPLE_LYRICS = """
[intro-medium]

[verse]
列车呼啸穿过隧道
窗外风景急速倒退
像是与过去在告别
奔向未定的终点线
心中混杂期待恐惧
请别问我要去哪里
也别追问归期何时

[chorus]
让我随风去流浪
我不想停留原地
原地只有无尽循环
不再规划人生
不再遵循地图
我渴望一场意外之旅
邂逅未知的自己

[inst-short]

[verse]
行李简单只有背囊
装着一颗渴望自由的心
旧照片已撕碎抛洒
不要任何牵绊
不要沉重过往

[chorus]
让我随风去流浪
我不想停留原地
原地只会滋生腐朽
不再规划人生
不再遵循地图
我渴望一场灵魂蜕变
在路途中找到答案

[bridge]
山川湖海皆是导师
星空之下顿悟了渺小
狭隘的悲欢被风吹散
融入天地间的壮阔

[chorus]
它教会我何为生命
何为存在的意义
渺小如尘却也独特
勇敢地写下自己的诗
这是我的远征之路
生命最绚烂的章节

[outro-medium]
""".strip()

APP_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
with open(op.join(APP_DIR, 'conf/vocab.yaml'), 'r', encoding='utf-8') as file:
    STRUCTS = yaml.safe_load(file)


def save_as_flac(sample_rate, audio_data):
    if isinstance(audio_data, tuple):
        sample_rate, audio_data = audio_data
    
    if audio_data.dtype == np.float64:
        audio_data = audio_data.astype(np.float32)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
    sf.write(temp_file, audio_data, sample_rate, format='FLAC')
    return temp_file.name


# 模拟歌曲生成函数
def generate_song(lyric, description=None, prompt_audio=None, genre=None, cfg_coef=None, temperature=0.1, top_k=-1, gen_type="mixed", progress=gr.Progress(track_tqdm=True)):
    global MODEL
    global STRUCTS
    params = {'cfg_coef':cfg_coef, 'temperature':temperature, 'top_k':top_k}
    params = {k:v for k,v in params.items() if v is not None}
    vocal_structs = ['[verse]', '[chorus]', '[bridge]']
    sample_rate = MODEL.cfg.sample_rate
    
    # format lyric
    lyric = lyric.replace("[intro]", "[intro-short]").replace("[inst]", "[inst-short]").replace("[outro]", "[outro-short]")
    paragraphs = [p.strip() for p in lyric.strip().split('\n\n') if p.strip()]
    if len(paragraphs) < 1:
        return None, json.dumps("Lyrics can not be left blank")
    paragraphs_norm = []
    vocal_flag = False
    for para in paragraphs:
        lines = para.splitlines()
        struct_tag = lines[0].strip().lower()
        if struct_tag not in STRUCTS:
            return None, json.dumps(f"Segments should start with a structure tag in {STRUCTS}")
        if struct_tag in vocal_structs:
            vocal_flag = True
            if len(lines) < 2 or not [line.strip() for line in lines[1:] if line.strip()]:
                return None, json.dumps("The following segments require lyrics: [verse], [chorus], [bridge]")
            else:
                new_para_list = []
                for line in lines[1:]:
                    new_para_list.append(re.sub(r"[^\w\s\[\]\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]", "", line))
                new_para_str = f"{struct_tag} {'.'.join(new_para_list)}"
        else:
            if len(lines) > 1:
                return None, json.dumps("The following segments should not contain lyrics: [intro], [intro-short], [intro-medium], [inst], [inst-short], [inst-medium], [outro], [outro-short], [outro-medium]")
            else:
                new_para_str = struct_tag
        paragraphs_norm.append(new_para_str)
    if not vocal_flag:
        return None, json.dumps(f"The lyric must contain at least one of the following structures: {vocal_structs}")
    lyric_norm = " ; ".join(paragraphs_norm)

    # format prompt 
    if prompt_audio is not None:
        genre = None
        description = None
    elif description is not None and description != "":
        genre = None
        if description[-1] != ".":
            description = description + "."

    progress(0.0, "Start Generation")
    start = time.time()
    
    audio_data = MODEL(lyric_norm, description, prompt_audio, genre, op.join(APP_DIR, "tools/new_auto_prompt.pt"), gen_type, params).cpu().permute(1, 0).float().numpy()

    end = time.time()
    
    # 创建输入配置的JSON
    input_config = {
        "lyric": lyric_norm,
        "genre": genre,
        "prompt_audio": prompt_audio,
        "description": description,
        "params": params,
        "inference_duration": end - start,
        "timestamp": datetime.now().isoformat(),
    }
    
    filepath = save_as_flac(sample_rate, audio_data)
    return filepath, json.dumps(input_config, indent=2)


# 创建Gradio界面
with gr.Blocks(title="SongGeneration Demo Space") as demo:
    gr.Markdown("# 🎵 SongGeneration Demo Space")
    gr.Markdown("Push to Levo 2.0 — faster and more controllable. The code is in [GIT](https://github.com/tencent-ailab/SongGeneration)")
    
    with gr.Row():
        with gr.Column():
            lyric = gr.Textbox(
                label="Lyrics",
                lines=5,
                max_lines=15,
                value=EXAMPLE_LYRICS,
                info="Each paragraph represents a segment starting with a structure tag and ending with a blank line, each line is a sentence without punctuation, segments [intro], [inst], [outro] should not contain lyrics, while [verse], [chorus], and [bridge] require lyrics.",
                placeholder="""Lyric Format
'''
[structure tag]
lyrics

[structure tag]
lyrics
'''
1. One paragraph represents one segments, starting with a structure tag and ending with a blank line
2. One line represents one sentence, punctuation is not recommended inside the sentence
3. The following segments should not contain lyrics: [intro-short], [intro-medium], [inst-short], [inst-medium], [outro-short], [outro-medium]
4. The following segments require lyrics: [verse], [chorus], [bridge]
"""
            )

            with gr.Tabs(elem_id="extra-tabs"):
                with gr.Tab("Genre Select"):
                    genre = gr.Radio(
                        choices=["Auto", "Pop", "Latin", "Rock", "Electronic", "Metal", "Country", "R&B/Soul", "Ballad", "Jazz", "World", "Hip-Hop", "Funk", "Soundtrack"],
                        label="Genre Select(Optional)",
                        value="Auto",
                        interactive=True,
                        elem_id="single-select-radio"
                    )
                with gr.Tab("Text Prompt"):
                    gr.Markdown("For detailed usage, please refer to [here](https://github.com/tencent-ailab/SongGeneration?tab=readme-ov-file#-description-input-format)")
                    description = gr.Textbox(
                        label="Song Description (Optional)",
                        info="Describe the gender, genre, emotion, and instrument. Only English is supported currently.​",
                        placeholder="female, rock, motivational, electric guitar, bass guitar, drum kit.",
                        lines=1,
                        max_lines=2
                    )
                with gr.Tab("Audio Prompt"):
                    prompt_audio = gr.Audio(
                        label="Prompt Audio (Optional)",
                        type="filepath",
                        elem_id="audio-prompt"
                    )

            with gr.Accordion("Advanced Config", open=False):
                cfg_coef = gr.Slider(
                    label="CFG Coefficient",
                    minimum=0.1,
                    maximum=3.0,
                    step=0.1,
                    value=1.8,
                    interactive=True,
                    elem_id="cfg-coef",
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    step=0.1,
                    value=0.8,
                    interactive=True,
                    elem_id="temperature",
                )
                # top_k = gr.Slider(
                #     label="Top-K",
                #     minimum=1,
                #     maximum=100,
                #     step=1,
                #     value=50,
                #     interactive=True,
                #     elem_id="top_k",
                # )
            with gr.Row():
                generate_btn = gr.Button("Generate Song", variant="primary")
                # generate_bgm_btn = gr.Button("Generate Pure Music", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Song", type="filepath")
            output_json = gr.JSON(label="Generated Info")
    
        # # 示例按钮
        # examples = gr.Examples(
        #     examples=[
        #         ["male, bright, rock, happy, electric guitar and drums, the bpm is 150."],
        #         ["female, warm, jazz, romantic, synthesizer and piano, the bpm is 100."]
        #     ],
        #     inputs=[description],
        #     label="Text Prompt examples"
        # )

        # examples = gr.Examples(
        #     examples=[
        #     "[intro-medium]\n\n[verse]\n在这个疯狂的世界里\n谁不渴望一点改变\n在爱情面前\n我们都显得那么不安全\n你紧紧抱着我\n告诉我再靠近一点\n别让这璀璨的夜晚白白浪费\n我那迷茫的眼睛\n看不见未来的路\n在情感消散之前\n我们对爱的渴望永不熄灭\n你给我留下一句誓言\n想知道我们的爱是否能持续到永远\n[chorus]\n\n约定在那最后的夜晚\n不管命运如何摆布\n我们的心是否依然如初\n我会穿上红衬衫\n带着摇滚的激情\n回到我们初遇的地方\n约定在那最后的夜晚\n就算全世界都变了样\n我依然坚守诺言\n铭记这一天\n你永远是我心中的爱恋\n\n[outro-medium]\n",
        #     "[intro-short]\n\n[verse]\nThrough emerald canyons where fireflies dwell\nCerulean berries kiss morning's first swell\nCrystalline dew crowns each Vitamin Dawn's confection dissolves slowly on me\nAmbrosia breezes through honeycomb vines\nNature's own candy in Fibonacci lines\n[chorus] Blueberry fruit so sweet\n takes you higher\n can't be beat\n In your lungs\n it starts to swell\n You're under its spell\n [verse] Resin of sunlight in candied retreat\nMarmalade moonbeams melt under bare feet\nNectar spirals bloom chloroplast champagne\nPhotosynthesis sings through my veins\nChlorophyll rhythms pulse warm in my blood\nThe forest's green pharmacy floods every bud[chorus] Blueberry fruit so sweet\n takes you higher\n can't be beat\n In your lungs\n it starts to swell\n You're under its spell\n feel the buzz\n ride the wave\n Limey me\n blueberry\n your mind's enslaved\n In the haze\n lose all time\n floating free\n feeling fine\n Blueberry\n fruit so sweet\n takes you higher\n can't be beat\n In your lungs\n it starts to swell\n cry\n You're under its spell\n\n[outro-short]\n",
        #     ],
        #     inputs=[lyric],
        #     label="Lyrics examples",
        # )
    
    # 生成按钮点击事件
    generate_btn.click(
        fn=generate_song,
        inputs=[lyric, description, prompt_audio, genre, cfg_coef, temperature, gr.State(5000)],
        outputs=[output_audio, output_json]
    )
    # generate_bgm_btn.click(
    #     fn=generate_song,
    #     inputs=[lyric, description, prompt_audio, genre, cfg_coef, temperature, gr.State(50), gr.State("bgm")],
    #     outputs=[output_audio, output_json]
    # )
    

# 启动应用
if __name__ == "__main__":
    torch.set_num_threads(1)
    MODEL = LeVoInference(sys.argv[1])
    demo.launch(server_name="0.0.0.0", server_port=8081)

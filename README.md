# SongGeneration

<p align="center"><img src="img/logo.jpg" width="40%"></p>
<p align="center">
    <a href="https://levo-demo.github.io/">Demo</a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2506.07520">Paper</a>  &nbsp;|&nbsp; <a href="https://huggingface.co/waytan22/SongGeneration">Hugging Face</a>  &nbsp;|&nbsp; <a href="https://huggingface.co/spaces/waytan22/SongGeneration-LeVo">Space Demo</a>
</p>



This repository is the official repository for “LeVo: High-Quality Song Generation with Multi-Preference Alignment” (NeurIPS 2025). In this repository, we provide the SongGeneration model, inference scripts,pretrained checkpoints, and some music generation tools.

## News and Updates

* **2025.10.15🔥**:  We have updated the codebase to improve **inference speed** and **generation quality**, and adapted it to the **latest model version**. Please **update to the newest code** to ensure the **best performance and user experience**.
* **2025.10.14 🔥**: We have released the **large model (SongGeneration-large)**.
* **2025.10.13 🔥**: We have released the **full time model (SongGeneration-base-full)** and **evaluation performance**.
* **2025.10.12 🔥**: We have released the **english enhanced model (SongGeneration-base-new)**.
* **2025.09.23 🔥**: We have released the [Data Processing Pipeline](https://github.com/tencent-ailab/SongPrep), which is capable of **analyzing the structure and lyrics** of entire songs and **providing precise timestamps** without the need for additional source separation. On the human-annotated test set [SSLD-200](https://huggingface.co/datasets/waytan22/SSLD-200), the model’s performance outperforms mainstream models including Gemini-2.5, Seed-ASR, and Qwen3-ASR.
* **2025.07.25 🔥**: SongGeneration can now run with as little as **10GB of GPU memory**.
* **2025.07.18 🔥**: SongGeneration now supports generation of **pure music**, **pure vocals**, and **dual-track (vocals + accompaniment separately)** outputs.
* **2025.06.16 🔥**: We have released the **SongGeneration** series.

## TODOs📋

- [ ] Release SongGeneration-v1.5 (trained on a larger multilingual dataset, supports more languages, and integrates a Reward Model with Reinforcement Learning to enhance musicality and lyric alignment)
- [ ] Release finetuning scripts.
- [ ] Release Music Codec and VAE.
- [x] Release large model.
- [x] Release full time model.
- [x] Release English enhanced model.
- [x] Release data processing pipeline.
- [x] Update Low memory usage model.
- [x] Support single vocal/bgm track generation.

## Model Versions

| Model                     | Max Length |       Language       | GPU Menmory | RFT(A100) | Download Link                                                |
| ------------------------- | :--------: | :------------------: | :---------: | :-------: | ------------------------------------------------------------ |
| SongGeneration-base       |   2m30s    |          zh          |   10G/16G   |   1.26    | [Huggingface](https://huggingface.co/tencent/SongGeneration/tree/main/ckpt/songgeneration_base) |
| SongGeneration-base-new   |   2m30s    |        zh, en        |   10G/16G   |   1.26    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-new) |
| SongGeneration-base-full  |   4m30s    |        zh, en        |   12G/18G   |   1.30    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-full) |
| SongGeneration-large      |   4m30s    |        zh, en        |   22G/28G   |   1.51    | [Huggingface](https://huggingface.co/lglg666/SongGeneration-large) |
| SongGeneration-v1.5-small |     2m     | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |
| SongGeneration-v1.5-base  |   4m30s    | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |
| SongGeneration-v1.5-large |   4m30s    | zh, en, es, ja, etc. |      -      |     -     | Coming soon                                                  |

💡 **Notes:**

- **GPU Memory** — “X / Y” means X: no prompt audio; Y: with prompt audio.
- **RFT** — Real Forward Time (pure inference, excluding model loading).

## Overview

We develop the SongGeneration model. It is an LM-based framework consisting of **LeLM** and a **music codec**. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. The music codec reconstructs the dual-track tokens into highfidelity music audio. SongGeneration significantly improves over the open-source music generation models and performs competitively with current state-of-the-art industry systems. For more details, please refer to our [paper](https://arxiv.org/abs/2506.07520).

<img src="img/over.jpg" alt="img" style="zoom:100%;" /> 

## Installation

### Start from scratch

You can install the necessary dependencies using the `requirements.txt` file with Python>=3.8.12 and CUDA>=11.8:

```bash
pip install -r requirements.txt
pip install -r requirements_nodeps.txt --no-deps
```

**(Optional)** Then install flash attention from git. For example, if you're using Python 3.10 and CUDA 12.0

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Start with docker

```bash
docker pull juhayna/song-generation-levo:hf0613
docker run -it --gpus all --network=host juhayna/song-generation-levo:hf0613 /bin/bash
```

### Other deploy examples 

 - Windows platform with ComfyUI: https://github.com/smthemex/ComfyUI_SongGeneration
 - Windows installer: http://bilibili.com/video/BV1ATK8zQE8L/?vd_source=22cfc54298226c4161b1aff457d17585
 - Quick start with ComfyUI on CNB: https://cnb.cool/tencent/tencent-ailab/examples/SongGeneration-comfyui

## Inference

To ensure the model runs correctly, **please download all the required folders** from the original source at [Hugging Face](https://huggingface.co/collections/lglg666/levo-68d0c3031c370cbfadade126).

- Download `ckpt` and `third_party` folder from [Hugging Face](https://huggingface.co/lglg666/SongGeneration-Runtime/tree/main) or  [Hugging Face](https://huggingface.co/tencent/SongGeneration/tree/main), and move them into the **root directory** of the project. You can also download models using hugging face-cli.

  ```
  huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./runtime
  mv runtime/ckpt ckpt
  mv runtime/third_party third_party
  ```

- Download the specific model checkpoint and save it to your specified checkpoint directory: `ckpt_path` (We provide multiple versions of model checkpoints. Please select the most suitable version based on your needs and download the corresponding file. Also, ensure the folder name matches the model version name.) Your can also download models using hugging face-cli.

  ```
  # download SongGeneration-base
  huggingface-cli download lglg666/SongGeneration-base --local-dir ./songgeneration_base
  # download SongGeneration-base-new
  huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./songgeneration_base_new
  # download SongGeneration-base-full
  huggingface-cli download lglg666/SongGeneration-base-full --local-dir ./songgeneration_base_full
  # download SongGeneration-large
  huggingface-cli download lglg666/SongGeneration-large --local-dir ./songgeneration_large
  ```

Once everything is set up, you can run the inference script using the following command:

```bash
sh generate.sh ckpt_path lyrics.jsonl output_path
```

- You may provides sample inputs in JSON Lines (`.jsonl`) format. Each line represents an individual song generation request. The model expects each input to contain the following fields:

  - `idx`: A unique identifier for the output song. It will be used as the name of the generated audio file.
  - `gt_lyric`:The lyrics to be used in generation. It must follow the format of `[Structure] Text`, where `Structure` defines the musical section (e.g., `[Verse]`, `[Chorus]`). See Input Guide.
  - `descriptions` : (Optional) You may customize the text prompt to guide the model’s generation. This can include attributes like gender, timbre, genre, emotion, instrument, and BPM. See Input Guide.
  - `prompt_audio_path`: (Optional) Path to a 10-second reference audio file. If provided, the model will generate a new song in a similar style to the given reference.

  - `auto_prompt_audio_type`: (Optional) Used only if `prompt_audio_path` is not provided. This allows the model to automatically select a reference audio from a predefined library based on a given style. Supported values include:
    - `'Pop'`, `'R&B'`, `'Dance'`, `'Jazz'`, `'Folk'`, `'Rock'`,`'Chinese Style'`, `'Chinese Tradition'`, `'Metal'`, `'Reggae'`, `'Chinese Opera'`, `'Auto'`.
  - **Note:** If certain optional fields are not required, they can be omitted. 

- Outputs of the loader `output_path`:

  - `audio`: generated audio files
  - `jsonl`: output jsonls

- An example command may look like:

  ```bash
  sh generate.sh songgeneration_base sample/lyrics.jsonl sample/output
  ```

If you encounter **out-of-memory (OOM**) issues, you can manually enable low-memory inference mode using the `--low_mem` flag. For example:

```bash
sh generate.sh ckpt_path lyrics.jsonl output_path --low_mem
```

If your GPU device does **not support Flash Attention** or your environment does **not have Flash Attention installed**, you can disable it by adding the `--not_use_flash_attn` flag. For example:

```bash
sh generate.sh ckpt_path lyrics.jsonl output_path --not_use_flash_attn
```

By default, the model generates **songs with both vocals and accompaniment**. If you want to generate **pure music**, **pure vocals**, or **separated vocal and accompaniment tracks**, please use the following flags:

- `--bgm`  Generate **pure music**
- `--vocal` Generate **vocal-only (a cappella)**
- `--separate` Generate **separated vocal and accompaniment tracks**

For example:

```bash
sh generate.sh ckpt_path lyrics.jsonl output_path --separate
```

## Input Guide

An example input file can be found in `sample/lyrics.jsonl` 

### 🎵 Lyrics Input Format

The `gt_lyric` field defines the lyrics and structure of the song. It consists of multiple musical section, each starting with a structure label. The model uses these labels to guide the musical and lyrical progression of the generated song.

#### 📌 Structure Labels

- The following segments **should not** contain lyrics (they are purely instrumental):

  - `[intro-short]`, `[intro-medium]`, `[inst-short]`, `[inst-medium]`, `[outro-short]`, `[outro-medium]`

  > - `short` indicates a segment of approximately 0–10 seconds
  > - `medium` indicates a segment of approximately 10–20 seconds
  > - We find that [inst] label is less stable, so we recommend that you do not use it.

- The following segments **require lyrics**:

  - `[verse]`, `[chorus]`, `[bridge]`

#### 🧾 Lyrics Formatting Rules

- Each section is **separated by ` ; `**

- Within lyrical segments (`[verse]`, `[chorus]`, `[bridge]`), lyrics must be written in complete sentences and separated by a period (`.`)

- A complete lyric string may look like:

  ```
  [intro-short] ; [verse] These faded memories of us. I can't erase the tears you cried before. Unchained this heart to find its way. My peace won't beg you to stay ; [bridge] If ever your truth still remains. Turn around and see. Life rearranged its games. All these lessons in mistakes. Even years may never erase ; [inst-short] ; [chorus] Like a fool begs for supper. I find myself waiting for her. Only to find the broken pieces of my heart. That was needed for my soul to love again ; [outro-short]
  ```

- More examples can be found in `sample/test_en_input.jsonl` and `sample/test_zh_input.jsonl`.

### 📝 Description Input Format

The `descriptions` field allows you to control various musical attributes of the generated song. It can describe up to six musical dimensions: **Gender** (e.g., male, female), **Timbre** (e.g., dark, bright, soft), **Genre** (e.g., pop, jazz, rock), **Emotion** (e.g., sad, energetic, romantic), **Instrument** (e.g., piano, drums, guitar), **BPM** (e.g., the bpm is 120). 

- All six dimensions are optional — you can specify any subset of them.

- The order of dimensions is flexible.

- Use **commas (`,`)** to separate different attributes.

- Although the model supports open vocabulary, we recommend using predefined tags for more stable and reliable performance. A list of commonly supported tags for each dimension is available in the `sample/description/` folder.

- Here are a few valid `descriptions` inputs:

  ```
  - female, dark, pop, sad, piano and drums.
  - male, piano, jazz.
  - male, dark, the bpm is 110.
  ```

### 🎧Prompt Audio Usage Notes

- The input audio file can be longer than 10 seconds, but only the first 10 seconds will be used.
- For best musicality and structure, it is recommended to use the chorus section of a song as the prompt audio.
- You can use this field to influence genre, instrumentation, rhythm, and voice

#### ⚠️ Important Considerations

- **Avoid providing both `prompt_audio_path` and `descriptions` at the same time.**
  If both are present, and they convey conflicting information, the model may struggle to follow instructions accurately, resulting in degraded generation quality.
- If `prompt_audio_path` is not provided, you can instead use `auto_prompt_audio_type` for automatic reference selection.

## Gradio UI

You can start up the UI with the following command:

```bash
sh tools/gradio/run.sh ckpt_path
```

## Evaluation Performance

### Chinese

 <table>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Open-Source</th>
    <th rowspan="2">PER↓</th>
    <th colspan="4" style="text-align:center;">Audiobox Aesthetics ↑</th>
    <th colspan="5" style="text-align:center;">SongEval ↑</th>
  </tr>
  <tr>
    <th>CE</th><th>CU</th><th>PC</th><th>PQ</th>
    <th>COH</th><th>MUS</th><th>MEM</th><th>CLA</th><th>NAT</th>
  </tr>
  <tr>
    <td>Suno</td>
    <td>❌</td>
    <td>21.6%</td>
    <td>7.65</td><td>7.86</td><td>5.94</td><td>8.35</td>
    <td><b>4.41</b></td><td><b>4.34</b></td><td><b>4.44</b></td><td><b>4.38</b></td><td><b>4.26</b></td>
  </tr>
  <tr>
    <td>Mureka</td>
    <td>❌</td>
    <td>7.2%</td>
    <td>7.71</td><td>7.83</td><td><b>6.39</b></td><td><b>8.44</b></td>
    <td>4.01</td><td>3.85</td><td>3.73</td><td>3.87</td><td>3.75</td>
  </tr>
  <tr>
    <td>Haimian</td>
    <td>❌</td>
    <td>11.8%</td>
    <td>7.56</td><td>7.85</td><td>5.89</td><td>8.27</td>
    <td>3.69</td><td>3.43</td><td>3.51</td><td>3.52</td><td>3.34</td>
  </tr>
  <tr>
    <td>ACE-Step</td>
    <td>✅</td>
    <td>37.1%</td>
    <td>7.37</td><td>7.52</td><td><b>6.26</b></td><td>7.85</td>
    <td>3.68</td><td>3.45</td><td>3.54</td><td>3.48</td><td>3.38</td>
  </tr>
  <tr>
    <td>Diffrhythm-v1,2</td>
    <td>✅</td>
    <td>8.78%</td>
    <td>6.91</td><td>7.45</td><td>5.45</td><td>7.99</td>
    <td>2.93</td><td>2.60</td><td>2.70</td><td>2.71</td><td>2.60</td>
  </tr>
  <tr>
    <td>YUE</td>
    <td>✅</td>
    <td>14.9%</td>
    <td>7.29</td><td>7.53</td><td>6.19</td><td>7.96</td>
    <td>3.68</td><td>3.43</td><td>3.49</td><td>3.49</td><td>3.42</td>
  </tr>
  <tr>
    <td>SongGeneration-base</td>
    <td>✅</td>
    <td>7.2%</td>
    <td>7.78</td><td>7.90</td><td>6.03</td><td>8.42</td>
    <td>3.96</td><td>3.80</td><td>3.85</td><td>3.74</td><td>3.71</td>
  </tr>
  <tr>
    <td>SongGeneration-base-new</td>
    <td>✅</td>
    <td><b>5.7%</b></td>
    <td><b>7.82</b></td><td><b>7.94</b></td><td>6.07</td><td>8.43</td>
    <td>4.07</td><td>3.92</td><td>3.98</td><td>3.93</td><td>3.86</td>
  </tr>
  <tr>
    <td>SongGeneration-base-full</td>
    <td>✅</td>
    <td>8.4%</td>
    <td><b>7.81</b></td><td><b>7.94</b></td><td>6.07</td><td>8.41</td>
    <td>4.02</td><td>3.88</td><td>3.94</td><td>3.87</td><td>3.80</td>
  </tr>
  <tr>
    <td>SongGeneration-large</td>
    <td>✅</td>
    <td><b>5.1%</b></td>
    <td><b>7.82</b></td><td><b>7.95</b></td><td>6.09</td><td><b>8.46</b></td>
    <td><b>4.08</b></td><td><b>3.94</b></td><td><b>4.00</b></td><td><b>3.94</b></td><td><b>3.87</b></td>
  </tr>
</table>

### English

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Open-Source</th>
    <th rowspan="2">PER↓</th>
    <th colspan="4" style="text-align:center;">Audiobox Aesthetics ↑</th>
    <th colspan="5" style="text-align:center;">SongEval ↑</th>
  </tr>
  <tr>
    <th>CE</th><th>CU</th><th>PC</th><th>PQ</th>
    <th>COH</th><th>MUS</th><th>MEM</th><th>CLA</th><th>NAT</th>
  </tr>
  <tr>
    <td>Suno</td>
    <td>❌</td>
    <td>15.6%</td>
    <td>7.64</td><td>7.85</td><td>5.84</td><td>8.19</td>
    <td><b>4.49</b></td><td><b>4.35</b></td><td><b>4.47</b></td><td><b>4.35</b></td><td><b>4.23</b></td>
  </tr>
  <tr>
    <td>Mureka</td>
    <td>❌</td>
    <td><b>12.6%</b></td>
    <td>7.71</td><td>7.93</td><td><b>6.46</b></td><td>8.39</td>
    <td>4.06</td><td>3.88</td><td>3.90</td><td>3.90</td><td>3.73</td>
  </tr>
  <tr>
    <td>Haimian</td>
    <td>❌</td>
    <td>26.6%</td>
    <td><b>7.85</b></td><td><b>8.01</b></td><td>5.28</td><td><b>8.44</b></td>
    <td>3.83</td><td>3.68</td><td>3.71</td><td>3.61</td><td>3.45</td>
  </tr>
  <tr>
    <td>ACE-Step</td>
    <td>✅</td>
    <td>32.1%</td>
    <td>7.19</td><td>7.37</td><td>6.16</td><td>7.57</td>
    <td>3.59</td><td>3.34</td><td>3.43</td><td>3.36</td><td>3.27</td>
  </tr>
  <tr>
    <td>Diffrhythm-v1.2</td>
    <td>✅</td>
    <td>17.8%</td>
    <td>7.02</td><td>7.58</td><td>5.96</td><td>7.81</td>
    <td>3.51</td><td>3.12</td><td>3.32</td><td>3.21</td><td>3.08</td>
  </tr>
  <tr>
    <td>YUE</td>
    <td>✅</td>
    <td>27.3%</td>
    <td>7.04</td><td>7.22</td><td>5.89</td><td>7.67</td>
    <td>3.58</td><td>3.24</td><td>3.42</td><td>3.37</td><td>3.30</td>
  </tr>
  <tr>
    <td>SongGeneration-base</td>
    <td>✅</td>
    <td>-</td>
    <td>-</td><td>-</td><td>-</td><td>-</td>
    <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr>
  <tr>
    <td>SongGeneration-base-new</td>
    <td>✅</td>
    <td>16.2%</td>
    <td><b>7.78</b></td><td>7.97</td><td>6.03</td><td>8.37</td>
    <td>4.05</td><td>3.90</td><td>3.99</td><td>3.91</td><td>3.79</td>
  </tr>
  <tr>
    <td>SongGeneration-base-full</td>
    <td>✅</td>
    <td>20.1%</td>
    <td>7.76</td><td>7.98</td><td>5.96</td><td>8.39</td>
    <td>4.02</td><td>3.87</td><td>3.97</td><td>3.86</td><td>3.74</td>
  </tr>
  <tr>
    <td>SongGeneration-large</td>
    <td>✅</td>
    <td><b>14.9%</b></td>
    <td><b>7.85</b></td><td><b>8.05</b></td><td><b>6.17</b></td><td><b>8.46</b></td>
    <td><b>4.08</b></td><td><b>3.94</b></td><td><b>4.03</b></td><td><b>3.93</b></td><td><b>3.82</b></td>
  </tr>
</table>

### Notes

1. The evaluation results of SongGeneration are based on **200 generated songs**, including **100 using descriptions** and **100 using `auto_prompt_audio_type=Auto`**. We also provide **40 English** and **40 Chinese** example inputs in
    `sample/test_en_input.jsonl` and `sample/test_zh_input.jsonl` for reference.
2. Since the model attempts to clone the timbre and musical style of the given prompt audio, the choice of prompt audio can significantly affect generation performance, and may lead to fluctuations in the evaluation metrics.
3. The format of the input lyrics has a strong impact on generation quality. If the output quality appears suboptimal, please check whether your lyrics format is correct. You can find more examples of properly formatted inputs in `sample/test_en_input.jsonl` and `sample/test_zh_input.jsonl`.

## Citation

```
@article{lei2025levo,
  title={LeVo: High-Quality Song Generation with Multi-Preference Alignment},
  author={Lei, Shun and Xu, Yaoxun and Lin, Zhiwei and Zhang, Huaicheng and Tan, Wei and Chen, Hangting and Yu, Jianwei and Zhang, Yixuan and Yang, Chenyu and Zhu, Haina and Wang, Shuai and Wu, Zhiyong and Yu, Dong},
  journal={arXiv preprint arXiv:2506.07520},
  year={2025}
}
```

## License

The code and weights in this repository is released in the [LICENSE](LICENSE)  file.


## Contact

Use WeChat or QQ to scan blow QR code.

<div style="display: flex; justify-content: center; gap: 20px; width: 100%;">
  <img src="img/contact.png" height="300" />
  <img src="img/contactQQ.jpg" height="300" />
</div>

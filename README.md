# SongGeneration 2

<p align="center"><img src="img/logo.jpg" width="40%"></p>

#### SongGeneration

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://github.com/tencent-ailab/songgeneration)[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/tencent/SongGeneration) [![Technical Report](https://img.shields.io/badge/Technical%20Report-Arxiv-red)](https://arxiv.org/abs/2506.07520) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://levo-demo.github.io/)

#### SongGeneration 2

[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)](https://github.com/tencent-ailab/songgeneration)[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)![Live Playground](https://img.shields.io/badge/Live%20PlayGround-Demo-orange)](https://huggingface.co/spaces/waytan22/SongGeneration-LeVo) [![Samples](https://img.shields.io/badge/Audio%20Samples-Page-green)](https://levo-demo.github.io/levo_v2_demo/)

🚀 We introduce LeVo 2 (SongGeneration 2), an open-source music foundation model designed to shatter the ceiling of open-source AI music by achieving true commercial-grade generation. 

Through a large-scale, rigorous expert evaluation (20 industry professionals, 6 core dimensions, 100 songs per model), LeVo 2 has proven its superiority:

- 🏆 Commercial-Grade Musicality: Comprehensively outperforms all open-source baselines across Overall Quality, Melody, Arrangement, Sound Quality, and Structure. Its subjective generation quality successfully rivals top-tier closed-source commercial systems (e.g., MiniMax 2.5).
- 🎯 Precise Lyric Accuracy: Achieves an outstanding Phoneme Error Rate (PER) of 8.55%, effectively solving the lyrical hallucination problem. This remarkable accuracy significantly outperforms top commercial models like Suno v5 (12.4%) and Mureka v8 (9.96%).
- 🎛️ Exceptional Controllability: Highly responsive to multi-modal instructions, including text descriptions and audio prompts, allowing for precise control over the generated music.

📊 *For detailed experimental setups and comprehensive metrics, please refer to the [Evaluation Performance](#Evaluation-Performance) section below or our upcoming technical report.*

## News and Updates

* **2026.03.01 🚀**: We proudly introduce SongGeneration 2! We have officially open-sourced the [**SongGeneration-v2-large**](https://huggingface.co/lglg666/SongGeneration-v2-large) (4B parameters) model. It achieves **commercial-grade music generation** with an **outstanding PER of 8.55%** and supports multi-lingual lyrics. Please **update to the newest code** to ensure **optimal performance and user experience**. We also launch the SongGeneration-v2-Fast version on [**Hugging Face Space**](https://huggingface.co/spaces/tencent/SongGeneration)! You can now generate a complete song in under 1 minute, trading a slight loss in musicality for significantly faster generation speed.
* **2025.10.16 🔥**: Our [**Demo webpage**](https://huggingface.co/spaces/tencent/SongGeneration) now supports **full-length song generation (up to 4m30s)**! 🎶  Experience end-to-end music generation with vocals and accompaniment — try it out now!
* **2025.10.15 🔥**: We have updated the codebase to improve **inference speed** and **generation quality**, and adapted it to the **latest model version**. Please **update to the newest code** to ensure the **best performance and user experience**.
* **2025.10.14 🔥**: We have released the **large model (SongGeneration-large)**.
* **2025.10.13 🔥**: We have released the **full time model (SongGeneration-base-full)** and **evaluation performance**.
* **2025.10.12 🔥**: We have released the **english enhanced model (SongGeneration-base-new)**.
* **2025.09.23 🔥**: We have released the [Data Processing Pipeline](https://github.com/tencent-ailab/SongPrep), which is capable of **analyzing the structure and lyrics** of entire songs and **providing precise timestamps** without the need for additional source separation. On the human-annotated test set [SSLD-200](https://huggingface.co/datasets/waytan22/SSLD-200), the model’s performance outperforms mainstream models including Gemini-2.5, Seed-ASR, and Qwen3-ASR.
* **2025.07.25 🔥**: SongGeneration can now run with as little as **10GB of GPU memory**.
* **2025.07.18 🔥**: SongGeneration now supports generation of **pure music**, **pure vocals**, and **dual-track (vocals + accompaniment separately)** outputs.
* **2025.06.16 🔥**: We have released the **SongGeneration** series.

## TODOs📋

- [ ] Release the Automated Music Aesthetic Evaluation Framework.
- [ ] Release finetuning scripts.
- [ ] Release Music Codec and VAE.
- [ ] Release SongGeneration-v2-fast.
- [ ] Release SongGeneration-v2-medium.
- [x] Release SongGeneration-v2-large.
- [x] Release large model.
- [x] Release full time model.
- [x] Release English enhanced model.
- [x] Release data processing pipeline.
- [x] Update Low memory usage model.
- [x] Support single vocal/bgm track generation.

## Model Versions

| Model                    | Max Length |       Language       | GPU Memory | RTF(H20) | Download Link                                                |
| ------------------------ | :--------: | :------------------: | :--------: | :------: | ------------------------------------------------------------ |
| SongGeneration-base      |   2m30s    |          zh          |  10G/16G   |   0.67   | [Huggingface](https://huggingface.co/tencent/SongGeneration/tree/main/ckpt/songgeneration_base) |
| SongGeneration-base-new  |   2m30s    |        zh, en        |  10G/16G   |   0.67   | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-new) |
| SongGeneration-base-full |   4m30s    |        zh, en        |  12G/18G   |   0.69   | [Huggingface](https://huggingface.co/lglg666/SongGeneration-base-full) |
| SongGeneration-large     |   4m30s    |        zh, en        |  22G/28G   |   0.82   | [Huggingface](https://huggingface.co/lglg666/SongGeneration-large) |
| SongGeneration-v2-large  |   4m30s    | zh, en, es, ja, etc. |  22G/28G   |   0.82   | [Huggingface](https://huggingface.co/lglg666/SongGeneration-v2-large) |
| SongGeneration-v2-medium |   4m30s    | zh, en, es, ja, etc. |  12G/18G   |   0.69   | Coming soon                                                  |
| SongGeneration-v2-fast   |   4m30s    | zh, en, es, ja, etc. |     -      |    -     | Coming soon                                                  |

💡 **Notes:**

- **GPU Memory** — “X / Y” means X: no prompt audio; Y: with prompt audio.
- **RTF** — Real Time Factor (pure inference, excluding model loading).

## Overview

<img src="../../Library/CloudStorage/OneDrive-%E4%B8%AA%E4%BA%BA/%E5%9B%BE%E7%89%87/over.jpg" alt="img" style="zoom:100%;" />

To shatter the ceiling of open-source AI music and achieve commercial-grade generation, SongGeneration 2 introduces a paradigm shift in both its underlying architecture and training strategy.

1. Model Architecture: Hybrid LLM-Diffusion Architecture & Hierarchical Language Model

   SongGeneration 2 adopts a hybrid LLM-Diffusion architecture to balance musicality and sound quality:

   - **LeLM (The "Composer Brain"):** The language model manages the global musical structure and performance details.
   - **Diffusion (The "Hi-Fi Renderer"):** Guided by the language model, it synthesizes complex acoustic details for high-fidelity audio.
   - **Hierarchical Language Model:** We introduce a hierarchical language model for the parallel modeling of **Mixed Tokens** (to capture high-level semantics like melody and structure) and **Dual-Track Tokens** (to model vocal and accompaniment tracks in parallel for fine-grained acoustic details).

2. Training Strategy: Automated Aesthetic Evaluation & Multi-stage Progressive Post-Training

   To resolve lyrical hallucinations and stiff musicality, we utilize a highly structured training pipeline:

   - **Automated Aesthetic Evaluation Framework:** We built a fine-grained evaluation framework trained on a massive expert-annotated dataset to provide the model with musicality priors.

   - **Multi-stage Progressive Post-training:** We implemented a 3-stage alignment process:

     **Stage 1 - SFT:** Narrows the data distribution using high-quality songs to build a solid generation baseline.

     **Stage 2 - Large-scale Offline DPO:** Utilizes ~200k strict positive/negative pairs to completely eliminate lyrical hallucinations and stabilize controllability.

     **Stage 3 - Semi-online DPO:** Periodically updates the model based strictly on aesthetic scores to maximize musicality limits.

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

## Inference

To ensure the model runs correctly, **please download all the required folders** from the original source at [Hugging Face](https://huggingface.co/collections/lglg666/levo-68d0c3031c370cbfadade126).

- Download `ckpt` and `third_party` folder from [Hugging Face 1](https://huggingface.co/lglg666/SongGeneration-Runtime/tree/main) or  [Hugging Face 2](https://huggingface.co/tencent/SongGeneration/tree/main), and move them into the **root directory** of the project. You can also download models using huggingface-cli.

  ```
  huggingface-cli download lglg666/SongGeneration-Runtime --local-dir ./runtime
  mv runtime/ckpt ckpt
  mv runtime/third_party third_party
  ```

- Download the specific model checkpoint and save it to your specified checkpoint directory: `ckpt_path` (We provide multiple versions of model checkpoints. Please select the most suitable version based on your needs and download the corresponding file. Also, ensure the folder name matches the model version name.) You can also download models using huggingface-cli.

  ```
  # download SongGeneration-base
  huggingface-cli download lglg666/SongGeneration-base --local-dir ./songgeneration_base
  # download SongGeneration-base-new
  huggingface-cli download lglg666/SongGeneration-base-new --local-dir ./songgeneration_base_new
  # download SongGeneration-base-full
  huggingface-cli download lglg666/SongGeneration-base-full --local-dir ./songgeneration_base_full
  # download SongGeneration-large
  huggingface-cli download lglg666/SongGeneration-large --local-dir ./songgeneration_large
  # download SongGeneration-v2-large
  huggingface-cli download lglg666/SongGeneration-v2-large --local-dir ./songgeneration_v2_large
  ```

Once everything is set up, you can run the inference script using the following command:

```bash
sh generate.sh ckpt_path lyrics.jsonl output_path
```

- You may provides sample inputs in JSON Lines (`.jsonl`) format. Each line represents an individual song generation request. The model expects each input to contain the following fields:

  - `idx`: A unique identifier for the output song. It will be used as the name of the generated audio file.
  - `gt_lyric`:The lyrics to be used in generation. It must follow the format of `[Structure] Text`, where `Structure` defines the musical section (e.g., `[Verse]`, `[Chorus]`). See [Input Guide](#Input-Guide).
  - `descriptions` : (Optional) You may customize the text prompt to guide the model’s generation. This can include attributes like gender, genre, emotion, instrument. See [Input Guide](#Input-Guide).
  - `prompt_audio_path`: (Optional) Path to a 10-second reference audio file. If provided, the model will generate a new song in a similar style to the given reference.

  - `auto_prompt_audio_type`: (Optional) Used only if `prompt_audio_path` is not provided. This allows the model to automatically select a reference audio from a predefined library based on a given style. Supported values include:
    - `'Pop'`, `'Latin'`, `'Rock'`, `'Electronic'`, `'Metal'`, `'Country'`,`'R&B/Soul'`, `'Ballad'`, `'Jazz'`, `'World'`, `'Hip-Hop'`,`'Funk'`,`'Soundtrack'`, `'Auto'`.
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

An example input file can be found in `sample/lyrics.jsonl`  and  `sample/test100_v2_sg_des.jsonl` 

### 🎵 Lyrics Input Format

The `gt_lyric` field defines the lyrics and structure of the song. It consists of multiple musical sections, each starting with a structure label. The model uses these labels to guide the musical and lyrical progression of the generated song.

#### 📌 Structure Labels

- The following segments **should not** contain lyrics (they are purely instrumental):

  - `[intro-short]`, `[intro-medium]`, `[inst-short]`, `[inst-medium]`, `[outro-short]`, `[outro-medium]`

  > - `short` indicates a segment of approximately 0–10 seconds
  > - `medium` indicates a segment of approximately 10–20 seconds

- The following segments **require lyrics**:

  - `[verse]`, `[chorus]`, `[bridge]`

#### 🧾 Lyrics Formatting Rules

- To ensure optimal generation quality, please strictly adhere to the following punctuation and formatting rules:

  1. **Section Separation:** Each section (whether instrumental or lyrical) must be separated by a semicolon (`;`).
  2. **Strictly English Punctuation:** Do **not** use any Chinese punctuation marks (e.g., `。`, `，`, `！`). All punctuation must be in English half-width format (e.g., `.`, `,`).
  3. **Sentence Separation & Endings:** Within lyrical segments (`[verse]`, `[chorus]`, `[bridge]`), use a period (`.`) to separate sentences or phrases.
     - **For English lyrics:** The final sentence in a lyrical block **must** end with a period (`.`) before the section separator (`;`).
     - **For Chinese lyrics:** Do **not** place a period (`.`) at the end of the final phrase in a lyrical block. Simply end the phrase, add a space, and use the section separator (`;`).

  💡 A complete lyric string may look like:

  **🇺🇸 English Example:**

  ```
  [intro-medium] ; [verse] Trails wind through the forest. Trees stand tall and honest. Moss covers the logs. Sunlight starts to fondest. Birds sing in the branches. Days feel like a promise. ; [chorus] Forest is the sanctuary where the promise does fondest. Is the tree that stands through the storm's honest. Is the moss that covers the log's modest. Is the peace that makes the restless heart honest. ; [inst-medium] ; [verse] Squirrels scamper by. Nuts hide in the sky. Mushrooms grow below. Fungi start to fly. Streams trickle through. Days feel like a sigh. ; [chorus] Forest is the sanctuary where the promise does fondest. Is the tree that stands through the storm's honest. Is the moss that covers the log's modest. Is the peace that makes the restless heart honest. ; [bridge] Hiking through the forest where the trees do sigh. Feeling the peace that the woods supply. Forest days with you are the sweetest high. ; [chorus] Forest is the sanctuary where the promise does fondest. Is the tree that stands through the storm's honest. Is the moss that covers the log's modest. Is the peace that makes the restless heart honest. ; [outro-medium]
  ```

  **🇨🇳 Chinese Example:**

  ```
  [intro-medium]; [verse] 凌晨三点的便利店.冰柜发出持续的嗡鸣.穿西装的男人在挑饭团.领带松垮像投降的白旗.热食区的关东煮.在汤汁里慢慢膨胀 ; [chorus] 这里是城市的守夜人.收容所有流浪的灵魂.荧光灯照亮的面孔.都写着未完待续的故事 ; [inst-medium]; [verse] 收银员打着哈欠.扫描仪发出嘀嗒声响.找零的硬币落入掌心.带着金属的冰冷温度 ; [chorus] 这里是临时的避风港.用食物交换片刻温暖.即使最孤独的夜晚.也有泡面陪伴到天明 ; [bridge] 自动门开合之间.涌进带着酒气的风.一个女孩蹲在门口.喂食流浪的玳瑁猫 ; [chorus] 这里是不打烊的剧场.上演着无声的悲喜剧.而我们都是临时演员.在黎明前悄然退场 ; [outro-medium]
  ```

  More examples can be found in `sample/test100_v2_sg_des.jsonl`.

### 📝 Description Input Format

The `descriptions` field allows you to control various musical attributes of the generated song. It can describe up to four musical dimensions:

- **Gender** (e.g., `male`, `female`)
- **Genre** (e.g., `pop`, `jazz`, `rock`)
- **Emotion** (e.g., `sad`, `energetic`, `romantic`)
- **Instrument** (e.g., `piano`, `drums`, `guitar`)

**⚠️ CRITICAL FORMATTING RULE: Use Comma-Separated Tags, NOT Sentences.** Please combine specific keywords or tags using commas (`,`). **Do not write full descriptive sentences or natural language paragraphs.**

- All four dimensions are optional — you can specify any subset of them.
- The order of dimensions is flexible.
- Although the model supports an open vocabulary, we highly recommend using **predefined tags** for more stable and reliable performance. A list of commonly supported tags for each dimension is available in the `sample/description/` folder.

#### 💡 Examples

✅ **Valid Inputs (Comma-separated keywords):**

```
female, synth-pop, sweet, synthesizer, drum machine, bass, backing vocals.
rock, loving, electric guitar, bass guitar, drum kit.
```

❌ **Invalid Inputs (Full sentences - DO NOT USE):**

```
Please generate a sad pop song sung by a female artist using piano and drums.
A dark jazz song with a male singer.
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

To rigorously assess the generation capabilities of LeVo 2 (SongGeneration 2), we conducted a large-scale subjective evaluation involving 20 music professionals. The models were evaluated across six core dimensions: Overall Quality, Melody, Arrangement, Sound Quality-Instrument, Sound Quality-Vocal, and Structure.

<img src="../../Library/CloudStorage/OneDrive-%E4%B8%AA%E4%BA%BA/%E5%9B%BE%E7%89%87/output-2214682.png" alt="img" style="zoom:100%;" />

As shown in the benchmarking results above, LeVo 2 comprehensively outperforms all existing open-source baselines and achieves generation quality that directly rivals top-tier closed-source commercial models.

### 📌 Notes on Evaluation & Generation

- **Evaluation Data:** The evaluation results are based on 100 generated songs using descriptions. We also provide all inputs used for this benchmark in sample/test100_v2_sg_des.jsonl for reference and reproducibility.
- **Impact of Audio Prompts:** Since the model attempts to clone the timbre and musical style of the given prompt audio, the choice of prompt audio can significantly affect generation performance, and may lead to fluctuations in the evaluation metrics.
- **Importance of Lyric Formatting:** The format of the input lyrics has a strong impact on generation quality. If the output quality appears suboptimal, please check whether your lyrics format is strictly correct according to our formatting rules. You can find more examples of properly formatted inputs in sample/test100_v2_sg_des.jsonl.

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

The code and weights in this repository are released under the [LICENSE](LICENSE)  file.


## Contact

Use WeChat or QQ to scan the below QR code

<div style="display: flex; justify-content: center; gap: 20px; width: 100%;">
  <img src="img/contact.jpg" height="300" />
  <img src="img/contactQQ.jpg" height="300" />
</div>

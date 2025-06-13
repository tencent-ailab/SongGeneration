from torch.utils.data import Dataset
from beartype.typing import Sequence, Callable, Optional, Dict, Tuple, List, Union
from beartype import beartype
from beartype.door import is_bearable
import random
import pandas as pd
import os
from torchaudio.functional import resample
import torch
import typing as tp
from pathlib import Path
import torchaudio as ta
import torch.nn.functional as F
import numpy as np
import json
import yaml
import torchaudio 
import math
import re 
from loguru import logger
import ffmpeg

class Read_and_PadCrop_Normalized_T(torch.nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, filename: str, duration: float, cur_sample_rate: int) -> Tuple[torch.Tensor, float, float, int, int]:
        if  self.n_samples < 0: #means not clip
            chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
            t_start = 0.
            t_end = 1.0
            offset = 0
        else:
            if(duration<(float(self.n_samples)/self.sample_rate+1)):
                # print(duration,(float(self.n_samples)/self.sample_rate+1))
                chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
                t_start = 0.
                t_end = min(1.0, float(self.n_samples) / float(self.sample_rate) / duration)
                offset = 0
                # print('c1:',chunk.shape)
            else:
                offset = np.random.randint(0,int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
                t_start = offset / float(cur_sample_rate) / duration
                t_end = t_start + float(self.n_samples) / float(self.sample_rate) / duration
                chunk, _ = torchaudio.load(filename, frame_offset=offset, num_frames=int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
                # print('offset:',offset)
                # print('c0:',chunk.shape)
            # Pad with silence if necessary.
        if(chunk.shape[0]>1):
            chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
        else:
            chunk = chunk[[0],:].float()
        if(cur_sample_rate!=self.sample_rate):
            # print('a:',cur_sample_rate,chunk.shape)
            chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sample_rate)
            # print('b:',self.sample_rate,chunk.shape)
            
        if self.n_samples > 0:
            if chunk.shape[-1] < self.n_samples:
                chunk = torch.cat([chunk, torch.zeros((1, self.n_samples - chunk.shape[-1],))],-1)
            else:
                chunk = chunk[:,0:self.n_samples]
        seconds_start = math.floor(offset / cur_sample_rate)
        seconds_total = math.floor(duration)

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total
        )

class Read_and_PadCrop_Normalized_T_Avoid_Watermark(torch.nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True, w_start = 0, w_interval = 11.3):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

        self.w_start = w_start
        self.w_interval = w_interval

    def __call__(self, filename: str, duration: float, cur_sample_rate: int) -> Tuple[torch.Tensor, float, float, int, int]:
        if  self.n_samples < 0: #means not clip
            chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
            t_start = 0.
            t_end = 1.0
            offset = 0
        else:
            if(duration<(float(self.n_samples)/self.sample_rate+1)):
                # print(duration,(float(self.n_samples)/self.sample_rate+1))
                chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
                t_start = 0.
                t_end = min(1.0, float(self.n_samples) / float(self.sample_rate) / duration)
                offset = 0
                # print('c1:',chunk.shape)
            else:
                n_offset_option = (duration - self.w_start) // self.w_interval
                if n_offset_option <= 1:
                    offset = 0
                else:
                    offset = int((random.randint(0,n_offset_option-1) * self.w_interval + self.w_start) * cur_sample_rate)
                # offset = np.random.randint(0,int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
                t_start = offset / float(cur_sample_rate) / duration
                t_end = t_start + float(self.n_samples) / float(self.sample_rate) / duration
                chunk, _ = torchaudio.load(filename, frame_offset=offset, num_frames=int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
                # print('offset:',offset)
                # print('c0:',chunk.shape)
            # Pad with silence if necessary.
        if(chunk.shape[0]>1):
            chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
        else:
            chunk = chunk[[0],:].float()
        if(cur_sample_rate!=self.sample_rate):
            # print('a:',cur_sample_rate,chunk.shape)
            chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sample_rate)
            # print('b:',self.sample_rate,chunk.shape)
            
        if self.n_samples > 0:
            if chunk.shape[-1] < self.n_samples:
                chunk = torch.cat([chunk, torch.zeros((1, self.n_samples - chunk.shape[-1],))],-1)
            else:
                chunk = chunk[:,0:self.n_samples]
        seconds_start = math.floor(offset / cur_sample_rate)
        seconds_total = math.floor(duration)

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total
        )

USE_DUMMY_AUDIO = False #当测试代码时，可以将其置为True，这样就不会读取实际数据，而是用生成的静默音频代替
if USE_DUMMY_AUDIO:
    logger.warning("USE_DUMMY_AUDIO flag is True, don't use it when train or test!")

class SafeAudioReader:
    """
       This class is an adaptor to Read_and_PadCrop_Normalized_T, make it safe to read audio data.
    """
    def __init__(self, 
                duration: float,  # 返回音频长度
                sample_rate: int, # 返回音频的采样率，如与实际音频采样率不同，会作resample
                randomize: bool = True,
                use_avoid_watermark_policy = False,
                ):
        self.n_samples = int(sample_rate * duration)
        self.reader = (
            Read_and_PadCrop_Normalized_T_Avoid_Watermark if use_avoid_watermark_policy \
            else Read_and_PadCrop_Normalized_T
            )(n_samples=self.n_samples, sample_rate=sample_rate, randomize=randomize)
    
    #NOTE:这个是核心的函数，所有数据集读取音频都是调用的这个函数！
    def __call__(self, 
                 filepath: os.PathLike,  # 音频路径
                 origin_sample_rate: Optional[int] = None,  # 从json文件中读取的实际采样率，如果不给定，则会从文件头中读取
                 origin_duration: float = None, # 从json文件中读取的实际时长，如果不给定，则会从文件头中读取
                 ) -> torch.Tensor:
        if USE_DUMMY_AUDIO:
            wav = torch.zeros(self.n_samples, dtype=torch.float32)
            return wav
        try:
            if origin_sample_rate is None or origin_duration is None:
                # audio_info = torchaudio.info(filepath)
                # origin_sample_rate = audio_info.sample_rate
                # origin_duration = audio_info.num_frames / origin_sample_rate
                info = ffmpeg.probe(filepath)
                origin_duration = float(info['format']['duration'])
                origin_sample_rate = int(info['streams'][0]['sample_rate'])
            wav, *ignored = self.reader(filepath, origin_duration, origin_sample_rate)
            wav = wav.squeeze_(0)
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            wav = torch.zeros(self.n_samples, dtype=torch.float32)
        return wav
    

class PromptTemplate:
    def __init__(self, template_text: str, tag_map: Dict[str, str], lang:str ='en'):
        self.template_text = template_text
        self.tag_map = tag_map
        self.lang = lang
    
    @property
    def tags(self):
        return tuple(self.tag_map.keys())

    def apply(self, **kwargs):
        for tag in list(kwargs.keys()):
            if kwargs[tag] == '':
                kwargs.pop(tag)
        for tag in self.tags:
            if tag in kwargs:
                kwargs[tag] = self.tag_map[tag].format(**{tag: kwargs[tag]}).strip('[]')
            else:
                kwargs[tag] = ''
        prompt = self.template_text.format(**kwargs)
        
        return self.beautify(prompt)
    
    def beautify(self, text):
        if self.lang == 'en':
            return self._beautify_en(text)
        elif self.lang == 'zh':
            return self._beautify_zh(text)
        else:
            raise ValueError(f'Unknown language {self.lang}')
        
    @staticmethod
    def _beautify_en(text):
        # no continuous commas without content between them
        text = re.sub(r'[,\s]*,[,\s]*', r', ', text)
        # no continuous whitespace
        text = re.sub(r'\s+', ' ', text)
        # the comma is NOT followed by whitespace, and should be followed by ONE whitespace
        text = re.sub(r'\s+,', r',', text)
        text = re.sub(r',\s+', r', ', text)
        # no whitespace before the full stop
        text = re.sub(r'\s+\.', r'.', text)
        # strip whitespace, comma, and replace ',.'
        text = text.strip(' ,')
        text = text.replace(',.', '.')
        return text
    
    @staticmethod
    def _beautify_zh(text):
        # no continuous commas without content between them
        text = re.sub(r'[，、\s]*，[，、\s]*', r'，', text)
        text = re.sub(r'[，、\s]*、[，、\s]*', r'、', text)
        # assume there should be NO whitespace in Chinese
        text = re.sub(r'\s+', r'', text)
        # strip whitespace, comma, and replace '，。'
        text = text.strip('， 、')
        text = text.replace('，。', '。')
        return text

    def __repr__(self):
        return f'PromptTemplate({self.template_text!r}, {self.tag_map!r})'

    __str__ = __repr__

def parse_prompt_template(prompt_template_text, lang='en'):
    span_pattern = re.compile(r'\[.*?{.+?}.*?\]', re.DOTALL)
    tag_pattern = re.compile(r'{.+?}', re.DOTALL)

    template_text = prompt_template_text.strip()
    span_texts = span_pattern.findall(prompt_template_text)
    tag_map = {} 
    for span_text in span_texts:
        tag = tag_pattern.findall(span_text)[0].strip('{}')
        tag_map[tag] = span_text
        template_text = template_text.replace(span_text, '{'+tag+'}')
    
    return PromptTemplate(template_text=template_text, tag_map=tag_map, lang=lang)

def load_prompt_templates(path, num = 5, lang='en') -> List[PromptTemplate]:
    with open(path, 'r') as f:
        lines = f.readlines()
    cnt = 0
    pts = []
    for line in lines:
        pt = parse_prompt_template(line, lang=lang)
        cnt += 1
        if len(pt.tags) < num:
            logger.error(f'Not enough tags on {path} in line {cnt}: {pt.tags}')
        pts.append(pt)

    return pts

    
def get_base_dir_file(key: os.PathLike):
    base = os.path.basename(key)
    dirname = os.path.basename(os.path.dirname(key))
    return os.path.join(dirname, base)

def read_jsonlike(path: os.PathLike):
    #json or jsonl
    if str(path).endswith(".json"):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return data
    elif str(path).endswith(".jsonl"):
        with open(path, 'r', encoding='utf8') as f:
            data = [json.loads(line) for line in f.readlines()]
        return data
    else:
        raise ValueError("Unknown file format")

dist_prob_map = {
    1: (1.0,),
    2: (0.5, 0.5),
    3: (0.3, 0.4, 0.3),
    4: (0.2, 0.3, 0.3, 0.2),
    5: (0.2, 0.2, 0.3, 0.2, 0.1),
    6: (0.1, 0.15, 0.2, 0.2, 0.2, 0.15),
    7: (0.05, 0.1, 0.1, 0.2, 0.25, 0.2, 0.1),
    8: (0.03, 0.05, 0.1, 0.15, 0.25, 0.2, 0.1, 0.12),
    9: (0.02, 0.1, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.08),
    10: (0.01, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05, 0.09)
}

'''
#更加偏向短文本的方案
dist_prob_map = {
    1: (1.0,),
    2: (0.7, 0.3),
    3: (0.7, 0.2, 0.1),
    4: (0.6, 0.2, 0.1, 0.1),
    5: (0.6, 0.2, 0.1, 0.05, 0.05),
    6: (0.6, 0.15, 0.1, 0.05, 0.05, 0.05),
    7: (0.05, 0.1, 0.1, 0.2, 0.25, 0.2, 0.1),
    8: (0.03, 0.05, 0.1, 0.15, 0.25, 0.2, 0.1, 0.12),
    9: (0.02, 0.1, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.08),
    10: (0.01, 0.1, 0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.05, 0.09)
}
'''

#全部都用的方案
# dist_prob_map = {
#     1: (1.0,),
#     2: (0, 1.0),
#     3: (0, 0, 1.0),
#     4: (0, 0, 0, 1.0),
#     5: (0, 0, 0, 0, 1.0),
#     6: (0, 0, 0, 0, 0, 1.0),
#     7: (0, 0, 0, 0, 0, 0, 1.0),
#     8: (0, 0, 0, 0, 0, 0, 0, 1.0),
#     9: (0, 0, 0, 0, 0, 0, 0, 0, 1.0),
#     10: (0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
# }

dist_prob_map_low = {
    1: (1.0,),
    2: (0.8, 0.2),
    3: (0.8, 0.1, 0.1),
    4: (0.7, 0.1, 0.1, 0.1),
    5: (0.7, 0.1, 0.1, 0.05, 0.05),
    6: (0.7, 0.1, 0.05, 0.05, 0.05, 0.05),
}

_bpm_range_rights = (
    (40, '20-40'),
    (60, '40-60'),
    (66, '60-66'),
    (76, '66-76'),
    (108, '76-108'),
    (120, '108-120'),
    (168, '120-168'),
    (176, '168-176'),
    (200, '176-200')
)
_bpm_desc_map = {
    '20-40': ("glacial pace", "extremely slow tempo", "crawl-like speed", "snail's pace", "almost motionless rhythm", "Larghissimo"),
    '40-60': ("broad and slow", "spacious tempo", "unhurried pace", "calm rhythm", "relaxed speed", "Largo"),
    '60-66': ("gentle tempo", "leisurely pace", "easy-going rhythm", "unrushed speed", "smooth and slow", 'Larghetto'),
    '66-76': ("slow and steady", "deliberate tempo", "unhurried pace", "relaxed rhythm", "easy speed", 'Adagio'),
    '76-108': ("walking pace", "moderate tempo", "steady rhythm", "balanced speed", "easy-flowing tempo", "Andante"),
    '108-120': ("medium pace", "comfortable tempo", "even rhythm", "measured speed", "controlled tempo", 'Moderato'), 
    '120-168': ("quick and lively", "brisk pace", "energetic tempo", "upbeat rhythm", "spirited speed", 'Allegro'),
    '168-176': ("lively and fast", "bright tempo", "sprightly pace", "vibrant rhythm", "animated speed", 'Vivace'),
    '176-200': ("very fast tempo", "rapid pace", "high-speed rhythm", "hurried speed", "accelerated tempo", 'Presto'),
    '>200': ("extremely fast", "breakneck speed", "blazing tempo", "lightning-fast rhythm", "supercharged pace", 'Prestissimo')
}
_bpm_desc_map_zh = {
    '20-40': ("极度缓慢", "极慢的节奏", "悠长的旋律", "迟缓的节奏", "几乎静止的节奏", "甚缓"),
    '40-60': ("宽广而缓慢", "宽敞的节奏", "从容不迫的速度", "平静的节奏", "轻松的速度", "广板"),
    '60-66': ("柔和的节奏", "悠闲的速度", "轻松的节奏", "不慌不忙的速度", "平滑而缓慢", '小广板'),
    '66-76': ("缓慢而稳定", "沉稳的旋律", "从容不迫的速度", "轻松的节奏", "轻松的速度", '慢板'),
    '76-108': ("步行速度", "适中的节奏", "稳定的节奏", "平衡的速度", "流畅的节奏", "行板"),
    '108-120': ("中等速度", "舒适的节奏", "均匀的节奏", "有节制的速度", "稳定的氛围", '中板'), 
    '120-168': ("快速而生动", "轻快的速度", "充满活力的节奏", "欢快的节奏", "富有精神的速度", '快板'),
    '168-176': ("生动而快速", "明快的节奏", "活泼的速度", "充满活力的节奏", "生气勃勃的速度", '活泼的'),
    '176-200': ("非常快的节奏", "快速的速度", "高速的节奏", "匆忙的速度", "加速的节奏", '急板'),
    '>200': ("极快的速度", "极速旋律", "炽热的节奏", "闪电般的节奏", "疾驰的速度", '最急板')
}
def get_bpm_range(bpm):
    bpm = int(bpm)
    for right, tag in _bpm_range_rights:
        if bpm <= right:
            return tag
    return '>200'

def gen_bpm_descript(bpm, lang='en'):
    bpm_range = get_bpm_range(bpm)
    if lang == 'en':
        return random.choice(_bpm_desc_map[bpm_range])
    elif lang == 'zh':
        return random.choice(_bpm_desc_map_zh[bpm_range])
    else:
        raise ValueError(f"Unknown language {lang}")

def read_translate(translate: Union[Dict[str, os.PathLike], os.PathLike, None]):
    if translate is None:
        return None 
    if isinstance(translate, str):
        return read_jsonlike(translate)
    return {k: read_jsonlike(path) for k, path in translate.items()}


def gen_plain_prompt(key_list, sep=', '):
    if len(key_list) == 0:
        return 'none'
    
    key_list = [k.strip() for k in key_list]
    
    if len(key_list) > 10:
        random.shuffle(key_list)
        key_list = key_list[:10]

    probs = dist_prob_map[len(key_list)]

    num_tags = random.choices(range(1, len(key_list)+1), probs, k=1)[0]

    random.shuffle(key_list)
    tags = key_list[:num_tags]
    tags_str = sep.join(tags)
    return tags_str    
    

class MagnaTagATuneDataset(Dataset):
    def __init__(self):
        pass


def tags_to_desc(tag_list, sep=',') -> str:
    if not isinstance(tag_list, Sequence):
        return str(tag_list)
    if isinstance(tag_list, str):
        return tag_list
    if len(tag_list) <= 0:
        return ''
    elif len(tag_list) <= 5:
        probs = dist_prob_map[len(tag_list)]
        tags_num = random.choices(range(1, len(tag_list)+1), probs)[0]
        random.shuffle(tag_list)
        tag_list = tag_list[:tags_num]
        return sep.join(tag_list)
    else:
        probs = dist_prob_map[5]
        tags_num = random.choices(range(1, 6), probs)[0]
        random.shuffle(tag_list)
        tag_list = tag_list[:tags_num]
        return sep.join(tag_list)

def get_sr_and_duration_info(item):
    return item.get('sample_rate', None), item.get('duration', None)

class MtgJamendoDatasetFromJson(Dataset):
    def __init__(self, 
                data_dir:str, 
                json_path:str, 
                duration:float=10, 
                sr:int = 0, 
                lang = 'en',
                plain_rate = 0,
                return_audio = True,
                return_path = False, 
                prompt_template_path: os.PathLike = None, 
                tag_types = [],
                translate:Optional[Dict[str, os.PathLike]] = None,
                use_literal_none = True,
                ):
        self.audio_reader = SafeAudioReader(duration, sr)

        self.data_dir = data_dir
        self._load_metadata_json(json_path)   
        self.sr = sr
        self.duration = duration
        self.plain_rate = plain_rate
        self.return_audio = return_audio
        self.return_path = return_path
        self.use_literal_none = use_literal_none
        self.lang = lang
    
        self.use_dynamic_prompt = prompt_template_path is not None and plain_rate < 1.0
        if self.use_dynamic_prompt:
            self.prompt_templates = load_prompt_templates(prompt_template_path, num = len(tag_types))
        self.tag_types = tag_types

        self.translate = read_translate(translate)
    
    #这些tag被认为是弱语义的，会避免产生仅包含这些tag的文本提示
    WEAK_TAG_LIST = ["title", "artist"]

    def _load_metadata_json(self, json_path):
        with open(json_path) as fp:
            self.data = json.load(fp)
    
    def convert_key_to_path(self, key):
        return os.path.join(self.data_dir, get_base_dir_file(key))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        path = self.convert_key_to_path(item['key'])
        description = self.generate_description(item)

        if self.return_audio:
            sr, duration = get_sr_and_duration_info(item)
            audio = self.audio_reader(path, sr, duration)
        else:
            audio = None

        if self.return_path:
            return audio, description, path
        return audio, description
    
    def tags_to_desc(self, tag_list, tag_type) -> str:
        if self.lang == 'en':
            return tags_to_desc(tag_list)
        elif self.lang == 'zh':
            translator = self.translate[tag_type]
            translated_tag_list = [translator[tag] for tag in tag_list if tag in translator ]
            return tags_to_desc(translated_tag_list, sep='、')
    
    def generate_description(self, item):
        if random.random() > self.plain_rate:
            # dynamically generate prompt from given prompt template
            prompt_template = random.choice(self.prompt_templates)
            description = self.generate_description_dynamic(item, prompt_template)
        else:
            # use plain prompt, i.e. tags sequence separated by comma
            description = self.generate_description_plain(item)
        return description
    
    def generate_description_dynamic(self, data, prompt_template: PromptTemplate):
        exists_tag = [key for key in data if (key in self.tag_types) and (data[key] is not None) and (len(data[key]) > 0)]
        exists_weak_tag = list(filter(lambda t: t in self.WEAK_TAG_LIST, exists_tag))
        exists_strong_tag = list(filter(lambda t: t not in self.WEAK_TAG_LIST, exists_tag))

        if len(exists_strong_tag) > 0:
            probs = dist_prob_map[len(exists_strong_tag)]
            tags_num = random.choices(range(1, len(exists_strong_tag)+1), probs)[0]
            random.shuffle(exists_strong_tag)
            tags = exists_strong_tag[:tags_num]
            weak_probs = dist_prob_map_low[len(exists_weak_tag) + 1]
            weak_tags_num = random.choices(range(0, len(exists_weak_tag) + 1), weak_probs)[0]
            random.shuffle(exists_weak_tag)
            weak_tags = exists_weak_tag[:weak_tags_num]
            tags += weak_tags
            tags_args = {tag: self.tags_to_desc(data[tag], tag) for tag in tags}
            prompt = prompt_template.apply(**tags_args)
        else: 
            # no strong tags, use all weak tags instead
            tags_args = {tag: self.tags_to_desc(data[tag], tag) for tag in exists_weak_tag}
            prompt = prompt_template.apply(**tags_args)
        
        if self.use_literal_none and len(tags_args) == 0:
            return 'none'
        
        return prompt

    def generate_description_plain(self, item):
        keywords = []
        for tag_t in self.tag_types:
            this_key = item[tag_t]
            if this_key is None: 
                continue
            if isinstance(this_key, str):
                this_key = [this_key]
            if self.lang != 'en':
                this_key = [self.get_translation(tag_t, k) for k in this_key]
            keywords += this_key
        return gen_plain_prompt(keywords, sep=self.keysep)

    def get_translation(self, tag_t, k):
        k = k.strip()
        if k in self.translate[tag_t]:
            return self.translate[tag_t][k]
        else:
            return k

    @property
    def keysep(self):
        if self.lang == 'zh':
            return '，' if random.random() > 0.5 else '、'
        elif self.lang == 'en':
            return ', '

class AudioStockDataset(Dataset):
    def __init__(self, 
                metadata_path:str, 
                duration:float=10, 
                sr:int = 0, 
                plain_rate = 0,
                return_path = False, 
                return_audio = True, 
                prompt_template_path: os.PathLike = None, 
                tag_types = [],
                lang = 'en',
                translate:Optional[Dict[str, os.PathLike]] = None,
                use_literal_none = True,
                ):
        self.audio_reader = SafeAudioReader(duration, sr)

        self._load_metadata(metadata_path) 
        self.sr = sr
        self.duration = duration
        self.plain_rate = plain_rate
        self.return_path = return_path
        self.return_audio = return_audio
        self.use_literal_none = use_literal_none

        self.use_dynamic_prompt = prompt_template_path is not None and plain_rate < 1.0
        if self.use_dynamic_prompt:
            self.prompt_templates = load_prompt_templates(prompt_template_path, num = len(tag_types), lang = lang)
        self.tag_types = tag_types

        self.lang = lang
        self.translate = read_translate(translate)

    def _load_metadata(self, metadata_path):
        with open(metadata_path) as fp:
            lines = fp.readlines()
            self.data = []
            for line in lines: 
                item = json.loads(line)
                self.data.append(item)
        self.is_info_recorded = bool('Tags' in self.data[0])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path:str = self.data[idx]["path"]
        json_path = path[:path.rfind('.')] + ".json"
        if self.is_info_recorded:
            item = self.data[idx]
        else:
            try:
                with open(json_path) as fp:
                    item:dict = json.load(fp)
            except Exception as e:
                print(f"Error loading json file {json_path} :\n{e}")
                item = {}
        description = self.generate_description(item)
        if self.return_audio:
            sr, duration = get_sr_and_duration_info(item)
            audio = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            return audio, description, path
        return audio, description
    
    def generate_description(self, item):
        if random.random() > self.plain_rate:
            # dynamically generate prompt from given prompt template
            prompt_template = random.choice(self.prompt_templates)
            description = self.generate_description_dynamic(item, prompt_template)
        else:
            # use plain prompt, i.e. tags sequence separated by comma
            description = self.generate_description_plain(item)
        return description
    
    def generate_description_dynamic(self, data, prompt_template: PromptTemplate):
        exists_tag = [key for key in data if (key in self.tag_types) and (data[key] is not None) and (len(data[key]) > 0)]

        if len(exists_tag) > 0:
            probs = dist_prob_map[len(exists_tag)]
            tags_num = random.choices(range(1, len(exists_tag)+1), probs)[0]
            random.shuffle(exists_tag)
            tags = exists_tag[:tags_num]
            tags_args = {tag: self.tags_to_desc(data[tag], tag) for tag in tags}
            tags_args = self.handle_BPM_tag(tags_args)
            prompt = prompt_template.apply(**tags_args)
        else: 
            return 'none'
        
        if self.use_literal_none and len(tags_args) == 0:
            return 'none'
        
        return prompt

    def get_translation(self, tag_t, k):
        k = k.strip()
        if k in self.translate[tag_t]:
            return self.translate[tag_t][k]
        else:
            return k

    def generate_description_plain(self, item):
        keywords = []
        for tag_t in self.tag_types:
            if tag_t == 'BPMDescript':
                bpm = item['BPM']
                if bpm is None or bpm.strip() == '' or bpm.strip() == '0':
                    continue
                this_key = gen_bpm_descript(bpm.strip(), lang=self.lang)
            elif tag_t == 'BPM':
                bpm = item['BPM']
                if bpm is None or bpm.strip() == '' or bpm.strip() == '0':
                    continue
                this_key = f"{bpm.strip()} bpm"
            else:
                this_key = item[tag_t]
                if this_key is None: 
                    continue
                if isinstance(this_key, str):
                    this_key = [this_key]
                if self.lang != 'en':
                    this_key = [self.get_translation(tag_t, k) for k in this_key]
            if this_key is None: 
                continue
            if isinstance(this_key, str):
                this_key = [this_key]
            keywords += this_key
        return gen_plain_prompt(keywords, sep=self.keysep)

    @property
    def keysep(self):
        if self.lang == 'zh':
            return '，' if random.random() > 0.5 else '、'
        elif self.lang == 'en':
            return ', '
    
    def tags_to_desc(self, tag_list, tag_type) -> str:
        if self.lang == 'en':
            return tags_to_desc(tag_list)
        elif self.lang == 'zh':
            if tag_type == 'BPM':
                return tags_to_desc(tag_list, sep='、')
            translator = self.translate[tag_type]
            translated_tag_list = [translator[tag] for tag in tag_list if tag in translator ]
            return tags_to_desc(translated_tag_list, sep='、')
    
    def handle_BPM_tag(self, tags_args):
        if "BPM" in tags_args and 'BPMDescript' in  self.tag_types:
            bpm = tags_args["BPM"]
            del tags_args["BPM"]
            tag_types_used = random.choice((('BPM',), ('BPMDescript',), ('BPM', 'BPMDescript')))
            for tag_type in tag_types_used:
                tags_args[tag_type] = bpm if tag_type == 'BPM' else gen_bpm_descript(bpm, lang=self.lang)
        return tags_args
    
def mp3_path_to_id(mp3_path):
    return int(
        mp3_path[mp3_path.rindex('/') + 1 : mp3_path.rindex('.')]
    )
 
class TmeDataset(Dataset):
    def __init__(self, 
                data_index:str, 
                music_info:str = None, 
                duration:float = 10, 
                sr:int = 0, 
                plain_rate = 0,
                return_path = False, 
                return_audio = True, 
                return_ID = False,
                prompt_format_path: os.PathLike = None, 
                tag_types = ['*'],
                lang = 'zh',
                translate: Optional[os.PathLike] = None,
                prompt_dir: os.PathLike = None, #使用GPT生成的预有的prompt
                ):
        if plain_rate > 0:
            print("Tme Dataset do not support plain rate > 0, use plain_rate = 0 instead.")
            plain_rate = 0
        self.audio_reader = SafeAudioReader(duration, sr)

        self.sr = sr
        self.duration = duration
        self.plain_rate = plain_rate
        self.return_path = return_path
        self.return_audio = return_audio
        self.return_ID = return_ID
        self.lang = lang

        self.use_ready_prompt = prompt_dir is not None

        data_index = read_jsonlike(data_index)
        self.data_index_dict = {mp3_path_to_id(d['path']) : d for d in data_index}
        self.data_ids = list(self.data_index_dict.keys())

        if not self.use_ready_prompt:
            #读取音乐的信息文件
            music_info = read_jsonlike(music_info)
            if 'music' in music_info:
                music_info = music_info['music']
            self.music_info_dict = {d["歌曲ID"]:d for d in music_info}
            self.data_index_dict = {k:v for k,v in self.data_index_dict.items() if k in self.music_info_dict}
            self.data_ids = list(self.data_index_dict.keys())
        
            with open(prompt_format_path) as fp:
                self.prompt_formats = yaml.load(fp, Loader=yaml.FullLoader)

            #加载tag types，并分成一般的tag_types和关键的key_tag_types
            if '*' in tag_types:
                self.tag_types = ['歌曲名', 'bpm', '专辑名', '歌手名', '作曲', 'tag']
            else:
                self.tag_types = tag_types
            
            self.key_tag_types = []
            if 'tag' in self.tag_types:
                self.tag_types.remove('tag')
                self.key_tag_types = list(self.prompt_formats['tag'].keys())

            #加载translate翻译 
            if translate is not None:
                self.translator = read_jsonlike(translate)
        else:
            data_ids_set = set(self.data_ids)
            self.prompts_dict = {}
            for fname in os.listdir(prompt_dir):
                items = read_jsonlike(os.path.join(prompt_dir, fname))
                for item in items:
                    if item['ID'] not in data_ids_set or not self.is_valid_prompt_text(item['Text']):
                        continue
                    if item['ID'] not in self.prompts_dict:
                        self.prompts_dict[item['ID']] = []
                        self.prompts_dict[item['ID']].append(item['Text'])
            self.data_index_dict = {k:v for k,v in self.data_index_dict.items() if k in self.prompts_dict}
            self.data_ids = list(self.data_index_dict.keys())
    
    def tags_to_desc(self, tag_list) -> str:
        if is_bearable(tag_list, int):
            return str(tag_list)
        if self.lang == 'zh':
            return tags_to_desc(tag_list, sep=self.sep)
        else:
            translated_tag_list = [self.translator[tag] for tag in tag_list if tag in self.translator ]
            return tags_to_desc(translated_tag_list, sep=self.sep)

    def gen_desc_of_tag(self, formats, tags):
        fmt = random.choice(formats)
        return fmt.format(self.tags_to_desc(tags))
    
    @staticmethod
    def check_valid(value):
        if isinstance(value, int) or isinstance(value, float):
            return value > 0
        if (value is not None) and (not isinstance(value, Sequence) or len(value) > 0):
            return True
        return False
    
    @staticmethod
    def remove_repeat(data):
        #若专辑名和歌曲名相同，则只使用后者
        album_name = data.get('专辑名', None)
        if album_name is not None and album_name == data.get('歌曲名', None):
            del data['专辑名']
        return data
    
    @property
    def comma(self):
        if self.lang == 'zh':
            return '，'
        elif self.lang == 'en':
            return ', '
        
    @property
    def sep(self):
        if self.lang == 'zh':
            return '、'
        elif self.lang == 'en':
            return ', '
    

    def generate_description(self, item):
        if random.random() > self.plain_rate:
            # dynamically generate prompt from given prompt template
            description = self.generate_description_dynamic(item)
        else:
            # use plain prompt, i.e. tags sequence separated by comma
            description = self.generate_description_plain(item)
        return description

    def generate_description_dynamic(self, data):
        data = self.remove_repeat(data)

        weak_tags = [key for key in data if (key in self.tag_types and self.check_valid(data[key]))] #弱语义的tag，这些tag的出现比例会放低

        key_tags = [key for key in data['tag'] if (key in self.key_tag_types and self.check_valid(data['tag'][key]))] #关键的tag，这些tag必须出现至少一个

        prompts = []
        if len(weak_tags) > 0:
            probs = dist_prob_map_low[len(weak_tags)]
            if len(key_tags) > 0:
                tags_num = random.choices(range(0, len(weak_tags)), probs)[0]
            else:
                tags_num = random.choices(range(1, len(weak_tags) + 1), probs)[0]
            random.shuffle(weak_tags)
            tags = weak_tags[:tags_num]
            for tag_type in tags:
                tag_desc = self.gen_desc_of_tag(self.prompt_formats[tag_type], int(data[tag_type]) if tag_type == 'bpm' else data[tag_type])
                prompts.append(tag_desc)

        if len(key_tags) > 0:
            probs = dist_prob_map[len(key_tags)]
            tags_num = random.choices(range(1, len(key_tags) + 1), probs)[0]
            random.shuffle(key_tags)
            tags = key_tags[:tags_num]
            for tag_type in tags:
                tag_desc = self.gen_desc_of_tag(self.prompt_formats['tag'][tag_type], data['tag'][tag_type])
                prompts.append(tag_desc)

        random.shuffle(prompts)        
        return self.comma.join(prompts)
    
    def generate_description_plain(self, item):
        keywords = item['tag']
        if self.lang != 'en':
            keywords = [self.translator[k.strip()] for k in keywords]
        return gen_plain_prompt(keywords, sep=self.keysep)

    @property
    def keysep(self):
        if self.lang == 'zh':
            return '，' if random.random() > 0.5 else '、'
        elif self.lang == 'en':
            return ', '

    def is_valid_prompt_text(self, text):
        for bad in ('抱歉','sorry', 'Sorry'):
            if bad in text:
                return False
        return True

    def get_ready_prompt(self, path):
        sid = mp3_path_to_id(path)
        return random.choice(self.prompts_dict[sid])

    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        item = self.data_index_dict[data_id]
        path = item['path']
        if not self.use_ready_prompt:
            info = self.music_info_dict[data_id]
            description = self.generate_description(info)
        else:
            description = self.get_ready_prompt(path)
        if self.return_audio:
            sr, duration = get_sr_and_duration_info(item)
            audio = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            if self.return_ID:
                return audio, description, path, info['歌曲ID']
            return audio, description, path
        if self.return_ID:
            return audio, description, info['歌曲ID']
        return audio, description


class Pond5Dataset(Dataset):
    MAX_PROMPT_LEN = 200
    def __init__(self, 
                metadata_path:str, 
                index_path:str,
                duration:float=10, 
                sr:int = 0, 
                plain_rate = 0,
                return_path = False, 
                return_audio = True, 
                lang = 'en',
                translate:Optional[Dict[str, os.PathLike]] = None,
                use_literal_none = True,
                use_avoid_watermark_policy = None,
                ):
        
        if use_avoid_watermark_policy is None:
            raise ValueError("`use_avoid_watermark_policy` is an important param, you need to explicitly specify it with bool type")
        self.use_avoid_watermark_policy = use_avoid_watermark_policy
        self.audio_reader = SafeAudioReader(duration, sr, use_avoid_watermark_policy=use_avoid_watermark_policy)

        self._load_metadata(metadata_path, index_path) 
        self.sr = sr
        self.duration = duration
        self.plain_rate = plain_rate
        self.return_path = return_path
        self.return_audio = return_audio
        self.use_literal_none = use_literal_none

        self.lang = lang
        self.translate = read_translate(translate)

    def _load_metadata(self, metadata_path, index_path):
        data_index = read_jsonlike(index_path)
        data_ids = set([item['id'] for item in data_index])

        with open(metadata_path) as fp:
            lines = fp.readlines()
        
        append_ids = set()

        self.data = []
        for line in lines: 
            item = json.loads(line)
            if item['id'] in data_ids and item['id'] not in append_ids:
                self.data.append(item)
                append_ids.add(item['id'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        path:str = item["path"]
        description = self.generate_description(item)
        if self.return_audio:
            sr, duration = get_sr_and_duration_info(item)
            audio = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            return audio, description, path
        return audio, description

    @property
    def keysep(self):
        if self.lang == 'zh':
            return '，' if random.random() > 0.5 else '、'
        elif self.lang == 'en':
            return ', '
    
    def generate_description(self, item):
        if random.random() > self.plain_rate:
            # dynamically generate prompt from given prompt template
            description = self.generate_description_dynamic(item)
        else:
            # use plain prompt, i.e. tags sequence separated by comma
            description = self.generate_description_plain(item)
        return description
    
    def get_translation(self, k):
        k = k.strip()
        if k in self.translate:
            return self.translate[k]
        else:
            return k

    def generate_description_plain(self, item):
        keywords = item['keywords']
        if self.lang != 'en':
            keywords = [self.get_translation(k) for k in keywords]
        return gen_plain_prompt(keywords, sep=self.keysep)
    
    def generate_description_dynamic(self,item):
        desc = item.get('desc', 'none')
        if desc is None:
            desc = 'none'
        desc = desc.strip()
        if len(desc) > self.MAX_PROMPT_LEN:
            shorter_desc = desc[:self.MAX_PROMPT_LEN]
            # find last stop
            stop_idx = shorter_desc.rfind('.')
            if stop_idx == -1:
                stop_idx = shorter_desc.rfind('!')
            if stop_idx == -1:
                stop_idx = shorter_desc.rfind(',')
            if stop_idx == -1:
                stop_idx = self.MAX_PROMPT_LEN - 1
            desc = desc[:stop_idx+1]
        return desc

class SoundDataset(Dataset):
    def __init__(self, 
                metadata_index: str,
                duration:float = 10, 
                min_non_silent_duration:float = 3,
                sr:int = 0, 
                return_path = False, 
                return_audio = True, 
                ):
        self.data = read_jsonlike(metadata_index)
        self.sr = sr
        self.reader = SafeAudioReader(duration, sr)
        self.duration = duration
        self.min_non_silent_duration = min_non_silent_duration
        self.return_audio = return_audio
        self.return_path = return_path

    def __getitem__(self, index):
        item = self.data[index]
        if self.return_audio:
            origin_duration = item['duration']
            if origin_duration < self.min_non_silent_duration:
                audio = self.read_and_repeat_and_pad(item)
            else: 
                audio = self.reader(item['path'], item['sample_rate'], origin_duration)
        else:
            audio = None
        desc = item['caption']
        if self.return_path:
            return audio, desc, item['path']
        else:
            return audio, desc

    def __len__(self):
        return len(self.data)
    
    def read_and_repeat_and_pad(self, item):
        path = item['path']
        try:
            # read
            clip, sr = torchaudio.load(path)
            if len(clip.shape) > 1:
                clip = torch.mean(clip, dim=0, keepdim=True)
            clip = resample(clip, sr, self.sr)
            #repeat
            n_repeats = math.ceil(self.min_non_silent_duration/item['duration']) 
            clip = torch.repeat_interleave(clip, n_repeats, dim=0).reshape(-1)
            #pad
            n_samples = int(self.duration * self.sr)
            if clip.shape[0] >= n_samples:
                audio = clip[:n_samples]
            else:
                audio = torch.zeros(int(self.duration * self.sr), dtype=clip.dtype)
                start_pos = np.random.randint(0, max(0,(n_samples - clip.shape[0])))
                audio[start_pos:start_pos+clip.shape[0]] = clip
            return audio

        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            wav = torch.zeros(int(self.duration * self.sr), dtype=torch.float32)
        return wav

class CombinedDataset(Dataset):
    @beartype
    def __init__(self, datasets: Sequence[Dataset], ratios: Sequence[int]):
        self.datasets = datasets
        self.datasets_index = []

        for i,dataset in enumerate(datasets):
            if dataset is None:
                continue
            for dup in range(ratios[i]):
                for j in range(len(dataset)):
                    self.datasets_index.append((i,j))
       
    def __len__(self):
        return len(self.datasets_index)

    def __getitem__(self, idx):
        index = self.datasets_index[idx]
        i,j = index
        return self.datasets[i][j]

class CombinedDataset_random(Dataset):
    @beartype
    def __init__(self, num_examples:int, datasets: Sequence[Dataset], ratios: Sequence[int]):
        self.datasets = datasets
        self.datasets_index = []

        for i,dataset in enumerate(datasets):
            if dataset is None:
                continue
            for dup in range(ratios[i]):
                for j in range(len(dataset)):
                    self.datasets_index.append((i,j))
        
        if num_examples > 0:
            self.random_choose = True
            self.dataset_len = num_examples
        else:
            self.random_choose = False
            self.dataset_len = len(self.datasets_index)
       
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        first_try = True
        try_cnt = 0
        while True:
            try:
                if(self.random_choose or not first_try):
                    index2 = []
                    index2.append(np.random.randint(0,len(self.datasets)))
                    index2.append(np.random.randint(0,len(self.datasets[index2[-1]])))
                else:
                    index2 = self.datasets_index[idx]
                first_try = False
                out = list(self.datasets[index2[0]][index2[1]])
                return out
            except:
                print("Error loadding ", index2)
                try_cnt += 1
                if(try_cnt>10):
                    raise ValueError()
    
class SoundMixedDataset(Dataset):
    @staticmethod
    def music_desc(desc):
        return f'Music:<{desc}>'
    @staticmethod
    def sound_desc(desc):
        return f'Effect:<{desc}>'

    def __init__(self, 
                 music_dataset: Dataset, 
                 sound_dataset: Dataset, 
                 mixed_ratios: Tuple[float, float, float] = (0.3, 0.3, 0.4)  # 只有音乐：只有音效：音乐音效混合 的比例
                 ) -> None:
        self.music_dataset = music_dataset
        self.sound_dataset = sound_dataset
        music_r, sound_r, mix_r = [r/sum(mixed_ratios) for r in mixed_ratios] #化为0-1间的比例
        #三个概率区间的左端点
        self.music_anchor = 0
        self.sound_anchor = music_r
        self.mix_anchor = music_r + sound_r

    def __len__(self):
        return len(self.music_dataset)
    
    def get_random_sound_data(self):
        idx = random.randint(0, len(self.sound_dataset)-1)
        return self.sound_dataset[idx]

    def __getitem__(self, idx):
        p = random.random()
        if p >= self.mix_anchor:
            music, m_desc = self.music_dataset[idx]
            sound, s_desc = self.get_random_sound_data()
            audio = music + sound
            if(audio.abs().max()>1.0):
                music = music / audio.abs().max() * 0.95
                audio = audio / audio.abs().max() * 0.95
            desc = self.music_desc(m_desc) + self.sound_desc(s_desc)
            return audio[None,:], music[None,:], desc
        elif p >= self.sound_anchor:
            audio, desc = self.get_random_sound_data()
            return audio[None,:], torch.zeros_like(audio[None,:]), self.sound_desc(desc)
        else:
            audio, desc = self.music_dataset[idx]
            return audio[None,:], audio[None,:], self.music_desc(desc)


class DecoTagDataset(Dataset):
    '''这个类把普通的datatset包装成适用于标签解耦学习的dataset'''

    TAG_TYPES = ('genre', 'mood', 'insrument')

    def __init__(self, dataset_class: type, tag_map: Dict[str, str], *args, **kwargs):
        self.datasets = []
        for i, tag_t in enumerate(self.TAG_TYPES):
            kwargs['tag_types'] = [tag_map[tag_t]]
            kwargs['return_audio'] = (i == 0) #只有第0个需要返回音频和文本，其余只需要返回文本
            self.datasets.append(dataset_class(*args, **kwargs))
        
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        audio, text = self.datasets[0][idx]
        texts = (text, self.datasets[1][idx][1], self.datasets[2][idx][1])
        return audio, texts
    

class DecoTagWrapper:
    '''这是一个包装器，便于选择是否使用标签解耦学习'''
    def __init__(self, dataset_class: Dataset, deco_tag_types: List[str] = list(), switch_on: bool = False):
        self.dataset_class = dataset_class
        self.tag_map = dict(zip(DecoTagDataset.TAG_TYPES, deco_tag_types))
        self.switch_on = switch_on

    def __call__(self, *args, **kwargs):
        if self.switch_on:
            return DecoTagDataset(self.dataset_class, self.tag_map, *args, **kwargs)
        else:
            return self.dataset_class(*args, **kwargs)

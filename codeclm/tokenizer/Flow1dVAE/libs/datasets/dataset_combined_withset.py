from torch.utils.data import Dataset
from beartype.typing import Sequence, Callable, Optional, Dict, Tuple, List
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

class Read_and_PadCrop_Normalized_T(torch.nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.prob = {"is_start":0.2, "is_end":0.9}
        self.shift_secs = 5

    def __call__(self, filename: str, duration: float, cur_sample_rate: int) -> Tuple[torch.Tensor, float, float, int, int]:
        if(duration<(float(self.n_samples)/self.sample_rate+1)):
            raise ValueError(duration,float(self.n_samples),self.sample_rate)
            chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
            t_start = 0.
            t_end = min(1.0, float(self.n_samples) / float(self.sample_rate) / duration)
            offset = 0
            is_start = True
            is_end = True
        else:
            prob = random.uniform(0,1)
            if(prob<self.prob['is_start']):
                is_start = True
                is_end = False
                offset = 0
            elif(prob>self.prob['is_end']):
                is_start = False
                is_end = True
                offset = int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate)
            else:
                is_start = False
                is_end = False
                offset = np.random.randint(self.shift_secs*cur_sample_rate, \
                    int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate)-self.shift_secs*cur_sample_rate)
            t_start = offset / float(cur_sample_rate) / duration
            t_end = t_start + float(self.n_samples) / float(self.sample_rate) / duration
            chunk, _ = torchaudio.load(filename, frame_offset=offset, num_frames=int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
        if(chunk.shape[0]>1):
            chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
        else:
            chunk = chunk[[0],:].float()
        if(cur_sample_rate!=self.sample_rate):
            # print('a:',cur_sample_rate,chunk.shape)
            chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sample_rate)
            # print('b:',self.sample_rate,chunk.shape)
        if chunk.shape[-1] != self.n_samples:
            raise ValueError(chunk.shape, self.n_samples, offset, int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
        # if chunk.shape[-1] < self.n_samples:
        #     chunk = torch.cat([chunk, torch.zeros((1, self.n_samples - chunk.shape[-1],))],-1)
        # else:
        #     chunk = chunk[:,0:self.n_samples]
        seconds_start = math.floor(offset / cur_sample_rate)
        seconds_total = math.floor(duration)

        # # In this dataset, we do not introduce zeros
        # if(is_start):
        #     chunk = torch.cat([torch.zeros(1, self.shift_secs*self.sample_rate), chunk],1)[:,0:self.n_samples]
        # elif(is_end):
        #     chunk = torch.cat([chunk, torch.zeros(1, self.shift_secs*self.sample_rate)],1)[:,self.shift_secs*self.sample_rate:]

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            is_start,
            is_end,
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
                randomize: bool = True
                ):
        self.n_samples = int(sample_rate * max(duration, 0))
        self.reader = Read_and_PadCrop_Normalized_T(n_samples=self.n_samples, sample_rate=sample_rate, randomize=randomize)
    
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
            # if origin_sample_rate is None or origin_duration is None:
            #     audio_info = torchaudio.info(filepath)
            #     origin_sample_rate = audio_info.sample_rate
            #     origin_duration = audio_info.num_frames / origin_sample_rate
            audio_info = torchaudio.info(filepath)
            origin_sample_rate = audio_info.sample_rate
            origin_duration = audio_info.num_frames / origin_sample_rate
            wav, *ignored, is_start, is_end = self.reader(filepath, origin_duration, origin_sample_rate)
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise FileNotFoundError(filepath)
        return wav, is_start, is_end


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

def read_translate(translate: Optional[Dict[str, os.PathLike]]):
    if translate is None:
        return None 
    if isinstance(translate, str):
        return read_jsonlike(translate)
    return {k: read_jsonlike(path) for k, path in translate.items()}


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
                *, 
                lang = 'en',
                return_path = False, 
                prompt_template_path: os.PathLike = None, 
                tag_types = [],
                translate:Optional[Dict[str, os.PathLike]] = None,
                ):
        self.audio_reader = SafeAudioReader(duration, sr)

        self.data_dir = data_dir
        self._load_metadata_json(json_path)   
        self.sr = sr
        self.duration = duration
        self.return_path = return_path
        self.lang = lang
    
        self.use_dynamic_prompt = prompt_template_path is not None
        if self.use_dynamic_prompt:
            self.prompt_templates = load_prompt_templates(prompt_template_path, num = len(tag_types))
            self.tag_types = tag_types

            self.translate = read_translate(translate)
        if not self.use_dynamic_prompt and self.lang != 'en':
            raise NotImplementedError
    
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

        sr, duration = get_sr_and_duration_info(item)
        audio, is_start, is_end = self.audio_reader(path, sr, duration)

        if self.return_path:
            return audio, description, path
        return audio, description, is_start, is_end
    
    def tags_to_desc(self, tag_list, tag_type) -> str:
        if self.lang == 'en':
            return tags_to_desc(tag_list)
        elif self.lang == 'zh':
            translator = self.translate[tag_type]
            translated_tag_list = [translator[tag] for tag in tag_list if tag in translator ]
            return tags_to_desc(translated_tag_list, sep='、')
    
    def generate_description(self, item):
        if self.use_dynamic_prompt:
            # dynamically generate prompt from given prompt template
            prompt_template = random.choice(self.prompt_templates)
            description = self.generate_description_dynamic(item, prompt_template)

        else:
            # use ordinary static prompt instead
            description = self.generate_description_ordinary(item)
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
        
        return prompt

    def generate_description_ordinary(self, data, thresh = 0.3):
        # Initialize the description with title and artist
        description = f'"{data["title"]+" is " if random.random() > thresh else ""}"a piece of music by {data["artist"]}'
        
        # Add genre if available
        if data["genre"] and random.random() > thresh:
            genres = ', '.join(data["genre"])
            description += f', belonging to the {genres} genres'
        
        # Add moods if available
        if data["moods"] and random.random() > thresh:
            moods = ', '.join(data["moods"])
            description += f'. This track conveys a {moods} mood'
        
        # Add instruments if available
        if data["instrument"] and random.random() > thresh:
            instruments = ', '.join(data["instrument"])
            description += f', and primarily features the following instruments: {instruments}'
        
        # Add a period to end the description
        description += '.'
        
        return description

class AudioStockDataset(Dataset):
    def __init__(self, 
                metadata_path:str, 
                duration:float=10, 
                sr:int = 0, 
                return_path = False, 
                return_audio = True, 
                prompt_template_path: os.PathLike = None, 
                tag_types = [],
                lang = 'en',
                translate:Optional[Dict[str, os.PathLike]] = None
                ):
        self.audio_reader = SafeAudioReader(duration, sr)

        self.duration = duration
        self._load_metadata(metadata_path) 
        self.sr = sr
        self.return_path = return_path
        self.return_audio = return_audio

        self.use_dynamic_prompt = prompt_template_path is not None
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
                if(item['duration']>self.duration+10):
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
            audio, is_start, is_end = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            return audio, description, path, is_start, is_end
        else:
            return audio, description, is_start, is_end
    
    def generate_description(self, item):
        if self.use_dynamic_prompt:
            # dynamically generate prompt from given prompt template
            prompt_template = random.choice(self.prompt_templates)
            description = self.generate_description_dynamic(item, prompt_template)
        else:
            # use ordinary static prompt instead
            description = self.generate_description_ordinary(item)
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
            # no strong tags, use all weak tags instead
            prompt = prompt_template.apply()
        
        return prompt
    
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

    def generate_description_ordinary(self, data, thresh = 0.3):
        if self.lang != 'en':
            raise ValueError(f'Language {self.lang} is not supported for ordinary description generation')
        description = f'a piece of music by {data["Artist"]}'
        
        # Add genre if available
        if data["Genre"] and random.random() > thresh:
            genres = ', '.join(data["Genre"])
            description += f', belonging to the {genres} genres'
        
        # Add moods if available
        if data["Tags"] and random.random() > thresh:
            tags = ', '.join(data["Tags"])
            description += f'. This track contains the tags:{tags}'

        # Add moods if available
        if data["Mood"] and random.random() > thresh:
            moods = ', '.join(data["Mood"])
            description += f'. This track conveys a {moods} mood.'
        
        # Add instruments if available
        if data["Instrument"] and random.random() > thresh:
            instruments = ', '.join(data["Instrument"])
            description += f'. and primarily features the following instruments: {instruments}'
        
        # Add a period to end the description
        description += '.'
        
        return description
    
def mp3_path_to_id(mp3_path):
    return int(
        mp3_path[mp3_path.rindex('/') + 1 : mp3_path.rindex('.mp3')]
    )
 
class TmeDataset(Dataset):
    def __init__(self, 
                data_index:str, 
                music_info:str = None, 
                duration:float = 10, 
                sr:int = 0, 
                return_path = False, 
                return_audio = True, 
                prompt_format_path: os.PathLike = None, 
                tag_types = ['*'],
                lang = 'zh',
                translate: Optional[os.PathLike] = None,
                prompt_dir: os.PathLike = None,
                ):
        self.audio_reader = SafeAudioReader(duration, sr)

        self.sr = sr
        self.duration = duration
        self.return_path = return_path
        self.return_audio = return_audio
        self.lang = lang

        self.use_ready_prompt = prompt_dir is not None

        data_index = read_jsonlike(data_index)
        data_index = [d for d in data_index if d['duration']>self.duration+10]
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
    
    def generate_description(self, data):
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
            audio, is_start, is_end = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            return audio, description, path, is_start, is_end
        else:
            return audio, description, is_start, is_end

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
        assert self.use_avoid_watermark_policy is False
        self.audio_reader = SafeAudioReader(duration, sr)

        self.duration = duration
        self._load_metadata(metadata_path, index_path) 
        self.sr = sr
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
            if item['id'] in data_ids and item['id'] not in append_ids and item["details"]["duration"] is not None and item["details"]["duration"]>self.duration+10:
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
            audio, is_start, is_end = self.audio_reader(path, sr, duration)
        else:
            audio = None
        if self.return_path:
            return audio, description, path
        return audio, description, is_start, is_end

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
    def __init__(self, 
        num_examples:int,
        datasets: Sequence[Dataset], ratios: Sequence[int]
    ):
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
                out = self.datasets[index2[0]][index2[1]]
                if(len(out[0].shape)==1):out[0]=out[0][None,:]
                return out
            except:
                print("Error loadding ", index2)
                try_cnt += 1
                if(try_cnt>10):
                    raise FileNotFoundError()

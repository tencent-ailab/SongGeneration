from torch.utils.data import Dataset
from beartype.typing import Sequence, Callable, Optional, Dict, List
from beartype.door import is_bearable
import random
import os
from torchaudio.functional import resample
import torch
import typing as tp
from pathlib import Path
import torchaudio as ta
import torch.nn.functional as F
import soundfile
import numpy as np
import json
import yaml
import random
import librosa
from loguru import logger
import re


def _av_read(filepath, seek_time=0, duration=None):
    if duration is not None:
        sr = librosa.get_samplerate(filepath)
        offset = seek_time
        num_samples = int(duration * sr)
        wav, _ = librosa.load(filepath, sr=sr, offset=offset, duration=duration)
    else:
        wav, sr = librosa.load(filepath, sr=None, offset=seek_time)

    return wav, sr

def audio_read(filepath: tp.Union[str, Path], seek_time: float = 0.,
               duration: float = -1., pad: bool = True) -> tp.Tuple[torch.Tensor, int]:
    """Read audio by picking the most appropriate backend tool based on the audio format.

    Args:
        filepath (str or Path): Path to audio file to read.
        seek_time (float): Time at which to start reading in the file.
        duration (float): Duration to read from the file. If set to -1, the whole file is read.
        pad (bool): Pad output audio if not reaching expected duration.
    Returns:
        tuple of torch.Tensor, int: Tuple containing audio data and sample rate.
    """
    fp = Path(filepath)
    if fp.suffix in ['.flac', '.ogg']:  # TODO: check if we can safely use av_read for .ogg
        # There is some bug with ffmpeg and reading flac
        info = soundfile.info(filepath)
        frames = -1 if duration <= 0 else int(duration * info.samplerate)
        frame_offset = int(seek_time * info.samplerate)
        wav, sr = soundfile.read(filepath, start=frame_offset, frames=frames, dtype=np.float32)
        assert info.samplerate == sr, f"Mismatch of sample rates {info.samplerate} {sr}"
        wav = torch.from_numpy(wav).t().contiguous()
        if len(wav.shape) == 1:
            wav = torch.unsqueeze(wav, 0)
    elif (
        fp.suffix in ['.wav', '.mp3'] and fp.suffix[1:] in ta.utils.sox_utils.list_read_formats()
        and duration <= 0 and seek_time == 0
    ):
        # Torchaudio is faster if we load an entire file at once.
        wav, sr = librosa.load(fp, sr=None, mono=True)
    else:
        wav, sr = _av_read(filepath, seek_time, duration)
    if pad and duration > 0:
        expected_frames = int(duration * sr)
        wav = F.pad(torch.tensor(wav), (0, expected_frames - wav.shape[-1]))
    if not isinstance(wav, torch.Tensor):
        wav = torch.tensor(wav)
    return wav, sr

def random_seek_read(filepath, duration):
    if duration > 0:
        total_duration = librosa.get_duration(path=filepath)
        acceptable_start = max(0, total_duration - duration)
        wav, sr = audio_read(filepath, random.uniform(0, acceptable_start), duration, pad=True)
    else: 
        wav, sr = audio_read(filepath, 0, -1, pad=False)
    return wav, sr

def safe_random_seek_read(filepath, duration, sample_rate):
    try:
        wav, sr = random_seek_read(filepath, duration)
        if sr != sample_rate:
            wav = resample(wav, sr, sample_rate)
            sr = sample_rate
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        sr = sample_rate
        wav = torch.zeros(sr * max(duration, 0), dtype=torch.float32)
    return wav, sr

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
    return {k: read_jsonlike(path) for k, path in translate.items()}

   
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


class AudioStockDataset(Dataset):
    def __init__(self, 
                num_examples:int,
                metadata_path:str, 
                duration:float=60, 
                sr:int = 0, 
                return_path = False, 
                return_audio = True, 
                prompt_template_path: os.PathLike = None, 
                tag_types = [],
                lang = 'en',
                translate:Optional[Dict[str, os.PathLike]] = None
                ):
        self.duration = duration
        self.MAX_DURATION = 360
        self._load_metadata(metadata_path) 
        if num_examples > 0:
            self.random_choose = True
            self.dataset_len = num_examples
        else:
            self.random_choose = False
            self.dataset_len = len(self.data)
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
        total_len = 0; valid_len = 0
        with open(metadata_path) as fp:
            lines = fp.readlines()
            self.data = []
            for line in lines: 
                item = json.loads(line)
                total_len += 1
                if(item['duration']>self.duration and item['duration']<self.MAX_DURATION):
                    valid_len += 1
                    self.data.append(item)
        print("Filter data from {} to {}".format(total_len, valid_len))
        self.is_info_recorded = bool('Tags' in self.data[0])

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        first_try = True
        try_cnt = 0
        while True:
            try:
                if(self.random_choose or not first_try):
                    index2 = np.random.randint(0,len(self.data))
                else:
                    index2 = idx
                first_try = False
                return self.getitem_main(index2)
            except:
                print("Error loadding ", self.data[idx]["path"])
                try_cnt += 1
                if(try_cnt>10):
                    raise ValueError()

    def getitem_main(self, idx):
        path:str = self.data[idx]["path"]
        json_path = path[:path.rfind('.')] + ".json"
        if self.is_info_recorded:
            item = self.data[idx]
        else:
            with open(json_path) as fp:
                item:dict = json.load(fp)
        description = self.generate_description(item)
        if self.return_audio:
            audio, sr = safe_random_seek_read(path, duration=self.duration, sample_rate=self.sr)
        else:
            audio = None
        if self.return_path:
            return audio, description, path
        return audio, description
    


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


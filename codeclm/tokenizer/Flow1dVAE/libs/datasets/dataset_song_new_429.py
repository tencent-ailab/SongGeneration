import re
import sys
import json 

from torch.utils.data import Dataset
import torchaudio
from torchaudio.functional import resample
import torch
import numpy as np

from torch.nn.utils.rnn import pad_sequence



def check_lryics(lyric):
    _FILTER_STRING = [
        '作词', '作曲', '编曲', '【', '策划', 
        '录音', '混音', '母带', '：', '制作', 
        '版权', '校对', '演奏', '制作', '伴奏'
    ]
    for item in _FILTER_STRING:
        if item in lyric:
            return True
    
    return False



def process_lyrics(lines):
    lyric_part = []
    timestamp_part = []
    
    timestamp_pattern = re.compile(r'\[\d+:\d+(\.\d+)?\]')

    for i, line in enumerate(lines):
        
        # 删除前几行的特定信息
        if i<10 and check_lryics(line):
            continue

        # 检查是否包含有效的时间戳和歌词内容
        if timestamp_pattern.match(line):
            timestamp_end = line.rfind(']')
            lyrics = line[timestamp_end + 1:].strip()
            timestamps = line[:timestamp_end + 1]
            
            if '：' in lyrics:
                if len(lyrics.split("：")[0]) <=5:
                     lyrics = "".join(lyrics.split("：")[1:])
            # if lyrics:  # 确保歌词部分不是空的
            #     lyric_part.append(lyrics)
            #     timestamp_part.append(timestamps)
    # print(processed_lyrics)
    return timestamp_part, lyric_part

def get_timestamps(timestamp_part):
    
    # 转换为秒

    timestamps = []
    
    for line in timestamp_part:
        match = re.match(r'\[(\d+):(\d+)(\.\d+)?\]', line)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            millis = float(match.group(3)) if match.group(3) else 0
            total_seconds = minutes * 60 + seconds + millis
            timestamps.append(total_seconds)
        
        
    return timestamps

def process_lyrics_lrc(lyrics):
    timestamp_part, lyric_part = process_lyrics(lyrics)
    # print(timestamp_part)
    # print(lyric_part)
    timestamps = get_timestamps(timestamp_part)
    # print(timestamps)
    if len(timestamps) == 0:
        # print(f'{lyric_path}')
        return []
    
    slice_start = timestamps[0]
    slice_start_idx = 0

    output_list = []
    for i in range(1, len(timestamps)):
        # 如果累积时间超过30秒，则进行切分, 如果整体小于30s, 整句会被丢掉
        if timestamps[i] - slice_start > 30:
            output_list.append(f'[{str(slice_start)}:{str(timestamps[i])}]' + ", ".join(lyric_part[slice_start_idx:i]))
                           
            slice_start = timestamps[i]
            slice_start_idx = i
    
    return output_list



def process_lyrics_yrc(lyrics):
    
    timestamps, lyric_part = extract_lrc(lyrics)
    
    # timestamp_part, lyric_part = process_lyrics(lyrics)
    # import pdb; pdb.set_trace()
    # print(timestamp_part)
    # print(lyric_part)
    # timestamps = get_timestamps(timestamp_part)
    # print(timestamps)
    if len(timestamps) == 0:
        # print(f'{lyric_path}')
        return []
    
    slice_start = timestamps[0]
    slice_start_idx = 0

    output_list = []
    for i in range(1, len(timestamps)):
        # 如果累积时间超过30秒，则进行切分
        if timestamps[i] - slice_start > 30:
            output_list.append(f'[{str(slice_start)}:{str(timestamps[i])}]' + ", ".join(lyric_part[slice_start_idx:i]))
                           
            slice_start = timestamps[i]
            slice_start_idx = i
    # import pdb; pdb.set_trace()
    return output_list

def extract_lrc(lyrics):
    timestamp_part, lyric_part = [], []
    
    for i,  text in enumerate(lyrics):
        # 提取中括号内的内容
        bracket_content = re.search(r'\[(.*?)\]', text).group(1)
        bracket_content = bracket_content.split(',')
        # 提取小括号内的内容
        parentheses_content = re.findall(r'\((.*?)\)', text)
        # 提取其他内容
        other_content = re.sub(r'\[(.*?)\]|\((.*?)\)', '', text).strip()
        
        # 数据怎么处理？
        if i<10 and check_lryics(other_content):
            continue
        timestamp_part.append(float(bracket_content[0])/1000)
        lyric_part.append(other_content)
    return timestamp_part, lyric_part



class WYYSongDataset(Dataset):
    def __init__(self, 
                metadata_path:str, 
                sr:int = 0, 
                use_lang = ['en', 'zh-cn'],
                num_examples = -1,
                max_dur = 20,
                pad_to_max= True,
                ):
        
        self.sr = sr
        self.use_lang = use_lang
        self._load_metadata(metadata_path)
        self.max_dur = max_dur
        self.pad_to_max = pad_to_max
        
        # buffer
        self.lyric_buffer = {}

        if(num_examples<=0):
            self.dataset_len = len(self.data)
            self.random_slc = False
        else:
            self.dataset_len = num_examples
            self.random_slc = True
    
    # 读取jsonl文件    
    def _load_metadata(self, metadata_path):
        with open(metadata_path) as fp:
            lines = fp.readlines()
            self.data = []
            for line in lines: 
                item = json.loads(line)
                if '伴奏' not in item['path']:
                    # if "lang_type" in item and item['lang_type'] == 'en':
                     if "lang_type" in item:
                        self.data.append(item)
        
    
    def __len__(self):
        return self.dataset_len
    
    
    def __getitem__(self, idx):
        try_cnt = 0 
        while True:
            if(self.random_slc): 
                idx = np.random.randint(0, len(self.data))
            yrc_lyrics = []
            lrc_lyrics = []
            try:
                info = self.data[idx]
                
                # audio path
                path = info["path"]
                lang_type = info["lang_type"]
                if info["lang_type"] == 'en':
                    lyrics = info['lyrics']
                else:
                    lyrics = info['lyrics_phone']
                
                # 随机选取一个lyric段落
                ly_id = torch.randint(low=1, high=len(lyrics), size=(1,))[0].item()                
                lyric = lyrics[ly_id].strip()
                
                st, et, lyric = self.parse_lyric(lyric)
                lyric = lyric.replace("\xa0", " ")
                
                lyric = " ".join(lyric.split()) 
                
                assert et - st < self.max_dur
                
                
                if info["lang_type"] == 'en':
                    # print(len(lyric.split())/(et-st))
                    assert 6 > len(lyric.split())/(et-st) > 1
                else:
                    # print(len(lyric.split())/(et-st))
                    lyric = lyric.replace("-", "")
                    assert 6 > len(lyric.split())/(et-st) > 1
                
                
                # 读取音频文件
                cur_sample_rate = torchaudio.info(path).sample_rate
                offset = int(cur_sample_rate*st)
                num_frames = int(cur_sample_rate * (et -st))
                chunk, _ = torchaudio.load(path, frame_offset=offset, num_frames=num_frames)
                # chunk = torch.zeros(1, 48000*15)
                
                # 随机选取一个channel
                if(chunk.shape[0]>1):
                    chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
                else:
                    chunk = chunk[[0],:].float()
                
                if(cur_sample_rate!=self.sr):
                    # print('a:',cur_sample_rate,chunk.shape)
                    chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sr)
                    
                if self.pad_to_max:
                    chunk = self.pad_2d_tensor(chunk, int(self.max_dur * self.sr), 0)
                        
                return chunk, lyric, et-st, path, lang_type
            except:
                    # print("Error loadding ", info["path"])
                    try_cnt += 1
                    idx  = np.random.randint(0, len(self.data))
                    if(try_cnt>20):
                        raise FileNotFoundError()
        
    def parse_lyric(self, lyric):
        pattern = r'\[(\d+\.\d+):(\d+\.\d+)\](.*)'
        match = re.search(pattern, lyric)

        start_time = float(match.group(1))
        end_time = float(match.group(2))
        content = match.group(3)
        return start_time, end_time, content
    
    def pad_2d_tensor(self, x, max_len, pad_id):
        # 获取输入 tensor 的形状
        batch_size, seq_len = x.size()
        max_len = max(max_len, seq_len)
        # 计算需要填充的长度
        pad_len = max_len - seq_len

        # 如果需要填充
        if pad_len > 0:
            # 创建填充 tensor
            pad_tensor = torch.full((batch_size, pad_len), pad_id, dtype=x.dtype, device=x.device)

            # 沿第二个维度（列）连接输入 tensor 和填充 tensor
            padded_tensor = torch.cat([x, pad_tensor], dim=1)
        else:
            # 如果不需要填充，直接返回输入 tensor
            padded_tensor = x

        return padded_tensor
    
def collect_data(data_list):
    audios =  pad_sequence([data[0].t() for data in data_list], batch_first=True, padding_value=0).transpose(1,2)
    lyrics = [data[1] for data in data_list]
    st_et = [data[2] for data in data_list]
    paths = [data[3] for data in data_list]
    lang_types = [data[4] for data in data_list]
    return audios, lyrics, st_et, lang_types
    # return audios, lyrics, st_et


def build_dataset():
    train_dataset = WYYSongDataset(
        metadata_path = "train.jsonl",
        sr = 48000,
        use_lang = ['zh-cn', 'en'],
        num_examples = 10*10000
    )

    valid_dataset = WYYSongDataset(
        metadata_path = "valid.jsonl",
        sr = 48000,
        use_lang = ['zh-cn', 'en'],
        num_examples = 500
    )

    return train_dataset, valid_dataset

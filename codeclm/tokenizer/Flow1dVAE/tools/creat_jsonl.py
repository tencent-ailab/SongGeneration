'''
TAMPLEATE = {
    "path": ""
    "duration": ""
    "sample_rate": ""
    "amplitude": null, 
    "weight": null, 
    "info_path": null
}
'''
import torchaudio
import json
from tqdm import tqdm

import torchaudio
import numpy as np
import torch, torch.nn as nn, random
from torchaudio import transforms
import os
import argparse
from tqdm import tqdm
import torchaudio
from torchaudio.transforms import Resample
from multiprocessing import Pool

def preprocess(args, wav_scp, thread_id):
    # f =  open("pretrain_tme_20230927.scp").readlines() 
    f = open("out.{}".format(thread_id), 'w')
    for line in tqdm(wav_scp):
        try:
            # import pdb; pdb.set_trace()
            line = line.strip()
            meta = torchaudio.info(line)
            duration = meta.num_frames / float(meta.sample_rate)
            sr = meta.sample_rate
            
            # json_path = line.replace(".flac", ".json")
            # with open(json_path, encoding='utf-8') as fh:
            #     data = json.load(fh)
            # duration = data['duration']
            wav_info = {
                "path": line,
                "duration": duration,
                "sample_rate": sr,
                "amplitude": None, 
                "weight": None, 
                "info_path": None
            }
            json_string = json.dumps(wav_info)
            # print(json_string)
            f.write("{}\n".format(json_string))
        except:
            print(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Deep Speaker Embedding Inference')
    parser.add_argument('--wav_scp', type=str)
    parser.add_argument('--num_thread', default=10, type=int, help='random seed')
    args = parser.parse_args()
    
    wav_scp_total = open(args.wav_scp).readlines()
    args.num_thread = min(len(wav_scp_total), args.num_thread)
    wav_scp_list = np.array_split(wav_scp_total, args.num_thread)

    p = Pool(args.num_thread)
    for thread_id, wav_scp in enumerate(wav_scp_list):
        r = p.apply_async(preprocess, (args, wav_scp, thread_id))
    p.close()
    p.join() 
    r.get()

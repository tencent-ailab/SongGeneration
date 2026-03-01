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

def preprocess(args, wav_json, thread_id):
    # f =  open("pretrain_tme_20230927.scp").readlines() 
    f = open("out.{}".format(thread_id), 'w')
    for line in tqdm(wav_json):
        try:
            # import pdb; pdb.set_trace()
            line = line.strip()
            wav_info = json.loads(line)
            meta = torchaudio.info(wav_info["path"])
            
            wav_info["num_channels"] = meta.num_channels
            json_string = json.dumps(wav_info)
            # print(json_string)
            f.write("{}\n".format(json_string))
        except:
            print(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Deep Speaker Embedding Inference')
    parser.add_argument('--wav_json', type=str)
    parser.add_argument('--num_thread', default=10, type=int, help='random seed')
    args = parser.parse_args()
    
    wav_json_total = open(args.wav_json).readlines()
    args.num_thread = min(len(wav_json_total), args.num_thread)
    wav_json_list = np.array_split(wav_json_total, args.num_thread)

    p = Pool(args.num_thread)
    for thread_id, wav_json in enumerate(wav_json_list):
        r = p.apply_async(preprocess, (args, wav_json, thread_id))
    p.close()
    p.join() 
    r.get()

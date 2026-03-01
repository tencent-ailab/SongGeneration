import torch,torchaudio
import os,sys,json
from tqdm import tqdm

#from codeclm_song_v1.codeclm.semantic_extractor.SpeechDecoder_v01.generate import Tango
from generate_4rvq import Tango
import kaldiio
from kaldiio import WriteHelper
import torch
import subprocess
import time
import sys

def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

if __name__ == "__main__":
    # Define Model
    json_path = sys.argv[1]
    outdir = sys.argv[2]
    ds = int(sys.argv[3])
    
    gpu_idx = int(os.environ['CUDA_VISIBLE_DEVICES'])
    while True:
        free_mem = get_gpu_memory()
        free_mem = free_mem[gpu_idx]
        if(free_mem > 25_000):
            print("GPU memory {}, run matrix cal".format(free_mem))
            break
        else:
            print("GPU memory {}, sleep 1min".format(free_mem))
            time.sleep(60)
    
    mus_infos = []
    with open(json_path) as f:
        for line in f:
            item = json.loads(line)
            mus_infos.append(item)

    tango = Tango(model_path = './saved/model_4rvq/model_2_fixed.safetensors', rvq_num=4)
    
    
    # Feature extraction loop
    # for i in tqdm(range(2000)):
    with WriteHelper('ark,scp:{}/token.ark,{}/token.scp'.format(outdir, outdir), write_function="pickle") as writer:
        print('ark,scp:{}/token.ark,{}/token.scp'.format(outdir, outdir))
        bar = torch.zeros(4, 16384)
        for item_idx, item in tqdm(enumerate(mus_infos)):
            try:
            # if True:
                idx = item['idx']
                # print(idx)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if(os.path.exists(item['path'])):
                        codes = tango.file2code_ds(item['path'], ds)
                    else:
                        codes = tango.file2code_ds('/mnt/share/' + item['path'], ds)
                codes = codes.cpu()
                writer(str(idx), codes)
                for i0 in range(codes.shape[-1]):
                    bar[0, codes[0, 0, i0]] += 1
                    bar[1, codes[0, 1, i0]] += 1
                    bar[2, codes[0, 2, i0]] += 1
                    bar[3, codes[0, 3, i0]] += 1
            except Exception as e:
                print(item['path'])
                # print(e.message, e.args)
                # exit(1)
                continue

            if(item_idx % 1000 == 0):
                print("=========")
                print(1 - (bar[0]==0).sum() / bar.shape[-1])
                print("=========")

            # idx = item['idx']
            # # print(idx)
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            #     codes = tango.file2code(item['path'])
            # writer(str(idx), codes.cpu())
import torch,torchaudio
import os,sys,json
from tqdm import tqdm

#from codeclm_song_v1.codeclm.semantic_extractor.SpeechDecoder_v01.generate import Tango
from generate_2rvq import Tango
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

    tango = Tango(model_path = './saved/model_2rvq/model_2_fixed.safetensors', rvq_num=2)
    
    
    # Feature extraction loop
    # for i in tqdm(range(2000)):
    with WriteHelper('ark,scp:{}/token.ark,{}/token.scp'.format(outdir, outdir), write_function="pickle") as writer:
        print('ark,scp:{}/token.ark,{}/token.scp'.format(outdir, outdir))
        for item in tqdm(mus_infos):
            try:
            # if True:
                idx = item['idx']
                # print(idx)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if(os.path.exists(item['path'])):
                        codes = tango.file2code(item['path'])
                    else:
                        codes = tango.file2code('/mnt/share/' + item['path'])
                writer(str(idx), codes.cpu())
            except:
                print(item['path'])
                continue
            # idx = item['idx']
            # # print(idx)
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            #     codes = tango.file2code(item['path'])
            # writer(str(idx), codes.cpu())
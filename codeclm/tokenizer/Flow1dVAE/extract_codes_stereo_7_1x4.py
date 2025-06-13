import torch,torchaudio
import os,sys,json
from tqdm import tqdm

#from codeclm_song_v1.codeclm.semantic_extractor.SpeechDecoder_v01.generate import Tango
from generate_4rvq import Tango
import kaldiio
from kaldiio import WriteHelper

if __name__ == "__main__":
    # Define Model
    json_path = sys.argv[1]
    outdir = sys.argv[2]
    
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
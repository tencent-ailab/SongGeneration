import sys
from safetensors import safe_open
import torch

if __name__=="__main__":
    inname = sys.argv[1]
    outname = sys.argv[2]

    main_weights = {}
    with safe_open(inname, framework="pt", device="cpu") as f:
        for key in f.keys():
            main_weights[key] = f.get_tensor(key)

    torch.save(main_weights, outname)
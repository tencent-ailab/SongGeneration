import torch
import sys
from safetensors.torch import load_file

if __name__ == "__main__":
    m0, m1 = sys.argv[1], sys.argv[2]
    m0 = load_file(m0)
    m1 = load_file(m1)
    
    ks = [k for k in m0.keys() if 'bestrq' in k]
    for k in ks:
        print(k, (m0[k] - m1[k]).abs().sum())
        
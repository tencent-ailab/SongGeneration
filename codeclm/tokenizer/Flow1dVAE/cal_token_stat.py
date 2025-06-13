import kaldiio
from tqdm import tqdm
import torch

if __name__ == "__main__":
    bar = torch.zeros(1, 16384)
    with open('token.scp', 'r') as f:
        for item_idx, line in tqdm(enumerate(f)):
            idx, pos = line.strip().split()
            codes = kaldiio.load_mat(pos)
            for i0 in range(codes.shape[-1]):
                bar[0, codes[0, 0, i0]] += 1
            if(item_idx % 1000 == 0):
                print("=========")
                print(1 - (bar[0]==0).sum() / bar.shape[-1])
                print("=========")
        print("=========")
        print(1 - (bar[0]==0).sum() / bar.shape[-1])
        print("=========")
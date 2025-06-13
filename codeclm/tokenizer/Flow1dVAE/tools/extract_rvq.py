import torch
import sys

if __name__=="__main__":
    p = sys.argv[1]
    bd = '/'.join(p.split('/')[:-1])
    bn = p.split('/')[-1]

    d = {}
    m = torch.load(p, map_location='cpu')
    for k in m.keys():
        if('rvq' in k):
            d[k] = m[k]

    torch.save(d, '{}/rvq.bin'.format(bd))
import torch
import sys

if __name__=="__main__":
    m1, m2 = sys.argv[1:3]
    m1 = torch.load(m1, map_location = 'cpu')
    m2 = torch.load(m2, map_location = 'cpu')
    m1_keys = set(m1.keys())
    m2_keys = set(m2.keys())

    m1_uniq_keys = m1_keys - m2_keys
    m2_uniq_keys = m2_keys - m1_keys
    m12_shared_keys = m1_keys & m2_keys

    print("m1_uniq_keys: ", m1_uniq_keys)
    print("m2_uniq_keys: ", m2_uniq_keys)
    print("m12_shared_keys but different: ")
    for k in m12_shared_keys:
        if(m1[k].numel() != m2[k].numel()):
            print(k,m1[k].shape,m2[k].shape)

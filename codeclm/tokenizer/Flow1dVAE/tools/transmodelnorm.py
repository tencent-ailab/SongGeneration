import torch

if __name__=="__main__":
    src_ckpt = 'saved/train_mulan_v3_48k_everything3/latest/pytorch_model_2.bin'
    tgt_ckpt = 'saved/train_mulan_v3_48k_everything3_sepnorm/src_pytorch_model_2.bin'
    # src_ckpt = 'saved/train_enhcodec2D_again/latest/pytorch_model_3.bin'
    # tgt_ckpt = 'saved/train_enhcodec2D_again_sepnorm/pytorch_model_3.bin'

    ckpt = torch.load(src_ckpt, map_location='cpu')

    ckpt['normfeat.sum_x'] = torch.ones(16, 32, dtype=ckpt['normfeat.sum_x'].dtype) * ckpt['normfeat.sum_x'] / ckpt['normfeat.counts']
    ckpt['normfeat.sum_x2'] = torch.ones(16, 32, dtype=ckpt['normfeat.sum_x2'].dtype) * ckpt['normfeat.sum_x2'] / ckpt['normfeat.counts']
    ckpt['normfeat.sum_target_x2'] = torch.ones(16, 32, dtype=ckpt['normfeat.sum_target_x2'].dtype) * ckpt['normfeat.sum_target_x2'] / ckpt['normfeat.counts']
    ckpt['normfeat.counts'] = torch.ones_like(ckpt['normfeat.counts'])
    torch.save(ckpt, tgt_ckpt)
    
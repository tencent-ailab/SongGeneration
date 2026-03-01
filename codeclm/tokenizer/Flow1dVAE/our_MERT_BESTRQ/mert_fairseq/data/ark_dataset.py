import logging
import torch
import torch.nn.functional as F
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from typing import Tuple
try:
    import kaldiio
except:
    kaldiio = None
import warnings

logger = logging.getLogger(__name__)


class ArkDataset(RawAudioDataset):
    def __init__(
        self,
        wav_scp, 
        dur_scp,
        sr = 24000,
        max_dur = 20,
        num_buckets=0,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sr,
            max_sample_size=max_dur*sr,
            min_sample_size=1200,
            shuffle=True,
            pad=True,
            normalize=normalize,
            compute_mask=False,
        )
        self.sr = sr 
        self.max_dur = max_dur
        self.normalize = normalize

        logger.info("Loading Kaldi scp files from {}".format(wav_scp))

        self.wav_data = kaldiio.load_scp(wav_scp)
        self.keys = list(self.wav_data.keys())
        dur_data = {}
        keys_set = set(self.keys)
        
        with open(dur_scp, 'r') as f:
            for line in f:
                line = line.strip().split()
                if line[0] in keys_set:
                    dur_data[line[0]] = float(line[-1])
        self.sizes = [int(dur_data[k]*self.sr/100) for k in self.keys]

        logger.info("Loading Kaldi scp files done")

        self.dataset_len = len(self.keys)
        self.set_bucket_info(num_buckets)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # print("getitem idx: ", idx)
        try_cnt = 0 
        while True:
            idx = idx + try_cnt
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    key = self.keys[idx]
                    # print(self.wav_data[key].keys())
                    wav = self.wav_data[key]['wav'] 
                
                wav = torch.from_numpy(wav).float()       
                wav = self.postprocess(wav)
                # print("success load", idx, " shape =", wav.shape)
                return {"id": idx, "source": wav}
            except Exception as e:
                # from traceback import print_exc
                # print_exc()
                # print("Error loadding ", idx)
                # return {"id": idx, "source": None}
                try_cnt += 1
                if try_cnt > 50:
                    return {"id": idx, "source": None}
                continue

    def size(self, idx):
        return self.sizes[idx]
    
    def postprocess(self, wav):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
    
    def collater(self, samples):
        # print("collate from:", [s['source'].shape for s in samples if s['source'] is not None])
        return super().collater(samples)

if __name__ == '__main__':
    import torch
    raw_tensor_str = torch.Tensor.__repr__
    torch.Tensor.__str__ = torch.Tensor.__repr__ = lambda self: f'Tensor{{Size({[*self.shape]}) {self.device} {str(self.dtype)[6]}{str(self.dtype)[-2:]}}}' if self.numel() > 10 else raw_tensor_str(self)

    ds = ArkDataset(
        wav_scp='data/ark_demo/wav_ark.scp',
        dur_scp='data/ark_demo/dur_ark.scp',
        sr=24000,
    )

    for i in range(len(ds)):
        print(ds[i])
import os, sys
from transformers import AutoModel
import torch
from torch import nn
import torchaudio.transforms as T
import einops
import numpy as np
import joblib
from torch.nn.utils.rnn import pad_sequence


def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """

    assert lengths.ndim == 1, lengths.ndim

    max_len = lengths.max()
    n = lengths.size(0)
    expaned_lengths = torch.arange(max_len).expand(n, max_len).to(lengths)

    return expaned_lengths >= lengths.unsqueeze(1)

class KmeansQuantizer(nn.Module):
    def __init__(self, centroids) -> None:
        super().__init__()
        if type(centroids) == np.ndarray:
            centroids = torch.from_numpy(centroids)
        # self.clusters = nn.Embedding(n_cluster, feature_dim)
        self.clusters = nn.Parameter(centroids)
    
    @classmethod
    def from_pretrained(cls, km_path):
        km_model = joblib.load(km_path)
        centroids = km_model.cluster_centers_
        return cls(centroids)

    @property
    def n_cluster(self) -> int:
        return self.clusters.shape[0]
    
    @property
    def feature_dim(self) -> int:
        return self.clusters.shape[1]
    

    def forward(self, inp: torch.Tensor):
        if inp.ndim == 3 and inp.shape[-1] == self.feature_dim:
            return self.feat2indice(inp)
        elif inp.ndim < 3:
            return self.indice2feat(inp)
        else:
            raise NotImplementedError

    def feat2indice(self, feat):
        '''
        feat: B,T,D
        '''
        batched_cluster_centers = einops.repeat(self.clusters, 'c d -> b c d', b = feat.shape[0])
        dists = torch.cdist(feat, batched_cluster_centers, p = 2)
        indices = dists.argmin(dim = -1)
        return indices 

    def indice2feat(self, indice):
        '''
        indice: B, T
        '''
        return nn.functional.embedding(input=indice, weight=self.clusters)

class MERTwithKmeans(nn.Module):
    def __init__(self, pretrained_model_name_or_path, kmeans_path=None, sampling_rate=44100, output_layer=-1, mean_pool=1) -> None:
        super().__init__()

        # assert pretrained_model_name_or_path in ["MERT-v1-95M", "MERT-v1-330M"]
        assert pretrained_model_name_or_path == "MERT-v1-330M"
        # loading our model weights
        # self.model = AutoModel.from_pretrained(f"vocal2accmpl/model/.cache/models--m-a-p--MERT-v1-95M/snapshots/8881df140a93e2ea270235b5d7be802245e3d2c6", trust_remote_code=True)
        self.model = AutoModel.from_pretrained('pretrained/models--m-a-p--MERT-v1-330M/snapshots/af10da70c94a0c849de9cc94b83e12769c4db499', trust_remote_code=True)
        # print(self.model)
        if kmeans_path is not None:
            centroids = joblib.load(kmeans_path).cluster_centers_
            self.kmeans = KmeansQuantizer(centroids) 
        else:
            self.kmeans = None

        # loading the corresponding preprocessor config
        # self.processor = Wav2Vec2FeatureExtractor.from_pretrained(f"m-a-p/{pretrained_model_name_or_path}",trust_remote_code=True)

        # make sure the sample_rate aligned
        self.sampling_rate = sampling_rate
        self.resampler = T.Resample(sampling_rate, 24000) if sampling_rate != 24000 else lambda x: x

        self.do_normalization = (pretrained_model_name_or_path == "MERT-v1-95M")
        self.output_layer = output_layer    
        self.mean_pool = mean_pool   
        assert self.mean_pool % 2 == 1 

    @torch.no_grad()
    def forward(self, input_audio, seq_len=None, apply_kmeans=True):
        '''
        input_audio: B,T
        seq_len: B,
        '''
        device = input_audio.device
        return_seq_len = True
        if seq_len is None:
            return_seq_len = False
            seq_len = [input_audio.shape[1] for _ in input_audio]
            
        input_audio = [self.resampler(x[:l]) for x, l in zip(input_audio, seq_len)]
        new_seq_len = torch.tensor([len(i) for i in input_audio], device=device)
 

        # std_inp = self.processor([x.numpy() for x in input_audio], sampling_rate=24000, return_tensors="pt", padding=True)
        
        if self.do_normalization:
            input_audio = self.zero_mean_unit_var_norm(input_audio, new_seq_len)

        padded_input = pad_sequence(input_audio, batch_first=True)
        attention_mask = ~ make_pad_mask(new_seq_len)

        # assert (~(attention_mask == std_inp['attention_mask'])).sum() == 0, f"{attention_mask}, {std_inp['attention_mask']}"
        # assert (~(padded_input.to(dtype=std_inp['input_values'].dtype) == std_inp['input_values'])).sum() == 0, f"{torch.sum((padded_input - std_inp['input_values']))}"

        outputs = self.model(input_values=padded_input, attention_mask=attention_mask, output_hidden_states=True)
        
        output = outputs['hidden_states'][self.output_layer]
        output_len = torch.round(new_seq_len.float() / 24000 * 75).long()
        # print(output_len)
        # output_len = output_len.masked_fill(output_len > output.shape[1], output.shape[1]).long()
        output = nn.functional.interpolate(output.transpose(-1,-2), output_len.max().item()).transpose(-1,-2)

        if self.mean_pool > 1:
            output_len = output_len // 3
            output = nn.functional.avg_pool1d(output.transpose(-1, -2), kernel_size=self.mean_pool, stride=self.mean_pool)
            output = output.transpose(-1,-2)
        # print(output.shape, output_len)
        # print(output.shape, output_len)


        if apply_kmeans:
            output = self.kmeans.feat2indice(output)
        
        if return_seq_len:
            return output, output_len

        return output

    

    # from transformers.models.wav2vec2.feature_extraction_wav2vec2
    # rewrite it by pytorch
    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: torch.Tensor, seq_len: torch.Tensor = None, padding_value: float = 0.0
    ) -> torch.Tensor:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if seq_len is not None:
            normed_input_values = []

            for vector, length in zip(input_values, seq_len):
                normed_slice = (vector - vector[:length].mean()) / torch.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
            # normed_input_values = torch.stack(normed_input_values, dim=0)
        else:
            normed_input_values = (input_values - input_values.mean(dim=-1, keepdim=True)) / torch.sqrt(input_values.var(dim=-1, keepdim=True) + 1e-7)

        return normed_input_values

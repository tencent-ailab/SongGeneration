# MIT License
#
# Copyright 2023 ByteDance Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import json
import random
import torch
from torch import nn
from einops import rearrange
import os

try:
    from ..modules.random_quantizer import RandomProjectionQuantizer
    from ..modules.features import MelSTFT
    from ..modules.conv import Conv2dSubsampling
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.random_quantizer import RandomProjectionQuantizer
    from modules.features import MelSTFT
    from modules.conv import Conv2dSubsampling


class MusicFM25Hz(nn.Module):
    """
    MusicFM

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        num_codebooks=1,
        codebook_dim=16,
        codebook_size=4096,
        features=["melspec_2048"],
        hop_length=240,
        n_mels=128,
        conv_dim=512,
        encoder_dim=1024,
        encoder_depth=12,
        mask_hop=0.4,
        mask_prob=0.6,
        is_flash=False,
        stat_path="./data/fma_stats.json",
        model_path="./data/pretrained_fma.pt",
        w2v2_config_path="facebook/wav2vec2-conformer-rope-large-960h-ft",
        use_rvq_target=False,
        rvq_ckpt_path=None,
    ):
        super(MusicFM25Hz, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.features = features

        # load feature mean / std stats
        import os
        if stat_path is not None and os.path.exists(stat_path):
            with open(stat_path, "r") as f:
                self.stat = json.load(f)
        else:
            self.stat = {"spec_256_cnt": 14394344256, "spec_256_mean": -23.34296658431829, "spec_256_std": 26.189295587132637, "spec_512_cnt": 28677104448, "spec_512_mean": -21.31267396860235, "spec_512_std": 26.52644536245769, "spec_1024_cnt": 57242624832, "spec_1024_mean": -18.852271129208273, "spec_1024_std": 26.443154583585663, "spec_2048_cnt": 114373665600, "spec_2048_mean": -15.638743433896792, "spec_2048_std": 26.115825961611545, "spec_4096_cnt": 228635747136, "spec_4096_mean": -11.715532502794836, "spec_4096_std": 25.763972210234062, "melspec_256_cnt": 14282760192, "melspec_256_mean": -26.962600400166156, "melspec_256_std": 36.13614100912126, "melspec_512_cnt": 14282760192, "melspec_512_mean": -9.108344167718862, "melspec_512_std": 24.71910937988429, "melspec_1024_cnt": 14282760192, "melspec_1024_mean": 0.37302579246531126, "melspec_1024_std": 18.684082325919388, "melspec_2048_cnt": 14282760192, "melspec_2048_mean": 6.768444971712967, "melspec_2048_std": 18.417922652295623, "melspec_4096_cnt": 14282760192, "melspec_4096_mean": 13.617164614990036, "melspec_4096_std": 18.08552130124525, "cqt_cnt": 9373061376, "cqt_mean": 0.46341379757927165, "cqt_std": 0.9543998080910191, "mfcc_256_cnt": 1339008768, "mfcc_256_mean": -11.681755459447485, "mfcc_256_std": 29.183186444668316, "mfcc_512_cnt": 1339008768, "mfcc_512_mean": -2.540581461792183, "mfcc_512_std": 31.93752185832081, "mfcc_1024_cnt": 1339008768, "mfcc_1024_mean": 6.606636263169779, "mfcc_1024_std": 34.151644801729624, "mfcc_2048_cnt": 1339008768, "mfcc_2048_mean": 5.281600844245184, "mfcc_2048_std": 33.12784541220003, "mfcc_4096_cnt": 1339008768, "mfcc_4096_mean": 4.7616569480166095, "mfcc_4096_std": 32.61458906894133, "chromagram_256_cnt": 1339008768, "chromagram_256_mean": 55.15596556703181, "chromagram_256_std": 73.91858278719991, "chromagram_512_cnt": 1339008768, "chromagram_512_mean": 175.73092252759895, "chromagram_512_std": 248.48485148525953, "chromagram_1024_cnt": 1339008768, "chromagram_1024_mean": 589.2947481634608, "chromagram_1024_std": 913.857929063196, "chromagram_2048_cnt": 1339008768, "chromagram_2048_mean": 2062.286388327397, "chromagram_2048_std": 3458.92657915397, "chromagram_4096_cnt": 1339008768, "chromagram_4096_mean": 7673.039107997085, "chromagram_4096_std": 13009.883158267234}

        # feature extractor
        self.preprocessor_melspec_2048 = MelSTFT(
            n_fft=2048, hop_length=hop_length, is_db=True
        )

        # random quantizer
        self.use_rvq_target = use_rvq_target
        
        seed = 142
        # two residual convolution layers + one projection layer
        self.conv = Conv2dSubsampling(
            1, conv_dim, encoder_dim, strides=[2, 2], n_bands=n_mels
        )

        # Conformer
        if is_flash:
            from modules.flash_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        else:
            from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
                Wav2Vec2ConformerEncoder,
                Wav2Vec2ConformerConfig,
            )
        import os
        if w2v2_config_path is None or not os.path.exists(w2v2_config_path):
            w2v2_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w2v2_config.json")
        config = Wav2Vec2ConformerConfig.from_pretrained(
            w2v2_config_path
        )
        config.num_hidden_layers = encoder_depth
        config.hidden_size = encoder_dim

        self.conformer = Wav2Vec2ConformerEncoder(config)

        # projection
        self.linear = nn.Linear(encoder_dim, codebook_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # cls token (used for sequence classification)
        random.seed(seed)
        self.cls_token = nn.Parameter(torch.randn(encoder_dim))

        # load model
        if model_path:
            S = torch.load(model_path)["state_dict"]
            SS = {k[6:]: v for k, v in S.items()}
            SS['quantizer_melspec_2048.random_projection'] = SS['quantizer_melspec_2048_0.random_projection']
            SS['quantizer_melspec_2048.codebook'] = SS['quantizer_melspec_2048_0.codebook']
            del SS['quantizer_melspec_2048_0.random_projection']
            del SS['quantizer_melspec_2048_0.codebook']
            unmatch = self.load_state_dict(SS, strict=False)
            if len(unmatch.missing_keys) > 0:
                print(f'Missing keys: {unmatch.missing_keys}')

    def masking(self, x):
        """random masking of 400ms with given probability"""
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop) # 9600 = 24000 * 0.4
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop) # 10 = 25Hz * 0.4

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob # Tensor{Size([3, 75]) cpu bol}
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        ) # Tensor{Size([1286400, 2]) cpu i64}
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        ) # Tensor{Size([1340, 2]) cpu i64}

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            out[key] = layer.float()(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
        return out

    def encoder(self, x):
        """2-layer conv + w2v-conformer"""
        x = self.conv(x) # [3, 128, 3000] -> [3, 750, 1024]
        out = self.conformer(x, output_hidden_states=True)
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(last_emb)
        logits = {
            key: logits[:, :, i * self.codebook_size : (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb

    @torch.no_grad()
    def normalize(self, x):
        """normalize the input audio to have zero mean unit variance"""
        for key in x.keys():
            x[key] = (x[key] - self.stat["%s_mean" % key]) / self.stat["%s_std" % key] # {'melspec_2048_cnt': 14282760192, 'melspec_2048_mean': 6.768444971712967}
        return x

    @torch.no_grad()
    def rearrange(self, x):
        """rearrange the batch to flatten every 4 steps"""
        for key in x.keys():
            if key == "chromagram":
                x[key] = rearrange(x[key], "b f t -> b t f")
            else:
                x[key] = rearrange(x[key], "b f (t s) -> b t (s f)", s=4)
        return x

    @torch.no_grad()
    def tokenize(self, x):
        out = {}
        raise NotImplementedError("tokenize is not implemented")
        
    def get_targets(self, x):
        x = self.preprocessing(x, features=self.features) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}
        x = self.normalize(x)
        x = self.rearrange(x) # -> {'melspec_2048': Tensor{Size([3, 750, 512]) cuda:0 f32}}
        target_tokens = self.tokenize(x) # -> {'melspec_2048': Tensor{Size([3, 750]) cuda:0 i64}}
        return target_tokens

    def get_predictions(self, x):
        # preprocessing
        x = self.preprocessing(x, features=["melspec_2048"])
        x = self.normalize(x) # -> {'melspec_2048': Tensor{Size([3, 128, 3000]) cuda:0 f32}}

        # encoding
        logits, hidden_emb = self.encoder(x["melspec_2048"])

        return logits, hidden_emb

    def get_latent(self, x, layer_ix=12):
        _, hidden_states = self.get_predictions(x)
        emb = hidden_states[layer_ix]
        return emb

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x) # {'melspec_2048': Tensor{Size([3, 750]) cuda:0 i64}}

        # masking
        x, masked_indices = self.masking(x) # (Tensor{Size([3, 720000]) cuda:0 f32}, Tensor{Size([1340, 2]) cpu i64})

        # forward
        logits, hidden_emb = self.get_predictions(x)

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        return logits, hidden_emb, losses, accuracies

if __name__ == "__main__":
    device = 'cuda'
    model = MusicFM25Hz(
        stat_path='msd_stats.json',
        w2v2_config_path='msd_stats.json',
        model_path=None, 
    ).to(device)
    wavs = torch.randn(3, 24000*30).to(device)
    logits, hidden_emb, losses, accuracies = model(wavs)

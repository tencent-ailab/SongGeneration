"""!
@author Yi Luo (oulyluo)
@copyright Tencent AI Lab
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint_sequential
from thop import profile, clever_format

class RMVN(nn.Module):
    """
    Rescaled MVN.
    """
    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()
        
        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0

        input = input.view(B, self.groups, -1, T)
        input_norm = (input - input.mean(2).unsqueeze(2)) / (input.var(2).unsqueeze(2) + self.eps).sqrt()
        input_norm = input_norm.view(B, N, T) * self.std.view(1, -1, 1) + self.mean.view(1, -1, 1)

        return input_norm

class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()
        
        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel-1)//2),
                                    RMVN(in_channel),
                                    nn.Conv1d(in_channel, hidden_channel*2, 1),
                                    nn.GLU(dim=1),
                                    nn.Conv1d(hidden_channel, in_channel, 1)
                                    )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel-1),
                                    RMVN(in_channel),
                                    nn.Conv1d(in_channel, hidden_channel*2, 1),
                                    nn.GLU(dim=1),
                                    nn.Conv1d(hidden_channel, in_channel, 1)
                                    )
        
    def forward(self, input):
        
        output = self.conv(input)
        if self.causal:
            output = output[...,:-self.kernel+1].contiguous()
        return input + output

class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()
        
        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel*4, kernel, causal=causal)
                                    )
        
    def forward(self, input):
        
        return self.blocks(input)
    
class ResRNN(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(ResRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.eps = torch.finfo(torch.float32).eps
        
        self.norm = RMVN(input_size)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=bidirectional)

        self.proj = nn.Linear(hidden_size*(int(bidirectional)+1), input_size)

    def forward(self, input, use_head=1):
        # input shape: batch, dim, seq

        B, N, T = input.shape

        rnn_output, _ = self.rnn(self.norm(input).transpose(1,2).contiguous())

        output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2]))
        output = output.view(B, T, -1).transpose(1,2).contiguous()
        
        return input + output

class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7, causal=False):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.seq_net = ICB(self.feature_dim, kernel=kernel, causal=causal)
        self.band_net = ResRNN(self.feature_dim, self.feature_dim*2, bidirectional=True)

    def forward(self, input):
        # input shape: B, nband, N, T

        B, nband, N, T = input.shape

        band_output = self.seq_net(input.view(B*nband, N, T)).view(B, nband, -1, T)

        # band comm
        band_output = band_output.permute(0,3,2,1).contiguous().view(B*T, -1, nband)
        output = self.band_net(band_output).view(B, T, -1, nband).permute(0,3,2,1).contiguous()

        return output.view(B, nband, N, T)
    
# https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_code, code_dim, decay=0.99, layer=0):
        super(VQEmbeddingEMA, self).__init__()

        self.num_code = num_code
        self.code_dim = code_dim
        self.decay = decay
        self.layer = layer
        self.stale_tolerance = 100
        self.eps = torch.finfo(torch.float32).eps

        embedding = torch.empty(num_code, code_dim).normal_() / ((layer+1) * code_dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("ema_count", torch.zeros(self.num_code))
        self.register_buffer("stale_counter", torch.zeros(self.num_code))

    def forward(self, input):

        B, N, T = input.shape
        assert N == self.code_dim

        input_detach = input.detach().mT.contiguous().view(B*T, N)  # B*T, dim

        # distance
        eu_dis = input_detach.pow(2).sum(-1).unsqueeze(-1) + self.embedding.pow(2).sum(-1).unsqueeze(0)  # B*T, num_code
        eu_dis = eu_dis - 2 * input_detach.mm(self.embedding.T)  # B*T, num_code

        # best codes
        indices = torch.argmin(eu_dis, dim=-1)  # B*T
        quantized = torch.gather(self.embedding, 0, indices.unsqueeze(-1).expand(-1, self.code_dim))  # B*T, dim
        quantized = quantized.view(B, T, N).mT.contiguous()  # B, N, T

        # calculate perplexity
        encodings = F.one_hot(indices, self.num_code).float()  # B*T, num_code
        avg_probs = encodings.mean(0)  # num_code
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps), -1)).mean()
        indices = indices.view(B, T)

        if self.training:
            # EMA update for codebook
            
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)  # num_code

            update_direction = encodings.T.mm(input_detach)  # num_code, dim
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * update_direction  # num_code, dim

            # Laplace smoothing on the counters
            # make sure the denominator will never be zero
            n = torch.sum(self.ema_count, dim=-1, keepdim=True)  # 1
            self.ema_count = (self.ema_count + self.eps) / (n + self.num_code * self.eps) * n  # num_code

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # calculate code usage
            stale_codes = (encodings.sum(0) == 0).float()  # num_code
            self.stale_counter = self.stale_counter * stale_codes + stale_codes

            # random replace codes that haven't been used for a while
            replace_code = (self.stale_counter == self.stale_tolerance).float() # num_code
            if replace_code.sum(-1).max() > 0:
                random_input_idx = torch.randperm(input_detach.shape[0])
                random_input = input_detach[random_input_idx].view(input_detach.shape)
                if random_input.shape[0] < self.num_code:
                    random_input = torch.cat([random_input]*(self.num_code // random_input.shape[0] + 1), 0)
                random_input = random_input[:self.num_code].contiguous()  # num_code, dim

                self.embedding = self.embedding * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_weight = self.ema_weight * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_count = self.ema_count * (1 - replace_code)
                self.stale_counter = self.stale_counter * (1 - replace_code)

        return quantized, indices, perplexity

class RVQEmbedding(nn.Module):
    def __init__(self, code_dim, decay=0.99, bit=[10]):
        super(RVQEmbedding, self).__init__()

        self.code_dim = code_dim
        self.decay = decay
        self.eps = torch.finfo(torch.float32).eps

        self.VQEmbedding = nn.ModuleList([])
        for i in range(len(bit)):
            self.VQEmbedding.append(VQEmbeddingEMA(2**bit[i], code_dim, decay, layer=i))

    def forward(self, input):
        quantized = []
        indices = []
        ppl = []

        residual_input = input
        for i in range(len(self.VQEmbedding)):
            this_quantized, this_indices, this_perplexity = self.VQEmbedding[i](residual_input)
            indices.append(this_indices)
            ppl.append(this_perplexity)
            residual_input = residual_input - this_quantized
            if i == 0:
                quantized.append(this_quantized)
            else:
                quantized.append(quantized[-1] + this_quantized)

        quantized = torch.stack(quantized, -1)
        indices = torch.stack(indices, -1)
        ppl = torch.stack(ppl, -1)
        latent_loss = 0
        for i in range(quantized.shape[-1]):
            latent_loss = latent_loss + F.mse_loss(input, quantized.detach()[...,i])

        return quantized, indices, ppl, latent_loss

class Codec(nn.Module):
    def __init__(self, nch=1, sr=44100, win=100, feature_dim=128, vae_dim=2, enc_layer=12, dec_layer=12, bit=[8]*5, causal=True):
        super(Codec, self).__init__()
        
        self.nch = nch
        self.sr = sr
        self.win = int(sr / 1000 * win)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.vae_dim = vae_dim
        self.bit = bit
        self.eps = torch.finfo(torch.float32).eps

        # 0-1k (50 hop), 1k-4k (100 hop), 4k-8k (200 hop), 8k-12k (400 hop), 12k-22k (500 hop)
        # 100 bands
        bandwidth_50 = int(np.floor(50 / (sr / 2.) * self.enc_dim))
        bandwidth_100 = int(np.floor(100 / (sr / 2.) * self.enc_dim))
        bandwidth_200 = int(np.floor(200 / (sr / 2.) * self.enc_dim))
        bandwidth_400 = int(np.floor(400 / (sr / 2.) * self.enc_dim))
        bandwidth_500 = int(np.floor(500 / (sr / 2.) * self.enc_dim))
        self.band_width = [bandwidth_50]*20
        self.band_width += [bandwidth_100]*30
        self.band_width += [bandwidth_200]*20
        self.band_width += [bandwidth_400]*10
        self.band_width += [bandwidth_500]*19
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        print(self.band_width, self.nband)

        self.VAE_BN = nn.ModuleList([])
        for i in range(self.nband):
            self.VAE_BN.append(nn.Sequential(RMVN((self.band_width[i]*2+1)*self.nch),
                                             nn.Conv1d(((self.band_width[i]*2+1)*self.nch), self.feature_dim, 1))
                          )

        self.VAE_encoder = []
        for _ in range(enc_layer):
            self.VAE_encoder.append(BSNet(self.feature_dim, kernel=7, causal=causal))
        self.VAE_encoder = nn.Sequential(*self.VAE_encoder)

        self.vae_FC = nn.Sequential(RMVN(self.nband*self.feature_dim, groups=self.nband),
                                    nn.Conv1d(self.nband*self.feature_dim, self.nband*self.vae_dim*2, 1, groups=self.nband)
                                   )
        self.codebook = RVQEmbedding(self.nband*self.vae_dim*2, bit=bit)
        self.vae_reshape = nn.Conv1d(self.nband*self.vae_dim, self.nband*self.feature_dim, 1, groups=self.nband)

        self.VAE_decoder = []
        for _ in range(dec_layer):
            self.VAE_decoder.append(BSNet(self.feature_dim, kernel=7, causal=causal))
        self.VAE_decoder = nn.Sequential(*self.VAE_decoder)
        
        self.VAE_output = nn.ModuleList([])
        for i in range(self.nband):
            self.VAE_output.append(nn.Sequential(RMVN(self.feature_dim), 
                                                 nn.Conv1d(self.feature_dim, self.band_width[i]*4*self.nch, 1), 
                                                 nn.GLU(dim=1))
                                  )
        
    def spec_band_split(self, input):

        B, nch, nsample = input.shape

        spec = torch.stft(input.view(B*nch, nsample).float(), n_fft=self.win, hop_length=self.stride, 
                          window=torch.hann_window(self.win).to(input.device), return_complex=True)

        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:,band_idx:band_idx+self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append([this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1]])  # B, BW, T
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec, subband_spec_norm, subband_power

    def feature_extractor(self, input):
        
        _, subband_spec_norm, subband_power = self.spec_band_split(input)
        
        # normalization and bottleneck
        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat([subband_spec_norm[i][0], subband_spec_norm[i][1], torch.log(subband_power[:,i].unsqueeze(1))], 1)
            concat_spec = concat_spec.view(-1, (self.band_width[i]*2+1)*self.nch, concat_spec.shape[-1])
            subband_feature.append(self.VAE_BN[i](concat_spec.type(input.type())))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature
    
    def vae_sample(self, input):

        B, nch, _ = input.shape

        subband_feature = self.feature_extractor(input)

        # encode
        enc_output = checkpoint_sequential(self.VAE_encoder, len(self.VAE_encoder), subband_feature)
        enc_output = self.vae_FC(enc_output.view(B, self.nband*self.feature_dim, -1)).view(B, self.nband, 2, self.vae_dim, -1)
        mu = enc_output[:,:,0].contiguous()
        logvar = enc_output[:,:,1].contiguous()

        # vae
        reparam_feature = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        vae_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(2)).mean()

        # quantization
        mu_var = torch.stack([mu, logvar], 1).view(B, self.nband*self.vae_dim*2, -1)
        quantized_emb, indices, ppl, latent_loss = self.codebook(mu_var.detach())

        return reparam_feature, quantized_emb, mu_var, indices, ppl, latent_loss, vae_loss
    
    def vae_decode(self, vae_feature, nsample=None):
        B = vae_feature.shape[0]
        dec_input = self.vae_reshape(vae_feature.contiguous().view(B, self.nband*self.vae_dim, -1))
        output = checkpoint_sequential(self.VAE_decoder, len(self.VAE_decoder), dec_input.view(B, self.nband, self.feature_dim, -1))
        
        est_spec = []
        for i in range(self.nband):
            this_RI = self.VAE_output[i](output[:,i]).view(B*self.nch, 2, self.band_width[i], -1)
            est_spec.append(torch.complex(this_RI[:,0].float(), this_RI[:,1].float()))
        est_spec = torch.cat(est_spec, 1)
        if nsample is not None:
            output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride, 
                                window=torch.hann_window(self.win).to(vae_feature.device), length=nsample).view(B, self.nch, -1)
        else:
            output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride, 
                                window=torch.hann_window(self.win).to(vae_feature.device)).view(B, self.nch, -1)
        
        return output.type(vae_feature.type())
        
    def forward(self, input):

        B, nch, nsample = input.shape
        assert nch == self.nch

        vae_feature, quantized_emb, mu_var, indices, ppl, latent_loss, vae_loss = self.vae_sample(input)
        output = self.vae_decode(vae_feature, nsample=nsample).view(input.shape)
        

        return output # , vae_feature, quantized_emb, mu_var, indices, ppl, latent_loss, vae_loss

def get_bsrnnvae(ckpt):
    nch = 1
    model = Codec(nch = nch, \
        win = 100, \
        feature_dim = 128, \
        vae_dim = 8, \
        bit = [14]*5, \
        causal = True)
    weight = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(weight)
    return model.eval()

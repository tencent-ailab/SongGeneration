import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# https://github.com/bshall/VectorQuantizedVAE/blob/master/model.py
class VQEmbeddingEMA(nn.Module):
    def __init__(self, nband, num_code, code_dim, decay=0.99, layer=0):
        super(VQEmbeddingEMA, self).__init__()

        self.nband = nband
        self.num_code = num_code
        self.code_dim = code_dim
        self.decay = decay
        self.layer = layer
        self.stale_tolerance = 50
        self.eps = torch.finfo(torch.float32).eps

        if layer == 0:
            embedding = torch.empty(nband, num_code, code_dim).normal_()
            embedding = embedding / (embedding.pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1) # TODO
        else:
            embedding = torch.empty(nband, num_code, code_dim).normal_() / code_dim
            embedding[:,0] = embedding[:,0] * 0 # TODO
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("ema_count", torch.zeros(self.nband, self.num_code))
        self.register_buffer("stale_counter", torch.zeros(nband, self.num_code))

    def forward(self, input):
        num_valid_bands = 1
        B, C, N, T = input.shape
        assert N == self.code_dim
        assert C == num_valid_bands

        input_detach = input.detach().permute(0,3,1,2).contiguous().view(B*T, num_valid_bands, self.code_dim)  # B*T, nband, dim
        embedding = self.embedding[:num_valid_bands,:,:].contiguous()
        # distance
        eu_dis = input_detach.pow(2).sum(2).unsqueeze(2) + embedding.pow(2).sum(2).unsqueeze(0)  # B*T, nband, num_code
        eu_dis = eu_dis - 2 * torch.stack([input_detach[:,i].mm(embedding[i].T) for i in range(num_valid_bands)], 1)  # B*T, nband, num_code

        # best codes
        indices = torch.argmin(eu_dis, dim=-1)  # B*T, nband
        quantized = []
        for i in range(num_valid_bands):
            quantized.append(torch.gather(embedding[i], 0, indices[:,i].unsqueeze(-1).expand(-1, self.code_dim)))  # B*T, dim
        quantized = torch.stack(quantized, 1)
        quantized = quantized.view(B, T, C, N).permute(0,2,3,1).contiguous()  # B, C, N, T

        # calculate perplexity
        encodings = F.one_hot(indices, self.num_code).float()  # B*T, nband, num_code
        avg_probs = encodings.mean(0)  # nband, num_code
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps), -1)).mean()

        if self.training:
            # EMA update for codebook
            
            self.ema_count[:num_valid_bands] = self.decay * self.ema_count[:num_valid_bands] + (1 - self.decay) * torch.sum(encodings, dim=0)  # nband, num_code

            update_direction = encodings.permute(1,2,0).bmm(input_detach.permute(1,0,2))  # nband, num_code, dim
            self.ema_weight[:num_valid_bands] = self.decay * self.ema_weight[:num_valid_bands] + (1 - self.decay) * update_direction  # nband, num_code, dim

            # Laplace smoothing on the counters
            # make sure the denominator will never be zero
            n = torch.sum(self.ema_count[:num_valid_bands], dim=-1, keepdim=True)  # nband, 1
            self.ema_count[:num_valid_bands] = (self.ema_count[:num_valid_bands] + self.eps) / (n + self.num_code * self.eps) * n  # nband, num_code

            self.embedding[:num_valid_bands] = self.ema_weight[:num_valid_bands] / self.ema_count[:num_valid_bands].unsqueeze(-1)

            # calculate code usage
            stale_codes = (encodings.sum(0) == 0).float()  # nband, num_code
            self.stale_counter[:num_valid_bands] = self.stale_counter[:num_valid_bands] * stale_codes + stale_codes
            print("Lyaer {}, Ratio of unused vector : {}, {:.1f}, {:.1f}".format(self.layer, encodings.sum(), stale_codes.sum()/torch.numel(stale_codes)*100., (self.stale_counter > self.stale_tolerance //2).sum() /torch.numel(self.stale_counter)*100.))

            # random replace codes that haven't been used for a while
            replace_code = (self.stale_counter[:num_valid_bands] == self.stale_tolerance).float() # nband, num_code
            if replace_code.sum(-1).max() > 0:
                random_input_idx = torch.randperm(input_detach.shape[0])
                random_input = input_detach[random_input_idx].view(input_detach.shape)
                if random_input.shape[0] < self.num_code:
                    random_input = torch.cat([random_input]*(self.num_code // random_input.shape[0] + 1), 0)
                random_input = random_input[:self.num_code,:].contiguous().transpose(0,1)  # nband, num_code, dim

                self.embedding[:num_valid_bands] = self.embedding[:num_valid_bands] * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_weight[:num_valid_bands] = self.ema_weight[:num_valid_bands] * (1 - replace_code).unsqueeze(-1) + random_input * replace_code.unsqueeze(-1)
                self.ema_count[:num_valid_bands] = self.ema_count[:num_valid_bands] * (1 - replace_code)
                self.stale_counter[:num_valid_bands] = self.stale_counter[:num_valid_bands] * (1 - replace_code)

            # TODO:
            # code constraints
            if self.layer == 0:
                self.embedding[:num_valid_bands] = self.embedding[:num_valid_bands] / (self.embedding[:num_valid_bands].pow(2).sum(-1) + self.eps).sqrt().unsqueeze(-1)
            # else:
            #     # make sure there is always a zero code
            #     self.embedding[:,0] = self.embedding[:,0] * 0
            #     self.ema_weight[:,0] = self.ema_weight[:,0] * 0

        return quantized, indices.reshape(B, T, -1), perplexity

class RVQEmbedding(nn.Module):
    def __init__(self, nband, code_dim, decay=0.99, num_codes=[1024, 1024]):
        super(RVQEmbedding, self).__init__()

        self.nband = nband
        self.code_dim = code_dim
        self.decay = decay
        self.eps = torch.finfo(torch.float32).eps
        self.min_max = [10000, -10000]
        self.bins = [256+i*8 for i in range(32)]

        self.VQEmbedding = nn.ModuleList([])
        for i in range(len(num_codes)):
            self.VQEmbedding.append(VQEmbeddingEMA(nband, num_codes[i], code_dim, decay, layer=i))

    def forward(self, input):
        norm_value = torch.norm(input, p=2, dim=-2) # b c t
        if(norm_value.min()<self.min_max[0]):self.min_max[0]=norm_value.min().cpu().item()
        if(norm_value.max()>self.min_max[-1]):self.min_max[-1]=norm_value.max().cpu().item()
        print("Min-max : {}".format(self.min_max))
        norm_value = (((norm_value - 256) / 20).clamp(min=0, max=7).int() * 20 + 256 + 10).float()
        print("Min-max : {}, {}".format(norm_value.min(), norm_value.max()))
        input = torch.nn.functional.normalize(input, p = 2, dim = -2)

        quantized_list = []
        perplexity_list = []
        indices_list = []
        c = []

        residual_input = input
        for i in range(len(self.VQEmbedding)):
            this_quantized, this_indices, this_perplexity = self.VQEmbedding[i](residual_input)
            perplexity_list.append(this_perplexity)
            indices_list.append(this_indices)
            residual_input = residual_input - this_quantized
            if i == 0:
                quantized_list.append(this_quantized)
            else:
                quantized_list.append(quantized_list[-1] + this_quantized)
        
        quantized_list = torch.stack(quantized_list, -1) # b,1,1024,768,1
        perplexity_list = torch.stack(perplexity_list, -1)
        indices_list = torch.stack(indices_list, -1) # B T 1 codebooknum
        latent_loss = 0
        for i in range(quantized_list.shape[-1]):
            latent_loss = latent_loss + F.mse_loss(input, quantized_list.detach()[:,:,:,:,i])
        # TODO: remove unit norm 
        quantized_list = quantized_list / (quantized_list.pow(2).sum(2) + self.eps).sqrt().unsqueeze(2)  # unit norm

        return quantized_list, norm_value, indices_list, latent_loss

import yaml
import random
import inspect
import numpy as np
from tqdm import tqdm
import typing as tp
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from einops import repeat
from tools.torch_tools import wav_to_fbank

import diffusers
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDPMScheduler
from models.transformer_2d_flow import Transformer2DModel
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel,HubertModel
# from tools.get_mulan import get_mulan
from third_party.wespeaker.extract_embd import XVECModel
# from libs.rvq2 import RVQEmbedding
from libs.rvq.descript_quantize3_4layer_freezelayer1 import ResidualVectorQuantize

from models_gpt.models.gpt2_rope2_time_new_correct_mask_noncasual_reflow import GPT2Model
from models_gpt.models.gpt2_config import GPT2Config

from torch.cuda.amp import autocast


from our_MERT_BESTRQ.test import load_model

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        print("hidden_size:",config.hidden_size)
        print("classifier_proj_size:",config.classifier_proj_size)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class SampleProcessor(torch.nn.Module):
    def project_sample(self, x: torch.Tensor):
        """Project the original sample to the 'space' where the diffusion will happen."""
        """Project back from diffusion space to the actual sample space."""
        return z

class Feature1DProcessor(SampleProcessor):
    def __init__(self, dim: int = 100, power_std = 1., \
                 num_samples: int = 100_000, cal_num_frames: int = 600):
        super().__init__()

        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.cal_num_frames = cal_num_frames
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim))
        self.register_buffer('sum_x2', torch.zeros(dim))
        self.register_buffer('sum_target_x2', torch.zeros(dim))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        if(self.counts < 10):
            mean = torch.zeros_like(mean)
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        if(self.counts < 10):
            std = torch.ones_like(std)
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        if self.counts.item() < self.num_samples:
            self.counts += len(x)
            self.sum_x += x[:,:,0:self.cal_num_frames].mean(dim=(2,)).sum(dim=0)
            self.sum_x2 += x[:,:,0:self.cal_num_frames].pow(2).mean(dim=(2,)).sum(dim=0)
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, -1, 1)) * rescale.view(1, -1, 1)
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.std / self.target_std) ** self.power_std
        # print(rescale, self.mean)
        x = x * rescale.view(1, -1, 1) + self.mean.view(1, -1, 1)
        return x

def pad_or_tunc_tolen(prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds, len_size=77):
    if(prior_text_encoder_hidden_states.shape[1]<len_size):
        prior_text_encoder_hidden_states = torch.cat([prior_text_encoder_hidden_states, \
            torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], \
            prior_text_encoder_hidden_states.shape[2], device=prior_text_mask.device, \
            dtype=prior_text_encoder_hidden_states.dtype)],1)
        prior_text_mask = torch.cat([prior_text_mask, torch.zeros(prior_text_mask.shape[0], len_size-prior_text_mask.shape[1], device=prior_text_mask.device, dtype=prior_text_mask.dtype)],1)
    else:
        prior_text_encoder_hidden_states = prior_text_encoder_hidden_states[:,0:len_size]
        prior_text_mask = prior_text_mask[:,0:len_size]
    prior_text_encoder_hidden_states = prior_text_encoder_hidden_states.permute(0,2,1).contiguous()
    return prior_text_encoder_hidden_states, prior_text_mask, prior_prompt_embeds

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        estimator,
        mlp,
        ssl_layer
    ):
        super().__init__()
        self.sigma_min = 1e-4

        self.estimator = estimator
        self.mlp = mlp
        self.ssl_layer = ssl_layer

    @torch.inference_mode()
    def forward(self, mu, n_timesteps, temperature=1.0):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span)

    def solve_euler(self, x, latent_mask_input,incontext_x, incontext_length, t_span, mu,attention_mask, guidance_scale):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        noise = x.clone()

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in tqdm(range(1, len(t_span))):
            # print("incontext_x.shape:",incontext_x.shape)
            # print("noise.shape:",noise.shape)
            # print("t.shape:",t.shape)
            x[:,0:incontext_length,:] = (1 - (1 - self.sigma_min) * t) * noise[:,0:incontext_length,:] + t * incontext_x[:,0:incontext_length,:]
            if(guidance_scale > 1.0):

                model_input = torch.cat([ \
                    torch.cat([latent_mask_input, latent_mask_input], 0), \
                    torch.cat([incontext_x, incontext_x], 0), \
                    torch.cat([torch.zeros_like(mu), mu], 0), \
                    torch.cat([x, x], 0), \
                    ], 2)
                timestep=t.unsqueeze(-1).repeat(2)

                dphi_dt = self.estimator(inputs_embeds=model_input, attention_mask=attention_mask,time_step=timestep).last_hidden_state
                dphi_dt_uncond, dhpi_dt_cond = dphi_dt.chunk(2,0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dhpi_dt_cond - dphi_dt_uncond)
            else:
                model_input = torch.cat([latent_mask_input, incontext_x, mu, x], 2)
                timestep=t.unsqueeze(-1)
                dphi_dt = self.estimator(inputs_embeds=model_input, attention_mask=attention_mask,time_step=timestep).last_hidden_state
            
            dphi_dt = dphi_dt[: ,:, -x.shape[2]:]
            # print("dphi_dt.shape:",dphi_dt.shape)
            # print("x.shape:",x.shape)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def projection_loss(self,hidden_proj, bestrq_emb):
        bsz = hidden_proj.shape[0]

        hidden_proj_normalized = F.normalize(hidden_proj, dim=-1)
        bestrq_emb_normalized = F.normalize(bestrq_emb, dim=-1)

        proj_loss = -(hidden_proj_normalized * bestrq_emb_normalized).sum(dim=-1) 
        proj_loss = 1+proj_loss.mean()

        return proj_loss

    def compute_loss(self, x1, mu,  latent_masks,attention_mask,wav2vec_embeds, validation_mode=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_channels, mel_timesteps, n_feats)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_channels, mel_timesteps, n_feats)
        """
        b = mu[0].shape[0]
        len_x = x1.shape[2]
        # random timestep
        if(validation_mode):
            t = torch.ones([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype) * 0.5
        else:
            t = torch.rand([b, 1, 1], device=mu[0].device, dtype=mu[0].dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # print("y.shape:",y.shape)
        #self.unet(inputs_embeds=model_input, attention_mask=attention_mask,encoder_hidden_states=text_embedding,encoder_attention_mask=txt_attn_mask,time_step=timesteps).last_hidden_state
        model_input = torch.cat([*mu,y], 2)
        t=t.squeeze(-1).squeeze(-1)
        # print("model_input.shape:",model_input.shape)
        # print("attention_mask.shape:",attention_mask.shape)
        out = self.estimator(inputs_embeds=model_input, attention_mask=attention_mask,time_step=t,output_hidden_states=True)
        hidden_layer = out.hidden_states[self.ssl_layer]
        hidden_proj = self.mlp(hidden_layer)
        # print("hidden_proj.shape:",hidden_proj.shape)
        # print("mert_emb.shape:",mert_emb.shape)
        # exit()

        
        out = out.last_hidden_state
        
        out=out[:,:,-len_x:]
        # out=self.proj_out(out)

        weight = (latent_masks > 1.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() + (latent_masks < 0.5).unsqueeze(-1).repeat(1, 1, out.shape[-1]).float() * 0.01
        # print("out.shape",out.shape)
        # print("u.shape",u.shape)
        loss_re = F.mse_loss(out * weight, u * weight, reduction="sum") / weight.sum()
        # print("hidden_proj.shape:",hidden_proj.shape)
        # print("wav2vec_embeds.shape:",wav2vec_embeds.shape)
        loss_cos = self.projection_loss(hidden_proj, wav2vec_embeds)
        loss = loss_re + loss_cos * 0.5
        # print("loss_cos:",loss_cos,loss_cos.device)
        print("loss:",loss,loss.device)
        # exit()
        return loss, loss_re, loss_cos

class PromptCondAudioDiffusion(nn.Module):
    def __init__(
        self,
        num_channels,
        unet_model_name=None,
        unet_model_config_path=None,
        snr_gamma=None,
        hubert_layer=None,
        ssl_layer=None,
        uncondition=True,
        out_paint=False,
    ):
        super().__init__()

        assert unet_model_name is not None or unet_model_config_path is not None, "Either UNet pretrain model name or a config file path is required"

        self.unet_model_name = unet_model_name
        self.unet_model_config_path = unet_model_config_path
        self.snr_gamma = snr_gamma
        self.uncondition = uncondition
        self.num_channels = num_channels
        self.hubert_layer = hubert_layer
        self.ssl_layer = ssl_layer

        # https://huggingface.co/docs/diffusers/v0.14.0/en/api/schedulers/overview
        self.normfeat = Feature1DProcessor(dim=64)

        self.sample_rate = 48000
        self.num_samples_perseg = self.sample_rate * 20 // 1000
        self.rsp48toclap = torchaudio.transforms.Resample(48000, 24000)
        self.rsq48towav2vec = torchaudio.transforms.Resample(48000, 16000)
        # self.wav2vec = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", trust_remote_code=True)
        # self.wav2vec_processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", trust_remote_code=True)
        self.bestrq = load_model(
            model_dir='path/to/our-MERT/mert_fairseq',
            checkpoint_dir='checkpoint-120000.pt',
        )
        self.rsq48tobestrq = torchaudio.transforms.Resample(48000, 24000)
        self.rsq48tohubert = torchaudio.transforms.Resample(48000, 16000)
        for v in self.bestrq.parameters():v.requires_grad = False
        self.rvq_bestrq_emb = ResidualVectorQuantize(input_dim = 1024, n_codebooks = 2, codebook_size = 16_384, codebook_dim = 32, quantizer_dropout = 0.0, stale_tolerance=200)
        # for v in self.rvq_bestrq_emb.parameters():
        #     print(v)
        freeze_parameters='quantizers.0'
        for name, param in self.rvq_bestrq_emb.named_parameters():
            if freeze_parameters in name:
                param.requires_grad = False
                print("Freezing RVQ parameters:", name)
        self.hubert = HubertModelWithFinalProj.from_pretrained("huggingface_cache/models--lengyue233--content-vec-best/snapshots/c0b9ba13db21beaa4053faae94c102ebe326fd68")
        for v in self.hubert.parameters():v.requires_grad = False
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(32*32,))
        # self.xvecmodel = XVECModel()
        config = GPT2Config(n_positions=1000,n_layer=39,n_head=30,n_embd=1200)
        unet = GPT2Model(config)
        mlp =  nn.Sequential(
            nn.Linear(1200, 1024), 
            nn.SiLU(),                  
            nn.Linear(1024, 1024),      
            nn.SiLU(),                 
            nn.Linear(1024, 768)  
        )
        self.set_from = "random"
        self.cfm_wrapper = BASECFM(unet, mlp,self.ssl_layer)
        self.mask_emb = torch.nn.Embedding(3, 48)
        print("Transformer initialized from pretrain.")
        torch.cuda.empty_cache()
        # self.unet.set_attn_processor(AttnProcessor2_0())
        # self.unet.set_use_memory_efficient_attention_xformers(True)
        
        # self.start_embedding = nn.Parameter(torch.randn(1,1024))
        # self.end_embedding = nn.Parameter(torch.randn(1,1024))

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def preprocess_audio(self, input_audios, threshold=0.9):
        assert len(input_audios.shape) == 2, input_audios.shape
        norm_value = torch.ones_like(input_audios[:,0])
        max_volume = input_audios.abs().max(dim=-1)[0]
        norm_value[max_volume>threshold] = max_volume[max_volume>threshold] / threshold
        return input_audios/norm_value.unsqueeze(-1)

    def extract_wav2vec_embeds(self, input_audios,output_len):
        wav2vec_stride = 2

        wav2vec_embeds = self.hubert(self.rsq48tohubert(input_audios), output_hidden_states=True).hidden_states # 1, 4096, 1024
        # print(wav2vec_embeds)
        # print("audio.shape:",input_audios.shape)
        wav2vec_embeds_last=wav2vec_embeds[self.hubert_layer]
        # print("wav2vec_embeds_last.shape:",wav2vec_embeds_last.shape)
        wav2vec_embeds_last=torch.nn.functional.interpolate(wav2vec_embeds_last.permute(0, 2, 1), size=output_len, mode='linear', align_corners=False).permute(0, 2, 1)
        return wav2vec_embeds_last

    def extract_mert_embeds(self, input_audios):
        prompt_stride = 3
        inputs = self.clap_embd_extractor.mulan.audio.processor(self.rsp48toclap(input_audios), sampling_rate=self.clap_embd_extractor.mulan.audio.sr, return_tensors="pt")
        input_values = inputs['input_values'].squeeze(0).to(input_audios.device, dtype = input_audios.dtype)
        prompt_embeds = self.clap_embd_extractor.mulan.audio.model(input_values, output_hidden_states=True).hidden_states # batch_size, Time steps, 1024 
        mert_emb= prompt_embeds[-1]
        mert_emb = torch.nn.functional.interpolate(mert_emb.permute(0, 2, 1), size=500, mode='linear', align_corners=False).permute(0, 2, 1)

        return mert_emb
    
    def extract_bestrq_embeds(self, input_audio_0,input_audio_1,layer):
        self.bestrq.eval()
        # print("audio shape:",input_audio_0.shape)
        input_wav_mean = (input_audio_0 + input_audio_1) / 2.0
        # print("input_wav_mean.shape:",input_wav_mean.shape)
        # input_wav_mean = torch.randn(2,1720320*2).to(input_audio_0.device)
        input_wav_mean = self.bestrq(self.rsq48tobestrq(input_wav_mean), features_only = True)
        layer_results = input_wav_mean['layer_results']
        # print("layer_results.shape:",layer_results[layer].shape)
        bestrq_emb = layer_results[layer]
        bestrq_emb = bestrq_emb.permute(0,2,1).contiguous()
        #[b,t,1024] t=t/960
        #35.84s->batch,896,1024
        return bestrq_emb


    def extract_spk_embeds(self, input_audios):
        spk_embeds = self.xvecmodel(self.rsq48towav2vec(input_audios))
        spk_embeds = self.spk_linear(spk_embeds).reshape(spk_embeds.shape[0], 16, 1, 32)
        return spk_embeds

    def extract_lyric_feats(self, lyric):
        with torch.no_grad():
            try:
                text_encoder_hidden_states, text_mask, text_prompt_embeds = self.clap_embd_extractor(texts = lyric, return_one=False)
            except:
                text_encoder_hidden_states, text_mask, text_prompt_embeds = self.clap_embd_extractor(texts = [""] * len(lyric), return_one=False)
            text_encoder_hidden_states = text_encoder_hidden_states.to(self.device)
            text_mask = text_mask.to(self.device)
            text_encoder_hidden_states, text_mask, text_prompt_embeds = \
                pad_or_tunc_tolen(text_encoder_hidden_states, text_mask, text_prompt_embeds)
            text_encoder_hidden_states = text_encoder_hidden_states.permute(0,2,1).contiguous()
            return text_encoder_hidden_states, text_mask

    def extract_energy_bar(self, input_audios):
        if(input_audios.shape[-1] % self.num_samples_perseg > 0):
            energy_bar = input_audios[:,:-1 * (input_audios.shape[-1] % self.num_samples_perseg)].reshape(input_audios.shape[0],-1,self.num_samples_perseg)
        else:
            energy_bar = input_audios.reshape(input_audios.shape[0],-1,self.num_samples_perseg)
        energy_bar = (energy_bar.pow(2.0).mean(-1).sqrt() + 1e-6).log10() * 20 # B T
        energy_bar = (energy_bar / 2.0 + 16).clamp(0,16).int()
        energy_embedding = self.energy_embedding(energy_bar)
        energy_embedding = energy_embedding.view(energy_embedding.shape[0], energy_embedding.shape[1] // 2, 2, 32).reshape(energy_embedding.shape[0], energy_embedding.shape[1] // 2, 64).permute(0,2,1) # b 128 t
        return energy_embedding

    def forward(self, input_audios, lyric, latents, latent_masks, validation_mode=False, \
        additional_feats = ['spk', 'lyric'], \
        train_rvq=True, train_ssl=False,layer=5):
        if not hasattr(self,"device"):
            self.device = input_audios.device
        if not hasattr(self,"dtype"):
            self.dtype = input_audios.dtype
        device = self.device
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)
        input_audios_wav2vec = (input_audio_0 + input_audio_1) / 2.0
        # energy_embedding = self.extract_energy_bar(input_audios)
        # print("energy_embedding.shape:",energy_embedding.shape)
        # with autocast(enabled=False):
        if(train_ssl):
            self.wav2vec.train()
            wav2vec_embeds = self.extract_wav2vec_embeds(input_audios)
            self.clap_embd_extractor.train()
            prompt_embeds = self.extract_mert_embeds(input_audios)
            if('spk' in additional_feats):
                self.xvecmodel.train()
                spk_embeds = self.extract_spk_embeds(input_audios).repeat(1,1,prompt_embeds.shape[-1]//2,1)
        else:
            with torch.no_grad():
                with autocast(enabled=False):
                    bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
                    # mert_emb = self.extract_mert_embeds(input_audios_mert)
                    
                    wav2vec_embeds = self.extract_wav2vec_embeds(input_audios_wav2vec,bestrq_emb.shape[2])

                bestrq_emb = bestrq_emb.detach()
        if('lyric' in additional_feats):
            text_encoder_hidden_states, text_mask = self.extract_lyric_feats(lyric)
        else:
            text_encoder_hidden_states, text_mask = None, None

        
        if(train_rvq):
            random_num=random.random()
            if(random_num<0.6):
                rvq_layer = 1
            elif(random_num<0.8):
                rvq_layer = 2
            else:
                rvq_layer = 4
            quantized_bestrq_emb, _, _, commitment_loss_bestrq_emb, codebook_loss_bestrq_emb,_ = self.rvq_bestrq_emb(bestrq_emb,n_quantizers=rvq_layer) # b,d,t
        else:
            bestrq_emb = bestrq_emb.float()
            self.rvq_bestrq_emb.eval()
            # with autocast(enabled=False):
            quantized_bestrq_emb, _, _, commitment_loss_bestrq_emb, codebook_loss_bestrq_emb,_ = self.rvq_bestrq_emb(bestrq_emb) # b,d,t
            commitment_loss_bestrq_emb = commitment_loss_bestrq_emb.detach()
            codebook_loss_bestrq_emb = codebook_loss_bestrq_emb.detach()
            quantized_bestrq_emb = quantized_bestrq_emb.detach()

        commitment_loss = commitment_loss_bestrq_emb
        codebook_loss = codebook_loss_bestrq_emb


        alpha=1
        quantized_bestrq_emb = quantized_bestrq_emb * alpha + bestrq_emb * (1-alpha)

        # print("quantized_bestrq_emb.shape:",quantized_bestrq_emb.shape)
        # print("latent_masks.shape:",latent_masks.shape)
        # quantized_bestrq_emb = torch.nn.functional.interpolate(quantized_bestrq_emb, size=(int(quantized_bestrq_emb.shape[-1]/999*937),), mode='linear', align_corners=True)

        

        scenario = np.random.choice(['start_seg', 'other_seg'])
        if(scenario == 'other_seg'):
            for binx in range(input_audios.shape[0]):
                # latent_masks[binx,0:64] = 1
                latent_masks[binx,0:random.randint(64,128)] = 1
        quantized_bestrq_emb = quantized_bestrq_emb.permute(0,2,1).contiguous()
        # print("quantized_bestrq_emb.shape:",quantized_bestrq_emb.shape)
        # print("quantized_bestrq_emb1.shape:",quantized_bestrq_emb.shape)
        # print("latent_masks.shape:",latent_masks.shape)
        quantized_bestrq_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)


        

        if self.uncondition:
            mask_indices = [k for k in range(quantized_bestrq_emb.shape[0]) if random.random() < 0.1]
            if len(mask_indices) > 0:
                quantized_bestrq_emb[mask_indices] = 0
        # print("latents.shape:",latents.shape)
        latents = latents.permute(0,2,1).contiguous()
        latents = self.normfeat.project_sample(latents)
        latents = latents.permute(0,2,1).contiguous()
        incontext_latents = latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        attention_mask=(latent_masks > 0.5)
        B, L = attention_mask.size()
        attention_mask = attention_mask.view(B, 1, L)
        attention_mask = attention_mask * attention_mask.transpose(-1, -2)
        attention_mask = attention_mask.unsqueeze(1)
        # print("incontext_latents.shape:",incontext_latents.shape)
        # print("quantized_bestrq_emb.shape:",quantized_bestrq_emb.shape)
        latent_mask_input = self.mask_emb(latent_masks)
        #64+48+64+1024
        loss,loss_re, loss_cos = self.cfm_wrapper.compute_loss(latents, [latent_mask_input,incontext_latents, quantized_bestrq_emb],  latent_masks,attention_mask,wav2vec_embeds, validation_mode=validation_mode)
        return loss,loss_re, loss_cos, commitment_loss.mean(), codebook_loss.mean()

    def init_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def fetch_codes(self, input_audios, additional_feats,layer,rvq_num=1):
        input_audio_0 = input_audios[[0],:]
        input_audio_1 = input_audios[[1],:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.bestrq.eval()

        # bestrq_middle,bestrq_last = self.extract_bestrq_embeds(input_audios)
        # bestrq_middle = bestrq_middle.detach()
        # bestrq_last = bestrq_last.detach()
        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()

        # self.rvq_bestrq_middle.eval()
        # quantized_bestrq_middle, codes_bestrq_middle, *_ = self.rvq_bestrq_middle(bestrq_middle) # b,d,t
        # self.rvq_bestrq_last.eval()
        # quantized_bestrq_last, codes_bestrq_last, *_ = self.rvq_bestrq_last(bestrq_last) # b,d,t

        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb)
        codes_bestrq_emb = codes_bestrq_emb[:,:rvq_num,:]
        # print("codes_bestrq_emb.shape:",codes_bestrq_emb.shape)
        # exit()


        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None

        # return [codes_prompt, codes_wav2vec], [prompt_embeds, wav2vec_embeds], spk_embeds
        # return [codes_prompt_7, codes_prompt_13, codes_prompt_20, codes_wav2vec_half, codes_wav2vec_last], [prompt_embeds_7, prompt_embeds_13, prompt_embeds_20, wav2vec_embeds_half, wav2vec_embeds_last], spk_embeds
        # return [codes_bestrq_middle, codes_bestrq_last], [bestrq_middle, bestrq_last], spk_embeds
        return [codes_bestrq_emb], [bestrq_emb], spk_embeds
        # return [codes_prompt_13, codes_wav2vec_last], [prompt_embeds_13, wav2vec_embeds_last], spk_embeds
    
    @torch.no_grad()
    def fetch_codes_batch(self, input_audios, additional_feats,layer,rvq_num=1):
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.bestrq.eval()

        # bestrq_middle,bestrq_last = self.extract_bestrq_embeds(input_audios)
        # bestrq_middle = bestrq_middle.detach()
        # bestrq_last = bestrq_last.detach()
        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()

        # self.rvq_bestrq_middle.eval()
        # quantized_bestrq_middle, codes_bestrq_middle, *_ = self.rvq_bestrq_middle(bestrq_middle) # b,d,t
        # self.rvq_bestrq_last.eval()
        # quantized_bestrq_last, codes_bestrq_last, *_ = self.rvq_bestrq_last(bestrq_last) # b,d,t

        self.rvq_bestrq_emb.eval()
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb)
        # print("codes_bestrq_emb.shape:",codes_bestrq_emb.shape)
        codes_bestrq_emb = codes_bestrq_emb[:,:rvq_num,:]
        # print("codes_bestrq_emb.shape:",codes_bestrq_emb.shape)
        # exit()


        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None

        # return [codes_prompt, codes_wav2vec], [prompt_embeds, wav2vec_embeds], spk_embeds
        # return [codes_prompt_7, codes_prompt_13, codes_prompt_20, codes_wav2vec_half, codes_wav2vec_last], [prompt_embeds_7, prompt_embeds_13, prompt_embeds_20, wav2vec_embeds_half, wav2vec_embeds_last], spk_embeds
        # return [codes_bestrq_middle, codes_bestrq_last], [bestrq_middle, bestrq_last], spk_embeds
        return [codes_bestrq_emb], [bestrq_emb], spk_embeds
        # return [codes_prompt_13, codes_wav2vec_last], [prompt_embeds_13, wav2vec_embeds_last], spk_embeds

    @torch.no_grad()
    def fetch_codes_batch_ds(self, input_audios, additional_feats, layer, rvq_num=1, ds=250):
        input_audio_0 = input_audios[:,0,:]
        input_audio_1 = input_audios[:,1,:]
        input_audio_0 = self.preprocess_audio(input_audio_0)
        input_audio_1 = self.preprocess_audio(input_audio_1)

        self.bestrq.eval()

        # bestrq_middle,bestrq_last = self.extract_bestrq_embeds(input_audios)
        # bestrq_middle = bestrq_middle.detach()
        # bestrq_last = bestrq_last.detach()
        bestrq_emb = self.extract_bestrq_embeds(input_audio_0,input_audio_1,layer)
        bestrq_emb = bestrq_emb.detach()

        # self.rvq_bestrq_middle.eval()
        # quantized_bestrq_middle, codes_bestrq_middle, *_ = self.rvq_bestrq_middle(bestrq_middle) # b,d,t
        # self.rvq_bestrq_last.eval()
        # quantized_bestrq_last, codes_bestrq_last, *_ = self.rvq_bestrq_last(bestrq_last) # b,d,t

        self.rvq_bestrq_emb.eval()
        bestrq_emb = torch.nn.functional.avg_pool1d(bestrq_emb, kernel_size=ds, stride=ds)
        quantized_bestrq_emb, codes_bestrq_emb, *_ = self.rvq_bestrq_emb(bestrq_emb)
        # print("codes_bestrq_emb.shape:",codes_bestrq_emb.shape)
        codes_bestrq_emb = codes_bestrq_emb[:,:rvq_num,:]
        # print("codes_bestrq_emb.shape:",codes_bestrq_emb.shape)
        # exit()


        if('spk' in additional_feats):
            self.xvecmodel.eval()
            spk_embeds = self.extract_spk_embeds(input_audios)
        else:
            spk_embeds = None

        # return [codes_prompt, codes_wav2vec], [prompt_embeds, wav2vec_embeds], spk_embeds
        # return [codes_prompt_7, codes_prompt_13, codes_prompt_20, codes_wav2vec_half, codes_wav2vec_last], [prompt_embeds_7, prompt_embeds_13, prompt_embeds_20, wav2vec_embeds_half, wav2vec_embeds_last], spk_embeds
        # return [codes_bestrq_middle, codes_bestrq_last], [bestrq_middle, bestrq_last], spk_embeds
        return [codes_bestrq_emb], [bestrq_emb], spk_embeds
        # return [codes_prompt_13, codes_wav2vec_last], [prompt_embeds_13, wav2vec_embeds_last], spk_embeds

    @torch.no_grad()
    def inference_codes(self, codes, spk_embeds, true_latents, latent_length, additional_feats, incontext_length=127,
                  guidance_scale=2, num_steps=20,
                  disable_progress=True, scenario='start_seg'):
        classifier_free_guidance = guidance_scale > 1.0
        device = self.device
        dtype = self.dtype
        # codes_bestrq_middle, codes_bestrq_last = codes
        codes_bestrq_emb = codes[0]


        batch_size = codes_bestrq_emb.shape[0]


        quantized_bestrq_emb,_,_=self.rvq_bestrq_emb.from_codes(codes_bestrq_emb)
        # quantized_bestrq_emb = torch.nn.functional.interpolate(quantized_bestrq_emb, size=(int(quantized_bestrq_emb.shape[-1]/999*937),), mode='linear', align_corners=True)
        quantized_bestrq_emb = quantized_bestrq_emb.permute(0,2,1).contiguous()
        print("quantized_bestrq_emb.shape:",quantized_bestrq_emb.shape)
        # quantized_bestrq_emb = torch.nn.functional.interpolate(quantized_bestrq_emb, size=(int(quantized_bestrq_emb.shape[-1]/999*937),), mode='linear', align_corners=True)


        

        if('spk' in additional_feats):
            spk_embeds = spk_embeds.repeat(1,1,quantized_bestrq_emb.shape[-2],1).detach()

        num_frames = quantized_bestrq_emb.shape[1]

        num_channels_latents = self.num_channels
        shape = (batch_size,  num_frames, 64)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)



        latent_masks = torch.zeros(latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device)
        latent_masks[:,0:latent_length] = 2
        if(scenario=='other_seg'):
            latent_masks[:,0:incontext_length] = 1

        

        quantized_bestrq_emb = (latent_masks > 0.5).unsqueeze(-1) * quantized_bestrq_emb \
            + (latent_masks < 0.5).unsqueeze(-1) * self.zero_cond_embedding1.reshape(1,1,1024)
        true_latents = true_latents.permute(0,2,1).contiguous()
        true_latents = self.normfeat.project_sample(true_latents)
        true_latents = true_latents.permute(0,2,1).contiguous()
        incontext_latents = true_latents * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
        incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]


        attention_mask=(latent_masks > 0.5)
        B, L = attention_mask.size()
        attention_mask = attention_mask.view(B, 1, L)
        attention_mask = attention_mask * attention_mask.transpose(-1, -2)
        attention_mask = attention_mask.unsqueeze(1)
        latent_mask_input = self.mask_emb(latent_masks)

        if('spk' in additional_feats):
            # additional_model_input = torch.cat([quantized_bestrq_middle, quantized_bestrq_last, spk_embeds],1)
            additional_model_input = torch.cat([quantized_bestrq_emb, spk_embeds],1)
        else:
            # additional_model_input = torch.cat([quantized_bestrq_middle, quantized_bestrq_last],1)
            additional_model_input = torch.cat([quantized_bestrq_emb],1)

        temperature = 1.0
        t_span = torch.linspace(0, 1, num_steps + 1, device=quantized_bestrq_emb.device)
        latents = self.cfm_wrapper.solve_euler(latents * temperature, latent_mask_input,incontext_latents, incontext_length, t_span, additional_model_input,attention_mask,  guidance_scale)

        latents[:,0:incontext_length,:] = incontext_latents[:,0:incontext_length,:]
        latents = latents.permute(0,2,1).contiguous()
        latents = self.normfeat.return_sample(latents)
        # latents = latents.permute(0,2,1).contiguous()
        return latents

    @torch.no_grad()
    def inference(self, input_audios, lyric, true_latents, latent_length, additional_feats, guidance_scale=2, num_steps=20,
                  disable_progress=True,layer=5,scenario='start_seg',rvq_num=1):
        codes, embeds, spk_embeds = self.fetch_codes(input_audios, additional_feats,layer,rvq_num)

        latents = self.inference_codes(codes, spk_embeds, true_latents, latent_length, additional_feats, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress,scenario=scenario)
        return latents
    
    @torch.no_grad()
    def inference_rtf(self, input_audios, lyric, true_latents, latent_length, additional_feats, guidance_scale=2, num_steps=20,
                  disable_progress=True,layer=5,scenario='start_seg'):
        codes, embeds, spk_embeds = self.fetch_codes(input_audios, additional_feats,layer)
        import time
        start = time.time()
        latents = self.inference_codes(codes, spk_embeds, true_latents, latent_length, additional_feats, \
            guidance_scale=guidance_scale, num_steps=num_steps, \
            disable_progress=disable_progress,scenario=scenario)
        return latents,time.time()-start

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, dtype, device):
        divisor = 4
        shape = (batch_size, num_channels_latents, num_frames, 32)
        if(num_frames%divisor>0):
            num_frames = round(num_frames/float(divisor))*divisor
            shape = (batch_size, num_channels_latents, num_frames, 32)
        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        return latents



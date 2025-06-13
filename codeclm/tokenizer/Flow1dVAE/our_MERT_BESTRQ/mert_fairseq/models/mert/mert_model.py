# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from lib2to3.pytree import _Results
import logging
from dataclasses import dataclass, field
from re import L
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torchaudio import transforms
# from torch import autocast
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
    ConformerWav2Vec2EncoderLayer,
)
from fairseq.modules import GradMultiply, LayerNorm, MultiheadAttention
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)

import os
import math

from nnAudio import features as nnAudioFeatures
from torchaudio.transforms import MelSpectrogram
# from chroma_torch import ChromaSpectrogram

logger = logging.getLogger(__name__)

MASK_REPLACE_TYPE_CHOICES = ChoiceEnum(["in_batch", "in_sample"])
AUDIO_FEAT_EXTRACTOR_TYPE_CHOICES = ChoiceEnum(["w2v_conv", "hstft_conv", "melspec"])

@dataclass
class MERTConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )

    # parameters for feature extractors
    # cr: yinghao implementation
    audio_extract_type: AUDIO_FEAT_EXTRACTOR_TYPE_CHOICES = field(
        default="w2v_conv",
        metadata={"help": "the type of audio feature extractor used to extract audio features"},
    )
    melspec_n_bins: int = field(
        default=84, metadata={'help':'the papra meter for mel bins in the input feature extractor'}
    )
    music_conv_nmel: int = field(
        default=80, metadata={'help':'the papra meter for logmel transformation in the input feature extractor'}
    )
    music_conv_hoplen: int = field(
        default=40, metadata={'help':'the papra meter for STFT hop length in the input feature extractor, default set to 40 for 16k audio to align with HuBERT'}
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )


    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )

    # dynamic mask prob 
    mask_dynamic_prob_step: str = field(
        default="[]",
        metadata={
            "help": "string describing steps to update mask prob strategy "
            "eg, [10000, 40000, 80000, 150000,] "
            "the last step should be less than max_updates "
        },
    )
    mask_dynamic_prob: str = field(
        default="[]",
        metadata={
            "help": "string describing mask prob strategy "
            "the len() should be len(mask_dynamic_prob_step) + 1"
            "eg, [0.1, 0.2, 0.4, 0.6, 0.8]"
        },
    )
    
    # dynamic mask length
    mask_dynamic_len_step: str = field(
        default="[]",
        metadata={
            "help": "string describing steps to update mask prob strategy "
            "eg, [20000, 80000, 150000,] "
            "the last step should be less than max_updates "
        },
    )
    mask_dynamic_len: str = field(
        default="[]",
        metadata={
            "help": "string describing mask prob strategy "
            "the len() should be len(mask_dynamic_prob_step) + 1"
            "eg, [2, 5, 10, 15]"
        },
    )

    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    # replacement in mask
    mask_replace: float = field(
        default=0.0,
        metadata={"help": "probability of replacing the mask embeddings with other embeddings"},
    )
    mask_replace_type: MASK_REPLACE_TYPE_CHOICES = field(
        default="in_sample",
        metadata={"help": "the strategy of mask replacement; in_sampe or in_batch"},
    )
    mask_origin: float = field(
        default=0.0,
        metadata={"help": "probability of keeping the original embeddings at the mask position"},
    )
    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})

    # codec loss
    audio_codec_type: str = field(
        default='encodec',
        metadata={"help": "audio codec type, candidates: encodec/dac/rvq"},
    )
    audio_codec_dac_model_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path to DAC model file. By default download from internet.'}  
    )
    audio_codec_ckpt_path: Optional[str] = field(
        default=None,
        metadata={'help': 'The path to the ckpt file of codec model'}
    )

    # cqt loss 
    audio_cqt_loss_m: bool = field(
        default=False,
        metadata={"help": "whether to predict the CQT of the audio of masked pard"},
    )
    audio_cqt_bins: int = field(
        default=84,
        metadata={"help": "the bins of CQT feature"},
    )
    # mel loss 
    audio_mel_loss_m: bool = field(
        default=False,
        metadata={"help": "whether to predict the CQT of the audio of masked pard"},
    )
    audio_mel_bins: int = field(
        default=84,
        metadata={"help": "the bins of CQT feature"},
    )

    # random quantizer(BEST-RQ) loss
    audio_rq_loss_m: bool =  field(
        default=False,
        metadata={"help": "whether to predict the random quantizer code (BEST-RQ) of the audio of masked pard"},
    )
    audio_rq_loss_embed_dim: int = field(
        default=16, 
        metadata={"help": "the dimension of the embeddings of random quantizer"},
    )
    audio_rq_loss_num_codebooks: int = field(
        default=1, 
        metadata={"help": "the codebooks number of random quantizer"},
    )
    audio_rq_loss_num_embeds: int = field(
        default=8192, 
        metadata={"help": "the embeddings number of random quantizer"},
    )
    audio_rq_loss_seed: int = field(
        default=-1,
        metadata={"help": "random seed for BEST RQ"},
    )
    audio_rq_loss_use_norm: bool = field(
        default=False,
        metadata={"help": "whether to use norm of random quantizer; default to False, but it is not recommended"},
    )
    audio_rq_loss_use_chroma: bool = field(
        default=False,
        metadata={"help": "whether to use chroma feature of random quantizer"},
    )
    audio_rq_loss_seed_chroma: int = field(
        default=-1,
        metadata={"help": "random seed for chromagram in BEST RQ"},
    )


    # cqt extractor
    feature_extractor_cqt: bool = field(
        default=False,
        metadata={"help": "whether to use CQT feature as extra input of transformer"},
    )
    feature_extractor_cqt_bins: int = field(
        default=84,
        metadata={"help": "the number of bins of CQT feature as extra input of transformer"},
    )

    mixture_prob: float = field(
        default=-1.0,
        metadata={"help": "whether to do in-batch noise mixture during training"},
    )

    inbatch_noise_augment_len_range: Optional[str] = field(
        default = "[8000, 24000]",
        metadata={
            "help": (
                "the range of length of the mix-up noise augmentation, unit in smaples"
            )
        },
    )

    inbatch_noise_augment_number_range: Optional[str] = field(
        default = "[1, 3]",
        metadata={
            "help": (
                "the range of numbers of the mix-up noise augmentation"
            )
        },
    )
    inbatch_noise_augment_volume: float = field(
        default = 1.0,
        metadata={
            "help": (
                "the coefficient used to modify the volume of the noise audios wavs"
            )
        },
    )

    learnable_temp: bool = field(
        default=False,
        metadata={"help": "whether to learn nce temperature duing training"},
    )
    learnable_temp_init: float = field(
        default = 0.1,
        metadata={
            "help": (
                "initial value for the learnable tempatures"
            )
        },
    )
    learnable_temp_max: float = field(
        default = 100.0,
        metadata={
            "help": (
                "maximum scale value of the exp(learnable tempatures)"
            )
        },
    )

    chunk_nce_cal: int = field(
        default = -1,
        metadata={
            "help": (
                "maximum scale value of the exp(learnable tempatures)"
            )
        },
    )

    pretrained_weights: str = field(
        default="",
        metadata={"help": "a path of model checkpoint to initialize the weights of the model"},
    )

    random_codebook: int = field(
        default=-1,
        metadata={"help": "whether to randomly select n of the codebooks during training"},
    )

    deepnorm: bool = field(
        default=False,
        metadata={"help": "whether to use deepnorm from DeepNet"},
    )

    subln: bool = field(
        default=False,
        metadata={"help": "whether to use deepnorm from SubLN"},
    )


    emb_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply word embedding var grads by this"},
    )

    attention_relax: float = field(
        default=-1.0,
        metadata={"help": "whether to use additional relaxing scale for attention module"},
    )

    do_cnn_feat_stable_layernorm: bool = field(
        default=False,
        metadata={"help": "whether to modify and add additional non-affine layer_norm after feature and proj(feature)"},
    )

    wav_normalize: bool = field(
        default=False,
        metadata={"help": "whether to do layernorm on waveform before fed to CNN"},
    )


class model_mel_pred(torch.nn.Module):
    def __init__(self, input_dim, n_bins=84, sr=16000, freq=50, use_as_target=True):
        super().__init__()
        # self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=sr//freq, f_min=32.7,  # win_length=None
                                        f_max=None, n_mels=n_bins, window_fn=torch.hann_window, center=True,  # normalized=False,
                                        pad_mode='constant',  # pad=0,
                                        mel_scale='htk', normalized=True)  # norm=None, nrom on slaney mel_scale  power: float = 2.0,
        
        if use_as_target:
            self.fc = nn.Linear(input_dim, n_bins)

            self.criterion = nn.MSELoss()
            self.forward_dict = {
                'masked_transformer_output': self.plain_forward
            }

    def compute_mel(self, x):
        '''
        convert waveform to CQT -> [batch, bins, len] -> transpose
        finally turns out [batch, len, bins]
        '''
        # align with the padding of HuBERT model, 
        # the truncation is calculated by bruteforce search since the nnAudio padding strategy and fairseq models are different
        # x = x[..., :-560] 
        mels = torch.transpose(self.spec_layer(x), -1, -2) + 1e-5 # [batch, len, bins]
        # compute log mel
        logmel = torch.log(mels)
        # return logmel
        # # normalize
        S = (logmel - logmel.mean()) / (logmel.std() + 1e-5)
        return S

    def plain_forward(self, x):
        '''
        take input from transformer hidden states: [batch * len_seq, channel]
        output: [batch * len_seq, n_bins]
        '''
        # x = self.fc1(x)
        # x = self.bn(self.relu(x))
        # x = self.fc2(x)

        x = self.fc(x)

        return x

    def forward(self, x, forward_type='masked_transformer_output'):
        '''
        take input from transformer hidden states: [batch, len_seq, channel]
        output: [batch, len_seq, n_bins]
        '''
    
        return self.forward_dict[forward_type](x)

class model_cqt_pred(torch.nn.Module):
    def __init__(self, input_dim, n_bins=84, sr=16000, freq=50):
        super().__init__()
        self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = nnAudioFeatures.cqt.CQT(sr=sr, hop_length=sr//freq, fmin=32.7, 
                                           fmax=None, n_bins=n_bins, bins_per_octave=n_bins//7, 
                                           filter_scale=1, norm=1, window='hann', center=True, 
                                           pad_mode='constant', trainable=False, 
                                           output_format='Magnitude', verbose=True)


        # Initializing a Hubert facebook/hubert-base-ls960 style configuration
        # configuration = HubertConfig()

        # Initializing a model from the facebook/hubert-base-ls960 style configuration
        # self.hubert = HubertModel(configuration)
        # self.encoder = NewConvFeatureExtractionModel(n_fft=n_fft, hop_len=sr//(freq*8))

        # 2-layer & non-linear version TODO: 增加一个参数，可以使用两种不同的模型结构
        # self.fc1 = nn.Linear(input_dim, 1024)
        # self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, n_bins)

        # 1-layer version

        self.fc = nn.Linear(input_dim, n_bins)

        self.criterion = nn.MSELoss()
        self.forward_dict = {
            'masked_transformer_output': self.plain_forward
        }
    def compute_cqt(self, x):
        '''
        convert waveform to CQT -> [batch, bins, len] -> transpose
        '''
        # align with the padding of HuBERT model, 
        # the truncation is calculated by bruteforce search since the nnAudio padding strategy and fairseq models are different
        # x = x[..., :-560] 
        return torch.transpose(self.spec_layer(x), -1, -2)

    def keep_dim_forward(self, x):
        # if the input is conv output: [batch, channel, len_seq]
        # the output will be [batch, n_bins, len_seq]
        # z = self.spec_layer(x)
        # z = self.hubert(x)
        # x = self.encoder(x)
        # print(x.shape)
        x = self.fc1(torch.transpose(x,1,2))
        # print(x.shape)
        x = self.bn(self.relu(torch.transpose(x,1,2)))
        # print(x.shape)
        x = self.fc2(torch.transpose(x,1,2))
        # print(x.shape)
        return torch.transpose(x,1,2)

    def plain_forward(self, x):
        '''
        take input from transformer hidden states: [batch * len_seq, channel]
        output: [batch * len_seq, n_bins]
        '''
        # x = self.fc1(x)
        # x = self.bn(self.relu(x))
        # x = self.fc2(x)

        x = self.fc(x)

        return x

    def forward(self, x, forward_type='masked_transformer_output'):
        '''
        take input from transformer hidden states: [batch, len_seq, channel]
        output: [batch, len_seq, n_bins]
        '''
    
        return self.forward_dict[forward_type](x)

def quantize_vector(latent: torch.Tensor, codebook: torch.Tensor):
    """
    Symbols in comments:
    B: batch_size.
    D: latent_dim.
    C: num_latent_classes per group
    G: num of codebook groups.

    Args:
        latent: [B, D]
        codebook: [C, G, D // G]

    Returns:
        (quantized, codes, onehot).
         - quantized: [B, D]
         - codes:     [B, G]
         - onehot:    [B, G, C]
    """

    assert len(codebook.size()) == 3
    b, d = latent.size()
    c, g, _ = codebook.size()
    assert d % g == 0

    device_l, device_c = latent.device, codebook.device

    latent = latent.reshape(b, g, d // g).to(device_c)

    # [B, G, C]
    # torch.transpose(codebook, [2,1,0])
    distance = (
        # [b, g, 1]
        torch.sum(latent**2, -1, keepdim=True) -
        # [b, g, c]
        2 * torch.einsum('bgd,cgd->bgc', latent, codebook) +
        # [1, g, c]
        torch.sum(codebook.permute([2, 1, 0])**2, 0, keepdim=True))

    # [B, G]
    codes = torch.argmin(distance, dim=-1)

    # [B, G, C]
    one_hot = torch.nn.functional.one_hot(codes, c).type(codebook.dtype)
    quantized = torch.einsum('bgc,cgd->bgd', one_hot, codebook)
    quantized = torch.reshape(quantized, [b, d])
    return quantized, codes.to(device_l), one_hot

def l2norm(x, dim=-1):
    return x / x.norm(p=2, dim=dim, keepdim=True)

import librosa
CHROMA_N_BINS = 12
class WavToChroma:
    def __init__(self, sr, freq):
        self.sr = sr
        self.freq = freq
    
    def __call__(self, wav:torch.Tensor): #[Batch, Time] -> [Batch, N_Bins=12, SeqLen]
        device = wav.device
        chroma = librosa.feature.chroma_cqt(y=wav.cpu().numpy(), sr=self.sr, hop_length=int(self.sr//self.freq))
        chroma = torch.from_numpy(chroma).to(device)
        return chroma

from contextlib import contextmanager
@contextmanager
def torch_rand_seed(seed):
    if seed is not None and seed > 0:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        yield rng_state
        torch.random.set_rng_state(rng_state)
    else:
        yield None


@register_model("mert", dataclass=MERTConfig)
class MERTModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: MERTConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"MERTModel Config: {cfg}")
        self.cfg = cfg
        self.task_cfg = task_cfg


        if self.use_encodec_target:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            # TODO: add encodec module
            # Done 
            def get_encodec(cfg: MERTConfig):
                if cfg.audio_codec_type == 'encodec' or cfg.audio_codec_type is None:
                    model = EncodecModel.encodec_model_24khz()
                    model.set_target_bandwidth(6.0)
                elif cfg.audio_codec_type == 'dac':
                    import dac
                    if cfg.audio_codec_dac_model_path is None:
                        model_path = dac.utils.download(model_type="44khz")
                    else:
                        model_path = cfg.audio_codec_dac_model_path
                    model = dac.DAC.load(model_path)
                elif cfg.audio_codec_type == 'rvq':
                    from .rvq import ResidualVectorQuantize
                    model = ResidualVectorQuantize(
                        input_dim = 120, 
                        n_codebooks = 8, 
                        codebook_size = 1024, 
                        codebook_dim = 16, 
                        quantizer_dropout = 0.0,
                    )
                    if cfg.audio_codec_ckpt_path is None or not os.path.exists(cfg.audio_codec_ckpt_path):
                        logger.warning('No checkpoint path for rvq model. Assume it in inference mode.')
                    else:
                        state_dict = torch.load(cfg.audio_codec_ckpt_path, map_location="cpu")
                        model.load_state_dict(state_dict)
                    import torchaudio
                    self.rvq_preprocess = torchaudio.transforms.MelSpectrogram(
                        sample_rate = 24000,
                        n_mels = 120,
                        n_fft=2048,
                        win_length = int(24000//75),
                        hop_length = int(24000//75),
                        center = True,
                        pad_mode='constant',  # pad=0,
                        mel_scale='htk', 
                        normalized=True,
                    )
                else:
                    raise ValueError(f"Unknown audio encoder type: {cfg.codec_type}.")
                
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # model.to(device)
                return model
            
            self.encodec = get_encodec(cfg).eval() #这里添加了self.encodec 
            self.rsp_src2encodec = nn.Identity()
            for param in self.encodec.parameters():
                param.requires_grad = False
        if self.use_rq_target:
            total_n_codebooks = 0
            if cfg.audio_rq_loss_use_norm:
                self.rq_norm = l2norm
            else:
                self.rq_norm = lambda x, *args, **kw: x
            with torch_rand_seed(cfg.audio_rq_loss_seed):
                # random projectoin
                self.rq_projection = torch.nn.parameter.Parameter(
                    torch.empty(cfg.melspec_n_bins, cfg.audio_rq_loss_embed_dim * cfg.audio_rq_loss_num_codebooks),
                    requires_grad=False,
                )
                torch.nn.init.xavier_uniform_(self.rq_projection)

                # codebooks
                # [num_embeddings, num_codebooks, num_embeddings] means
                # [C, G, D] see quantize_vector
                self.rq_embeddings = torch.nn.parameter.Parameter(
                    torch.empty(cfg.audio_rq_loss_num_embeds, cfg.audio_rq_loss_num_codebooks, cfg.audio_rq_loss_embed_dim),
                    requires_grad=False,
                )
                torch.nn.init.normal_(self.rq_embeddings)
                self.rq_embeddings.data = self.rq_norm(self.rq_embeddings.data, dim=-1)
            total_n_codebooks += cfg.audio_rq_loss_num_codebooks

            if cfg.audio_rq_loss_use_chroma:
                self.rq_wav2chroma = WavToChroma(sr=task_cfg.sample_rate, freq=cfg.label_rate)
                with torch_rand_seed(cfg.audio_rq_loss_seed_chroma):
                    # random projectoin
                    self.rq_projection_chroma = torch.nn.parameter.Parameter(
                        torch.empty(CHROMA_N_BINS, cfg.audio_rq_loss_embed_dim * cfg.audio_rq_loss_num_codebooks),
                        requires_grad=False,
                    )
                    torch.nn.init.xavier_uniform_(self.rq_projection_chroma)

                    # codebooks
                    # [num_embeddings, num_codebooks, num_embeddings] means
                    # [C, G, D] see quantize_vector
                    self.rq_embeddings_chroma = torch.nn.parameter.Parameter(
                        torch.empty(cfg.audio_rq_loss_num_embeds, cfg.audio_rq_loss_num_codebooks, cfg.audio_rq_loss_embed_dim),
                        requires_grad=False,
                    )
                    torch.nn.init.normal_(self.rq_embeddings_chroma)
                    self.rq_embeddings_chroma.data = self.rq_norm(self.rq_embeddings_chroma.data, dim=-1)
                total_n_codebooks += cfg.audio_rq_loss_num_codebooks

            assert len(dictionaries) == 1, "BEST-RQ target requires a single dictionary"

            dictionaries = [dictionaries[0] for _ in range(total_n_codebooks)]

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        
        if self.cfg.feature_extractor_cqt:
            self.feature_extractor_cqt = nnAudioFeatures.cqt.CQT(sr=task_cfg.sample_rate, hop_length=task_cfg.sample_rate//50, fmin=32.7, 
                    fmax=None, n_bins=cfg.feature_extractor_cqt_bins, bins_per_octave=cfg.feature_extractor_cqt_bins//7, 
                    filter_scale=1, norm=1, window='hann', center=True, 
                    pad_mode='constant', trainable=False, 
                    output_format='Magnitude', verbose=True)

        if cfg.audio_extract_type == 'w2v_conv':
            self.feature_extractor = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
            )
            self.embed = feature_enc_layers[-1][0]
        elif cfg.audio_extract_type == 'melspec':
            self.feature_extractor_melspec = model_mel_pred(
                input_dim=None,
                n_bins=cfg.melspec_n_bins,
                sr=int(task_cfg.sample_rate),
                freq=int(cfg.label_rate),
                use_as_target=False,
                )
            self.embed = cfg.melspec_n_bins
        else:
            raise NotImplementedError

        if self.cfg.feature_extractor_cqt:
            self.embed = feature_enc_layers[-1][0] + cfg.feature_extractor_cqt_bins

        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.do_cnn_feat_stable_layernorm = cfg.do_cnn_feat_stable_layernorm

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )
        if self.post_extract_proj is not None and self.do_cnn_feat_stable_layernorm:
            self.post_proj_layer_norm = LayerNorm(cfg.encoder_embed_dim, elementwise_affine=False)
        else:
            self.post_proj_layer_norm = None


        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_replace = cfg.mask_replace
        self.mask_replace_type = cfg.mask_replace_type
        self.mask_origin = cfg.mask_origin

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.emb_grad_mult = cfg.emb_grad_mult

        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        
        self.learnable_temp = cfg.learnable_temp


        self.wav_normalize = cfg.wav_normalize
        
        if not self.learnable_temp:
            self.logit_temp = cfg.logit_temp
        else:
            self.logit_temp_list = nn.Parameter(torch.FloatTensor(len(dictionaries)))
            nn.init.constant_(self.logit_temp_list, np.log(1/cfg.learnable_temp_init))
            # self.logit_temp_list = [ 
            #     nn.Parameter(torch.tensor([np.log(1/cfg.learnable_temp_init)])) for _ in range(len(dictionaries))
            # ]

        self.learnable_temp_max = cfg.learnable_temp_max

        self.chunk_nce_cal = cfg.chunk_nce_cal

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        if (
                cfg.attention_relax > 0 or \
                cfg.deepnorm or \
                cfg.subln
            ):
            if cfg.subln:
                assert cfg.layer_norm_first
            if cfg.deepnorm:
                assert not cfg.layer_norm_first

            self.encoder = TransformerEncoder_extend(cfg)

        else:
            self.encoder = TransformerEncoder(cfg)

        if self.do_cnn_feat_stable_layernorm:
            self.layer_norm = LayerNorm(self.embed, elementwise_affine=False)
        else:
            self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        self.random_codebook = cfg.random_codebook
        
        # use all codebook
        if self.random_codebook <=0:
            if self.untie_final_proj:
                self.final_proj = nn.Linear(
                    cfg.encoder_embed_dim, final_dim * len(dictionaries)
                )
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
        else:
            assert self.random_codebook <= len(dictionaries)
            if self.untie_final_proj:
                self.final_projs = nn.ModuleList([nn.Linear(cfg.encoder_embed_dim, final_dim) for _ in range(len(dictionaries))])
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim),
                # requires_grad=False,
            )
            nn.init.uniform_(self.label_embs_concat)

        if cfg.audio_cqt_loss_m:
            logger.info("train the model with extra task: reconstruct cqt from transformer output")
            self.encoder_cqt_model = model_cqt_pred(
                input_dim=cfg.encoder_embed_dim,
                n_bins=cfg.audio_cqt_bins,
                sr=int(task_cfg.sample_rate),
                freq=int(cfg.label_rate)
                )
        if cfg.audio_mel_loss_m:
            logger.info("train the model with extra task: reconstruct mel from transformer output")
            self.encoder_mel_model = model_mel_pred(
                input_dim=cfg.encoder_embed_dim,
                n_bins=cfg.audio_mel_bins,
                sr=int(task_cfg.sample_rate),
                freq=int(cfg.label_rate))
        
        self.num_updates = 0

        self.mask_dynamic_prob_step = eval(cfg.mask_dynamic_prob_step)
        self.mask_dynamic_prob = eval(cfg.mask_dynamic_prob)

        if len(self.mask_dynamic_prob_step) > 0 and len(self.mask_dynamic_prob) > 0:
            self.initialize_dynamic_mask_prob()
        else:
            self.mask_dynamic_prob_stage = -1
    
        self.mask_dynamic_len_step = eval(cfg.mask_dynamic_len_step)
        self.mask_dynamic_len = eval(cfg.mask_dynamic_len)
        if len(self.mask_dynamic_len_step) > 0 and len(self.mask_dynamic_len) > 0:
            self.initialize_dynamic_mask_len()
        else:
            self.mask_dynamic_len_stage = -1

        self.mixture_prob = cfg.mixture_prob
        self.inbatch_noise_augment_len_range = eval(cfg.inbatch_noise_augment_len_range)
        self.inbatch_noise_augment_number_range = eval(cfg.inbatch_noise_augment_number_range)
        self.inbatch_noise_augment_volume = cfg.inbatch_noise_augment_volume
        
        if os.path.isfile(cfg.pretrained_weights):
            load_patterns = ['feature_extractor.'] # ['feature_extractor.', 'encoder.']
            logger.info(f"initialize {load_patterns} weights with given checkpoint")
            pretrained_dict = torch.load(cfg.pretrained_weights)['model']
                            
            def filter_keys(patterns, keys):
                toloads = []
                for k in keys:
                    for pattern in patterns:
                        if pattern in k:
                            toloads.append(k)
                return toloads
            modules_to_load = filter_keys(load_patterns, pretrained_dict.keys())
            logger.info(f"found modules to load: {modules_to_load}")
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modules_to_load}
            logger.info(f"extractor sample weitghts before loading:\n{self.feature_extractor.conv_layers[0][2].weight}")
            self.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"extractor sample weitghts after loading:\n{self.feature_extractor.conv_layers[0][2].weight}")
            
            if cfg.feature_grad_mult <= 0:
                self.feature_extractor.eval()
            # for param in self.feature_extractor.parameters():
            #     param.requires_grad = False
        
        
        # from collections import defaultdict
        # param_dict = defaultdict(lambda:0)
        # for name, param in self.named_parameters():
        #     name = str(name)
        #     if '.' in name:
        #         name = name[:name.index('.')]
        #     param_dict[name] += param.nelement()
                
        # print(param_dict)

    @property
    def use_encodec_target(self):
        return not self.cfg.audio_rq_loss_m
    
    @property
    def use_rq_target(self):
        return self.cfg.audio_rq_loss_m

    def inbatch_noise_augment(self, 
        target_audio: torch.Tensor, target_audio_idx: int , 
        batch_audios: torch.Tensor, # [bsz, audio_lengths]
        noise_len_min: int, noise_len_max: int, 
        n_noise_min: int, n_noise_max: int,
        noise_vol: float = 1.0):
        '''
        augmenation that leverages in-batch noise audios.
        noise_len_min and noise_len_max are the range of the lengths of noises (counted as samples)
        n_noise_min and n_noise_max are the range of number of noises,
        '''    
        # assert noise_len_max <= target_audio.shape[0] and noise_len_min >= 1 # should assert this outside?

        augmented_audio = torch.clone(target_audio)

        # exclude the target audio and use the rest as noise candidates
        noise_pool = torch.flatten(torch.cat((batch_audios[:target_audio_idx,:], batch_audios[target_audio_idx+1:,:]), dim=0))

        # n_noise = np.random.randint(n_noise_min, n_noise_max+1)
        n_noise = torch.randint(n_noise_min, n_noise_max+1, size=(1,))

        # random_start_idxs = np.random.randint(0, noise_pool.shape[0] - noise_len_max + 1, size=(n_noise,))
        # random_durations = np.random.randint(noise_len_min, noise_len_max+1, size=(n_noise,))
        random_start_idxs = torch.randint(0, noise_pool.shape[0] - noise_len_max + 1, size=(n_noise,))
        random_durations = torch.randint(noise_len_min, noise_len_max+1, size=(n_noise,))


        for noise_idx in range(n_noise):
            # augmentation_position = np.random.randint(0, target_audio.shape[0] - random_durations[noise_idx]+1, size=None)
            augmentation_position = torch.randint(0, target_audio.shape[0] - random_durations[noise_idx]+1, size=(1,))
            
            # assign noise to the original audio
            augmented_audio[augmentation_position:augmentation_position+random_durations[noise_idx]] += \
                noise_vol * noise_pool[random_start_idxs[noise_idx]: random_start_idxs[noise_idx]+random_durations[noise_idx]]
                
        return augmented_audio

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: MERTConfig, task: HubertPretrainingTask):
        """Build a new model instance."""

        model = MERTModel(cfg, task.cfg, task.dictionaries)
        return model

    def compute_replace_mask(self, padding_mask, mask_indices):
        '''
        all variables are numpy array
        '''
        original_prob = np.random.rand(*mask_indices.shape)<=self.mask_origin
        original_indices = np.all([mask_indices, original_prob], axis=0)
        replace_prob = np.random.rand(*mask_indices.shape)<=self.mask_replace
        replace_indices = np.all([mask_indices, replace_prob], axis=0)
        mask_emb_indices = np.all([mask_indices, ~original_indices, ~replace_indices], axis=0)

        replace_target_indices = np.zeros(mask_indices.shape,dtype=bool)
        all_indices = np.ones(mask_indices.shape,dtype=bool)
        all_indices = np.all([~padding_mask, all_indices], axis=0) # exclude the padding part
        if self.mask_replace_type == 'in_batch':
            # replaces with anyone within the batch, no duplicated
            n_replace = np.sum(replace_indices)
            all_indices = np.where(all_indices) # turn into tuple indices

            replace_target = np.random.choice(len(all_indices[0]), n_replace, replace=False)
            replace_target_indices[(all_indices[0][replace_target], all_indices[1][replace_target])] = True

        elif self.mask_replace_type == 'in_sample':
            # replaces with anyone within the same sample, no duplicated
            for i in range(mask_indices.shape[0]):
                # find replacement for each sample
                n_replace_insample = np.sum(replace_indices[i])
                all_indices_insample = np.where(all_indices[i]) # (T - padding,)
                replace_target_insample = np.random.choice(len(all_indices_insample[0]), n_replace_insample, replace=False)
                replace_target_indices[i][all_indices_insample[0][replace_target_insample]] = True

        return mask_emb_indices, replace_indices, replace_target_indices

    def apply_mask(self, x, padding_mask, target_list):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            if self.mask_replace > 0:
                mask_emb_indices, replace_indices, replace_target_indices = self.compute_replace_mask(padding_mask, mask_indices)

                mask_indices = torch.from_numpy(mask_indices).to(x.device) # tokens involved in mask prediction task
                mask_emb_indices = torch.from_numpy(mask_emb_indices).to(x.device) # tokens replaced with [MASK]
                # origin_indices = torch.from_numpy(origin_indices).to(x.device) # tokens remains the same, no need to do assignment
                replace_indices = torch.from_numpy(replace_indices).to(x.device) # tokens that are replaced with other tokens
                replace_target_indices = torch.from_numpy(replace_target_indices).to(x.device) # tokens that are used to replace

                x[mask_emb_indices] = self.mask_emb
                x[replace_indices] = x[replace_target_indices]

            else:
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
                x[mask_indices] = self.mask_emb
        
                
        else:
            mask_indices = None


        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        if self.chunk_nce_cal > 0:
            logits = []

            for start in range(0, x.shape[0], self.chunk_nce_cal):
                end = start + self.chunk_nce_cal
                a = x[start:end]
                b = targets[start:end]
                # assert a.shape[0] == b.shape[0], f'mismatch shape of a {a.shape} and b {b.shape}, x {x.shape} and targets {targets.shape}'
                logits.append(torch.cosine_similarity(a.float(), b.float(), dim=-1).type_as(a))
            logits = torch.cat(logits,dim=0)
        else:
            logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf") # 忽略正样本对自身的相似性
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1) #相当于变成了 [Batch, Num_Code + 1, Num_Coodebook] 的相似度矩阵，其中第0个是目标
        return logits

    def compute_nce_learned_temp(self, x, pos, negs, logit_temp):
        
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        # logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        if self.chunk_nce_cal > 0:
            logits = []
            for start in range(0, x.shape[0], self.chunk_nce_cal):
                logits.append(torch.cosine_similarity(x[start:start+self.chunk_nce_cal].float(), targets[start:start+self.chunk_nce_cal].float(), dim=-1).type_as(x))
            logits = torch.cat(logits,dim=0)
        else:
            logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logit_scale = torch.clamp(logit_temp.exp(), max=self.learnable_temp_max)
        logits *= logit_scale
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits
        

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        '''
        features: BxCxT
        '''
        if self.wav_normalize:
            assert source.dim() == 2
            with torch.no_grad():
                source = torch.nn.functional.layer_norm(source, source.shape)

        if self.cfg.audio_extract_type == 'w2v_conv':
            if self.feature_grad_mult > 0:
                features = self.feature_extractor(source)
                if self.feature_grad_mult != 1.0:
                    features = GradMultiply.apply(features, self.feature_grad_mult)
            else:
                with torch.no_grad():
                    self.feature_extractor.eval()
                    features = self.feature_extractor(source)
            return features
        elif self.cfg.audio_extract_type == 'melspec':
            with torch.no_grad():
                features = self.feature_extractor_melspec.compute_mel(source).transpose(1, 2)
            return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask
    
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.mask_dynamic_prob_stage >=0:
         if num_updates == self.mask_dynamic_prob_step[self.mask_dynamic_prob_stage]:
            logger.info(f'updating mask_prob from {self.mask_dynamic_prob[self.mask_dynamic_prob_stage]} to {self.mask_dynamic_prob[self.mask_dynamic_prob_stage+1]} at step {num_updates}')
            self.mask_prob = self.mask_dynamic_prob[self.mask_dynamic_prob_stage+1]
            # stop updating since it gets to the last stage
            self.mask_dynamic_prob_stage = self.mask_dynamic_prob_stage + 1 if self.mask_dynamic_prob_stage < len(self.mask_dynamic_prob_step)-1 else -1 

        if self.mask_dynamic_len_stage >=0:
         if num_updates == self.mask_dynamic_len_step[self.mask_dynamic_len_stage]:
            logger.info(f'updating mask_length from {self.mask_dynamic_len[self.mask_dynamic_len_stage]} to {self.mask_dynamic_len[self.mask_dynamic_len_stage+1]} at step {num_updates}')
            self.mask_length = self.mask_dynamic_len[self.mask_dynamic_len_stage+1]
            
            # stop updating since it gets to the last stage
            self.mask_dynamic_len_stage = self.mask_dynamic_len_stage + 1 if self.mask_dynamic_len_stage < len(self.mask_dynamic_len_step)-1 else -1
            
        self.num_updates = num_updates
    
    def encodec_encode(self, inp):
        if self.cfg.audio_codec_type == 'encodec':
            encoded_frames = self.encodec.encode(inp) #list, B,[ 8,T ]
            codes = torch.cat([encoded[0].detach() for encoded in encoded_frames], dim=-1)
            return codes
        elif self.cfg.audio_codec_type == 'dac':
            self.encodec.eval()
            x = self.encodec.preprocess(inp, self.task_cfg.sample_rate)
            z, codes, latents, _, _ = self.encodec.encode(x)
            return codes #[Batch, NumCode=32, Freq*Secs]
        elif self.cfg.audio_codec_type == 'rvq':
            if len(inp.shape) == 3:
                inp = inp.squeeze(1)
            x = self.rvq_preprocess(inp)
            self.encodec.eval()
            quantized_prompt_embeds, codes, _, commitment_loss, codebook_loss, rvq_usage = self.encodec(x)
            return codes
        else:
            raise NotImplementedError
    
    #MERTforward
    def forward(
        self,
        source: torch.Tensor, # B,L
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cqt_labels: Optional[torch.Tensor] = None,
        mel_labels: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        # with autocast(device_type=source.device.type,  dtype=torch.float32):
        # import pdb
        # pdb.set_trace()
        # print('source_shape:',str(source.shape))
        # TODO: extrace target_list 
        #这里修改过
        # print('running')
        if not features_only and self.use_encodec_target:
            with torch.no_grad():
                encodec_inp = self.rsp_src2encodec(source)
                # if(self.num_updates==2):print(encodec_inp);exit()
                encodec_inp = encodec_inp.unsqueeze(1) # B,1,T
                # print(encodec_inp.shape)
                codes = self.encodec_encode(encodec_inp)
                # print(codes.shape)
                # print(target_list.shape)
                target_list = [codes[:,codebook_inx,:].detach() for codebook_inx in range(8)]

        if self.mixture_prob > 0:
            # compute cqt before mixture
            if self.cfg.audio_cqt_loss_m:
                cqt_targets = self.encoder_cqt_model.compute_cqt(source)
            if self.cfg.audio_mel_loss_m:
                mel_targets = self.encoder_mel_model.compute_mel(source)
            with torch.no_grad():
                batch_audios = torch.clone(source)
                for i in range(source.shape[0]):
                    if torch.rand(1).item() > self.mixture_prob:
                        try:
                            #这里加入批内噪声
                            source[i] = self.inbatch_noise_augment(
                                                    target_audio = batch_audios[i], target_audio_idx = i, batch_audios = batch_audios,
                                                    noise_len_min = self.inbatch_noise_augment_len_range[0], noise_len_max = self.inbatch_noise_augment_len_range[1], 
                                                    n_noise_min = self.inbatch_noise_augment_number_range[0], n_noise_max = self.inbatch_noise_augment_number_range[1],
                                                    noise_vol = self.inbatch_noise_augment_volume)
                        except:
                            source[i] = batch_audios[i]                

        features = self.forward_features(source)

        if not features_only and self.use_rq_target:
            target_list = []
            with torch.no_grad():
                features_projected = torch.matmul(features.transpose(1, 2), self.rq_projection.to(features.device)) # input_dim -> embedding_dim * num_codebooks

                B, T, C = features_projected.size()
                features_flatten = features_projected.view(B * T, C)
                _, codes, _ = quantize_vector(self.rq_norm(features_flatten, dim=-1), self.rq_embeddings)
                codes = codes.reshape(B, T, -1)  # [B, T, num_codebooks]
                target_list += [codes[:, :, codebook_inx] for codebook_inx in range(codes.shape[-1])]

                if self.cfg.audio_rq_loss_use_chroma:
                    chroma_features = self.rq_wav2chroma(source)
                    features_projected = torch.matmul(chroma_features.transpose(1, 2), self.rq_projection_chroma.to(chroma_features.device)) # input_dim -> embedding_dim * num_codebooks

                    B, T, C = features_projected.size()
                    features_flatten = features_projected.view(B * T, C)
                    _, codes, _ = quantize_vector(self.rq_norm(features_flatten, dim=-1), self.rq_embeddings_chroma)
                    codes = codes.reshape(B, T, -1)  # [B, T, num_codebooks]
                    target_list += [codes[:, :, codebook_inx] for codebook_inx in range(codes.shape[-1])]


        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)
        
        features_pen = features.float().pow(2).mean()
        
        features = features.transpose(1, 2) # BxTxC

        if self.cfg.feature_extractor_cqt:
            features_cqt = self.feature_extractor_cqt(source).transpose(1, 2)
            features_cqt = features_cqt[:,:features.shape[1],:] # align shape
            # version 1
            # features = features + features_cqt
            # features = self.layer_norm(features)
            # version 2
            # features_cqt = self.post_cqt_feature_proj(features_cqt) # v2
            # features = self.layer_norm(features) + self.layer_norm(features_cqt)
            # version 3
            features = torch.cat([features,features_cqt], 2)
            features = self.layer_norm(features) # BxTxC
        else:
            features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
            if self.post_proj_layer_norm is not None:
                features = self.post_proj_layer_norm(features)


        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if not features_only and mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        # 很美妙的设定，取第layer层的特征，如果超出范围则取最后一层

        if features_only:
            #这里有所修改，增加了一个所有层的结果
            return {"x": x, "padding_mask": padding_mask, "features": features,"layer_results": layer_results}

        def compute_pred(proj_x, target, label_embs, logit_temp=None):
            # skip the codebook that is not selected
            if proj_x is None:
                return None
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long()) # 选择目标对应的嵌入向量
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1) #将嵌入表重复batch次
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            if logit_temp is not None:
                return self.compute_nce_learned_temp(proj_x, y, negs, logit_temp)
            else:
                return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            if self.random_codebook <= 0:
                proj_x_m = self.final_proj(x[masked_indices]) #将特征投射到一个更低维的空间
                if self.untie_final_proj:
                    proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1) #按照不同的码本切成多个向量
                else:
                    proj_x_m_list = [proj_x_m for _ in range(len(target_list))] # no extra RAM taken here
            else:
                # pass
                selected_books = np.random.choice(len(target_list),self.random_codebook)
                proj_x_m_list = []
                for i in range(len(target_list)):
                    if i in selected_books:
                        if self.untie_final_proj:
                            proj_x_m_list.append(self.final_projs[i](x[masked_indices]))
                        else:
                            proj_x_m_list.append(self.final_proj(x[masked_indices]))
                    else:
                        proj_x_m_list.append(None)
            

            if self.learnable_temp:
                logit_m_list = [
                    compute_pred(proj_x_m, t[masked_indices], label_embs_list[i], logit_temp)
                    for i, (proj_x_m, t, logit_temp),  in enumerate(zip(proj_x_m_list, target_list, self.logit_temp_list))
                ]
            else:
                logit_m_list = [
                    compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                    for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
                ]  

            # else:
            #     # # mute to optimize the codes
            #     proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            #     if self.learnable_temp:
            #         logit_m_list = [
            #             compute_pred(proj_x_m, t[masked_indices], label_embs_list[i], logit_temp)
            #             for i, (t, logit_temp),  in enumerate(zip(target_list, self.logit_temp_list))
            #         ]

            #     else:
            #         logit_m_list = [
            #             compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
            #             for i, t in enumerate(target_list)
            #         ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        # if self.emb_grad_mult > 0 and self.emb_grad_mult !=1.0:
        #     self.label_embs_concat = GradMultiply.apply(self.label_embs_concat, self.emb_grad_mult)

            # word_embeddin =  ∗α+word_embedding .detach()∗(1−α).
            # self.label_embs_concat = self.label_embs_concat * self.emb_grad_mult + self.label_embs_concat.detach()*(1-self.emb_grad_mult)

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }

        if self.cfg.audio_cqt_loss_m:
            if cqt_labels is not None:
                cqt_targets = cqt_labels[:masked_indices.shape[0],:masked_indices.shape[1]] # dump the last
            else:
                if self.mixture_prob > 0:
                    # no need to compute again
                    assert cqt_targets is not None
                else:
                    cqt_targets = self.encoder_cqt_model.compute_cqt(source)
                cqt_targets = cqt_targets[:masked_indices.shape[0],:masked_indices.shape[1]] # dump the last

            cqt_pred_m = self.encoder_cqt_model(x[masked_indices])
            # logger.info(x[masked_indices].shape, cqt_pred_m.shape, cqt_targets.shape) 
            cqt_loss_m = self.encoder_cqt_model.criterion(cqt_pred_m, cqt_targets[masked_indices])
            if self.use_rq_target and self.cfg.audio_rq_loss_num_codebooks > 16:
                cqt_loss_m = cqt_loss_m * 10
            result["cqt_pred_m"] = cqt_loss_m

        if self.cfg.audio_mel_loss_m:
            if mel_labels is not None:
                mel_targets = mel_labels[:masked_indices.shape[0],:masked_indices.shape[1]] # dump the last
            else:
                if self.mixture_prob > 0:
                    # no need to compute again
                    assert mel_targets is not None
                else:
                    mel_targets = self.encoder_mel_model.compute_mel(source)
                mel_targets = mel_targets[:masked_indices.shape[0],:masked_indices.shape[1]] # dump the last

            mel_pred_m = self.encoder_mel_model(x[masked_indices])
            # logger.info(x[masked_indices].shape, cqt_pred_m.shape, cqt_targets.shape) 
            mel_loss_m = self.encoder_mel_model.criterion(mel_pred_m, mel_targets[masked_indices])
            result["mel_pred_m"] = mel_loss_m

        if self.learnable_temp:
            # for i in range(len(self.logit_temp_list)):
            for i in range(self.logit_temp_list.shape[0]):
                result[f"logit_temp_{i}"] = self.logit_temp_list[i].item()

        return result
    # def extract_all_feature(
    #     self,
    #     source: torch.Tensor, # B,L
    #     target_list: Optional[List[torch.Tensor]] = None,
    #     padding_mask: Optional[torch.Tensor] = None,
    #     cqt_labels: Optional[torch.Tensor] = None,
    #     mel_labels: Optional[torch.Tensor] = None,
    #     mask: bool = True,
    #     features_only: bool = False,
    #     output_layer: Optional[int] = None,
    # ) -> Dict[str, torch.Tensor]:
    #     """output layer is 1-based"""
    #     # with autocast(device_type=source.device.type,  dtype=torch.float32):
    #     # import pdb
    #     # pdb.set_trace()
    #     # print('source_shape:',str(source.shape))
    #     # TODO: extrace target_list 
    #     #这里修改过
    #     # print('running')
    #     with torch.no_grad():
    #         encodec_inp = self.rsp_src2encodec(source)
    #         encodec_inp = encodec_inp.unsqueeze(1) # B,1,T
    #         # print(encodec_inp.shape)
    #         encoded_frames = self.encodec.encode(encodec_inp) #list, B,[ 8,T ]
    #         codes = torch.cat([encoded[0].detach() for encoded in encoded_frames], dim=-1)
    #         # print(codes.shape)
    #         # print(target_list.shape)
    #         target_list = [codes[:,codebook_inx,:].detach() for codebook_inx in range(8)]

    #     if self.mixture_prob > 0:
    #         # compute cqt before mixture
    #         if self.cfg.audio_cqt_loss_m:
    #             cqt_targets = self.encoder_cqt_model.compute_cqt(source)
    #         if self.cfg.audio_mel_loss_m:
    #             mel_targets = self.encoder_mel_model.compute_mel(source)
    #         with torch.no_grad():
    #             batch_audios = torch.clone(source)
    #             for i in range(source.shape[0]):
    #                 if torch.rand(1).item() > self.mixture_prob:
    #                     try:
    #                         #这里加入批内噪声
    #                         source[i] = self.inbatch_noise_augment(
    #                                                 target_audio = batch_audios[i], target_audio_idx = i, batch_audios = batch_audios,
    #                                                 noise_len_min = self.inbatch_noise_augment_len_range[0], noise_len_max = self.inbatch_noise_augment_len_range[1], 
    #                                                 n_noise_min = self.inbatch_noise_augment_number_range[0], n_noise_max = self.inbatch_noise_augment_number_range[1],
    #                                                 noise_vol = self.inbatch_noise_augment_volume)
    #                     except:
    #                         source[i] = batch_audios[i]                

    #     features = self.forward_features(source)
    #     if target_list is not None:
    #         features, target_list = self.forward_targets(features, target_list)
        
    #     features_pen = features.float().pow(2).mean()
        
    #     features = features.transpose(1, 2) # BxTxC

    #     if self.cfg.feature_extractor_cqt:
    #         features_cqt = self.feature_extractor_cqt(source).transpose(1, 2)
    #         features_cqt = features_cqt[:,:features.shape[1],:] # align shape
    #         # version 1
    #         # features = features + features_cqt
    #         # features = self.layer_norm(features)
    #         # version 2
    #         # features_cqt = self.post_cqt_feature_proj(features_cqt) # v2
    #         # features = self.layer_norm(features) + self.layer_norm(features_cqt)
    #         # version 3
    #         features = torch.cat([features,features_cqt], 2)
    #         features = self.layer_norm(features) # BxTxC
    #     else:
    #         features = self.layer_norm(features)
    #     unmasked_features = features.clone()

    #     if padding_mask is not None:
    #         padding_mask = self.forward_padding_mask(features, padding_mask)

    #     if self.post_extract_proj is not None:
    #         features = self.post_extract_proj(features)
    #         if self.post_proj_layer_norm is not None:
    #             features = self.post_proj_layer_norm(features)

    #     features = self.dropout_input(features)
    #     unmasked_features = self.dropout_features(unmasked_features)

    #     if mask:
    #         x, mask_indices = self.apply_mask(features, padding_mask, target_list)
    #     else:
    #         x = features
    #         mask_indices = None

    #     # feature: (B, T, D), float
    #     # target: (B, T), long
    #     # x: (B, T, D), float
    #     # padding_mask: (B, T), bool
    #     # mask_indices: (B, T), bool
        
    #     all_layer_outputs = []
    #     all_layer_outputs.append(x.unsqueeze(0))
    #     with torch.no_grad():
    #         for layer in self.encoder.layers:
    #             output = layer(x)
    #             all_layer_outputs.append(output[0].unsqueeze(0))
    #     concatenated_output = torch.cat(all_layer_outputs,dim=0)
    #     return concatenated_output


    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def extract_all_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        feature_num = 24
        ):

        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        x = res['x'] # [bsz,T,1024]
        all_features = []
        all_features.append(x.unsqueeze(0))
        # print('x0 shape :',x.shape)
        for i in range(feature_num):
            res = self.forward(
                source,
                padding_mask=padding_mask,
                mask=mask,
                features_only=True,
                output_layer=i,
            )
            
            feature = res["features"]
            # print(feature.shape)
            all_features.append(feature.unsqueeze(0)) # [bsz,T,1024]
        concatenated_output = torch.cat(all_features,dim=0)
        return concatenated_output #[channels,bsz,T,1024]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")
        if "cqt_pred_m" in net_output:
            extra_losses.append(net_output["cqt_pred_m"])
            names.append("cqt_pred_m")
        if "mel_pred_m" in net_output:
            extra_losses.append(net_output["mel_pred_m"])
            names.append("mel_pred_m")
            
        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None

    def initialize_dynamic_mask_prob(self):
        # if len(self.mask_dynamic_prob_step) > 0 and len(self.mask_dynamic_prob) > 0:
        if self.num_updates == 0:
            logger.info(f'setting masking prob...')
        else:
            logger.info(f"loading checkpoint at step {self.num_updates}, resuming the mask_prob is set to trained with dynamic schedule")
        assert len(self.mask_dynamic_prob_step) + 1 == len(self.mask_dynamic_prob), ("the len(step) is the step of updating mask_prob")
        self.mask_dynamic_prob_stage = 0
        for i in self.mask_dynamic_prob_step:
            if self.num_updates >= i:
                self.mask_dynamic_prob_stage += 1
        self.mask_prob = self.mask_dynamic_prob[self.mask_dynamic_prob_stage]
        logger.info(f'set the masking prob as {self.mask_prob}, stage {self.mask_dynamic_prob_stage}')
        if self.num_updates >=  self.mask_dynamic_prob_step[-1]:
            self.mask_dynamic_prob_stage = -1 # no need for further updating

    def initialize_dynamic_mask_len(self):
        if self.num_updates == 0:
            logger.info(f'setting masking prob...')
        else:
            logger.info(f"loading checkpoint at step {self.num_updates}, resuming the mask_length is set to trained with dynamic schedule")
        assert len(self.mask_dynamic_len_step) + 1 == len(self.mask_dynamic_len), ("the len(step) is the step of updating mask_len")
        self.mask_dynamic_len_stage = 0
        for i in self.mask_dynamic_len_step:
            if self.num_updates >= i:
                self.mask_dynamic_len_stage += 1
        self.mask_length = self.mask_dynamic_len[self.mask_dynamic_len_stage]
        logger.info(f'set the masking length as {self.mask_length}, stage {self.mask_dynamic_len_stage}')
        if self.num_updates >=  self.mask_dynamic_len_step[-1]:
            self.mask_dynamic_len_stage = -1 # no need for further updating

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if len(self.mask_dynamic_prob_step) > 0 and len(self.mask_dynamic_prob) > 0:
            self.initialize_dynamic_mask_prob()
    
        if len(self.mask_dynamic_len_step) > 0 and len(self.mask_dynamic_len) > 0:
            self.initialize_dynamic_mask_len()

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class TransformerEncoder_extend(TransformerEncoder):
    def build_encoder_layer(self, args: MERTConfig):
        
        if args.layer_type == "transformer":

            if (args.deepnorm or args.subln or args.attention_relax > 0.0 ):
                residual_alpha = 1.0
                if args.deepnorm:
                    residual_alpha = math.pow(2.0 * args.encoder_layers, 0.25)

                layer = TransformerSentenceEncoderLayerExtend(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    residual_alpha=residual_alpha,
                    attention_relax=args.attention_relax,
                )
            else:
                layer = TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )


        elif args.layer_type == "conformer":
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,
                ffn_embed_dim=args.encoder_ffn_embed_dim,
                attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                activation_fn="swish",
                attn_type=args.attn_type,
                use_fp16=args.fp16,
                pos_enc_type="abs",
            )
        from fairseq.distributed import fsdp_wrap
        from fairseq.modules.checkpoint_activations import checkpoint_wrapper

        layer = fsdp_wrap(layer)
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args: MERTConfig):
        super().__init__(args)
        if args.deepnorm:
            # if is_encoder_decoder:
            #     init_scale = (
            #         math.pow(
            #             math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
            #         )
            #         / 1.15
            #     )
            # else:
            init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                    "fc1" in name
                    or "fc2" in name
                    or "out_proj" in name
                    or "v_proj" in name
                ):
                    p.data.div_(init_scale)

class TransformerSentenceEncoderLayerExtend(TransformerSentenceEncoderLayer):
    """
    Extend the Transformer Encoder Layer to support DeepNorm.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        residual_alpha: float = 1.0,
        subln: bool = False,
        attention_relax: float = -1.0,
    ) -> None:

        super().__init__()
        # nn.Module().__init__(self)

        self.residual_alpha = residual_alpha
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)


        if attention_relax > 0:
            # self.attention_relax = attention_relax
            logger.info(f"creating custom attention layer with relaxation scale: {attention_relax}")
            self.self_attn = MultiheadAttention_extend(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
                attention_relax=attention_relax,
            )

        else:
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=True,
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        
        self.ffn_layernorm = LayerNorm(ffn_embedding_dim) if subln else None
        
        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)
    
    def residual_connection(self, x, residual):
        return residual * self.residual_alpha + x

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)

            # for subln
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)

            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )

            x = self.dropout1(x)
            # x = residual + x
            x = self.residual_connection(x, residual)

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)

            layer_result = x

            x = self.dropout3(x)
            # x = residual + x
            x = self.residual_connection(x, residual)
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)
    

class MultiheadAttention_extend(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        # dictionary=None,
        q_noise=0.0,
        qn_block_size=8,
        attention_relax = -1.0,
        # TODO: pass in config rather than string.
        # config defined in xformers.components.attention.AttentionConfig
        xformers_att_config: Optional[str] = None,
        xformers_blocksparse_layout: Optional[
            torch.Tensor
        ] = None,  # This should be part of the config
        xformers_blocksparse_blocksize: Optional[
            int
        ] = 16,  # This should be part of the config
    ):
        # nn.Module.__init__(self)
        # super().__init__()
        # initialize the instance with the father class method
        # MultiheadAttention.__init__(self,
        # super(MultiheadAttention_extend, self).__init__(        
        # super(self).__init__(  
        super().__init__(   
            embed_dim,
            num_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self_attention,
            encoder_decoder_attention=encoder_decoder_attention,
            # dictionary=dictionary,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            xformers_att_config=xformers_att_config,
            xformers_blocksparse_layout=xformers_blocksparse_layout, 
            xformers_blocksparse_blocksize=xformers_blocksparse_blocksize,
        )

        self.attention_relax = attention_relax

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        if not self.skip_embed_dim_check:
            assert (
                embed_dim == self.embed_dim
            ), f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert src_len, key_bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
            # The Multihead attention implemented in pytorch forces strong dimension check
            # for input embedding dimention and K,Q,V projection dimension.
            # Since pruning will break the dimension check and it is not easy to modify the pytorch API,
            # it is preferred to bypass the pytorch MHA when we need to skip embed_dim_check
            and not self.skip_embed_dim_check
        ):
            assert key is not None and value is not None

            if self.use_xformers:
                return self._xformers_attn_forward(
                    query, key, value, key_padding_mask, need_weights, attn_mask
                )

            else:
                return F.multi_head_attention_forward(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    torch.empty([0]),
                    torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                    self.bias_k,
                    self.bias_v,
                    self.add_zero_attn,
                    self.dropout_module.p,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    self.training or self.dropout_module.apply_during_inference,
                    key_padding_mask.bool() if key_padding_mask is not None else None,
                    need_weights,
                    attn_mask,
                    use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj.weight,
                    k_proj_weight=self.k_proj.weight,
                    v_proj_weight=self.v_proj.weight,
                )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k, v, attn_mask, key_padding_mask = self._add_bias(
                k, v, attn_mask, key_padding_mask, bsz
            )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        kv_bsz = bsz  # need default value for scripting
        if k is not None:
            kv_bsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kv_bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(
                kv_bsz, self.num_heads, -1, self.head_dim
            )
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k, v, key_padding_mask, attn_mask = self._append_zero_attn(
                k=k, v=v, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kv_bsz, -1, self.num_heads) + q.size()[1:]),
                k.view((kv_bsz, self.num_heads) + k.size()[1:]),
            )
            attn_weights = attn_weights.reshape((-1,) + attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.view(
                    kv_bsz, -1, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v
        if self.attention_relax > 0 :
            # tgt_len == src_len

            # => (bsz, self.num_heads, tgt_len, src_len)
            # attn_weights_relax = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)/self.attention_relax
            
            # => (bsz * self.num_heads, tgt_len, src_len)
            attn_weights_relax = attn_weights / self.attention_relax

            # # => (bsz, self.num_heads, 1, src_len)
            # attn_max_relax = torch.max(attn_weights_relax, dim=-2, keepdim=False).unsqueeze(2)
            

            # find max according to K_j' => (bsz* self.num_heads, tgt_len, 1)
            attn_max_relax = torch.max(attn_weights_relax, dim=-1, keepdim=False).unsqueeze(2)

            # => (bsz * self.num_heads, tgt_len, src_len)
            attn_weights = (attn_weights_relax - attn_max_relax) * self.attention_relax
            # attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn: Optional[torch.Tensor] = None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(
                    (
                        kv_bsz,
                        -1,
                        self.num_heads,
                    )
                    + attn_probs.size()[1:]
                ),
                v.view(
                    (
                        kv_bsz,
                        self.num_heads,
                    )
                    + v.size()[1:]
                ),
            )
            attn = attn.reshape((-1,) + attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights
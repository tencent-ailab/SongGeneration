"""
Tokenizer or wrapper around existing models.
Also defines the main interface that a model must follow to be usable as an audio tokenizer.
"""

from abc import ABC, abstractmethod
import logging
import typing as tp
import torch
from torch import nn


logger = logging.getLogger()


class AudioTokenizer(ABC, nn.Module):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) :
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

    @staticmethod
    def get_pretrained(
            name: str, 
            vae_config: str,
            vae_model: str,
            device: tp.Union[torch.device, str] = 'cpu', 
            mode='extract',
            tango_device:str='cuda'
            ) -> 'AudioTokenizer':
        """Instantiate a AudioTokenizer model from a given pretrained model.

        Args:
            name (Path or str): name of the pretrained model. See after.
            device (torch.device or str): Device on which the model is loaded.
        """

        model: AudioTokenizer
        if name.split('_')[0] == 'Flow1dVAESeparate':
            model_type = name.split('_', 1)[1]
            logger.info("Getting pretrained compression model from semantic model %s", model_type)
            model = Flow1dVAESeparate(model_type, vae_config, vae_model, tango_device=tango_device)
        elif name.split('_')[0] == 'Flow1dVAE1rvq':
            model_type = name.split('_', 1)[1]
            logger.info("Getting pretrained compression model from semantic model %s", model_type)
            model = Flow1dVAE1rvq(model_type, vae_config, vae_model, tango_device=tango_device)
        else:
            raise NotImplementedError("{} is not implemented in models/audio_tokenizer.py".format(
                name))
        return model.to(device).eval()
    

class Flow1dVAE1rvq(AudioTokenizer):
    def __init__(
        self, 
        model_type: str = "model_2_fixed.safetensors",
        vae_config: str = "",
        vae_model: str = "",
        tango_device: str = "cuda"
        ):
        super().__init__()

        from codeclm.tokenizer.Flow1dVAE.generate_1rvq import Tango
        model_path = model_type
        self.model = Tango(model_path=model_path, vae_config=vae_config, vae_model=vae_model, device=tango_device)
        print ("Successfully loaded checkpoint from:", model_path)

            
        self.n_quantizers = 1

    def forward(self, x: torch.Tensor) :
        # We don't support training with this.
        raise NotImplementedError("Forward and training with DAC not supported.")
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        codes = self.model.sound2code(x) # [B T] -> [B N T]
        return codes, None

    
    @torch.no_grad()    
    def decode(self, codes: torch.Tensor, prompt = None, scale: tp.Optional[torch.Tensor] = None, ncodes=9):
        wav = self.model.code2sound(codes, prompt=prompt, guidance_scale=1.5, 
                                    num_steps=50, disable_progress=False) # [B,N,T] -> [B,T]
        return wav[None]

    
    @torch.no_grad()
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        # import pdb; pdb.set_trace()
        return self.model.quantizer.from_codes(codes.transpose(1,2))[0]

    @property
    def channels(self) -> int:
        return 2

    @property
    def frame_rate(self) -> float:
        return 25

    @property
    def sample_rate(self) -> int:
        return self.samplerate

    @property
    def cardinality(self) -> int:
        return 10000

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers

    @property
    def total_codebooks(self) -> int:
        # return self.model.RVQ
        return 1

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        assert n >= 1
        assert n <= self.total_codebooks
        self.n_quantizers = n

    def to(self, device=None, dtype=None, non_blocking=False):
        self = super(Flow1dVAE1rvq, self).to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self
    
    def cuda(self, device=None):
        if device is None:
            device = 'cuda:0'
        return super(Flow1dVAE1rvq, self).cuda(device)

class Flow1dVAESeparate(AudioTokenizer):
    def __init__(
        self, 
        model_type: str = "model_2.safetensors",
        vae_config: str = "",
        vae_model: str = "",
        tango_device: str = "cuda"
        ):
        super().__init__()

        from codeclm.tokenizer.Flow1dVAE.generate_septoken import Tango
        model_path = model_type
        self.model = Tango(model_path=model_path, vae_config=vae_config, vae_model=vae_model, device=tango_device)
        print ("Successfully loaded checkpoint from:", model_path)

            
        self.n_quantizers = 1

    def forward(self, x: torch.Tensor) :
        # We don't support training with this.
        raise NotImplementedError("Forward and training with DAC not supported.")
    
    @torch.no_grad()
    def encode(self, x_vocal: torch.Tensor, x_bgm: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        if x_vocal.ndim == 2:
            x_vocal = x_vocal.unsqueeze(1)
        if x_bgm.ndim == 2:
            x_bgm = x_bgm.unsqueeze(1)
        codes_vocal, codes_bgm = self.model.sound2code(x_vocal, x_bgm)
        return codes_vocal, codes_bgm
    
    @torch.no_grad()    
    def decode(self, codes: torch.Tensor, prompt_vocal = None, prompt_bgm = None, chunked=False, chunk_size=128):
        wav = self.model.code2sound(codes, prompt_vocal=prompt_vocal, prompt_bgm=prompt_bgm, guidance_scale=1.5, 
                                    num_steps=50, disable_progress=False, chunked=chunked, chunk_size=chunk_size) # [B,N,T] -> [B,T]
        return wav[None]

    
    @torch.no_grad()
    def decode_latent(self, codes: torch.Tensor):
        """Decode from the discrete codes to continuous latent space."""
        # import pdb; pdb.set_trace()
        return self.model.quantizer.from_codes(codes.transpose(1,2))[0]

    @property
    def channels(self) -> int:
        return 2

    @property
    def frame_rate(self) -> float:
        return 25

    @property
    def sample_rate(self) -> int:
        return self.samplerate

    @property
    def cardinality(self) -> int:
        return 10000

    @property
    def num_codebooks(self) -> int:
        return self.n_quantizers

    @property
    def total_codebooks(self) -> int:
        # return self.model.RVQ
        return 1

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        assert n >= 1
        assert n <= self.total_codebooks
        self.n_quantizers = n

    def to(self, device=None, dtype=None, non_blocking=False):
        self = super(Flow1dVAESeparate, self).to(device, dtype, non_blocking)
        self.model = self.model.to(device, dtype, non_blocking)
        return self

    def cuda(self, device=None):
        if device is None:
            device = 'cuda:0'
        self = super(Flow1dVAESeparate, self).cuda(device)
        return self

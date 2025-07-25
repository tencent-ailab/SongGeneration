"""
Main model for using CodecLM. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from codeclm.tokenizer.audio_tokenizer import AudioTokenizer
from .lm_levo import LmModel
from ..modules.conditioners import ConditioningAttributes, AudioCondition
from ..utils.autocast import TorchAutocast
import torch
from torch.nn import functional as F
import torchaudio
# from optim.ema import EMA


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

class CodecLM:
    """CodecLM main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, audiotokenizer: AudioTokenizer, lm: LmModel,
                 max_duration: tp.Optional[float] = None, seperate_tokenizer: AudioTokenizer = None):
        self.name = name
        self.audiotokenizer = audiotokenizer
        if self.audiotokenizer:
            self.frame_rate = self.audiotokenizer.frame_rate
        else:
            self.frame_rate = 25
        self.lm = lm
        self.seperate_tokenizer = seperate_tokenizer
        # import pdb; pdb.set_trace()
        if max_duration is None:
            if hasattr(lm, 'cfg'):
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly CodecLM")
        assert max_duration is not None

        self.max_duration: float = max_duration
        self.device = torch.device("cuda")
        self.generation_params: dict = {}
        # self.set_generation_params(duration=15)  # 15 seconds by default
        self.set_generation_params(duration=15, extend_stride=self.max_duration // 2)
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(enabled=False)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                             extend_stride: float = 18, record_tokens: bool = False,
                             record_window: int = 50):
        """Set the generation parameters for CodecLM.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride <= self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'record_tokens': record_tokens,
            'record_window': record_window,
        }

    def set_custom_progress_callback(self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None):
        """Override the default progress callback."""
        self._progress_callback = progress_callback

    # Inference
    def generate(self, lyrics: tp.List[str], 
                 descriptions: tp.List[str],
                 melody_wavs: torch.Tensor = None,
                 melody_is_wav: bool = True,
                 vocal_wavs: torch.Tensor = None,
                 bgm_wavs: torch.Tensor = None,
                 return_tokens: bool = False,
                 ) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if melody_wavs is not None:
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        if vocal_wavs is not None:
            if vocal_wavs.dim() == 2:
                vocal_wavs = vocal_wavs[None]
            if vocal_wavs.dim() != 3:
                raise ValueError("Vocal wavs should have a shape [B, C, T].")
            vocal_wavs = list(vocal_wavs)
        if bgm_wavs is not None:
            if bgm_wavs.dim() == 2:
                bgm_wavs = bgm_wavs[None]
            if bgm_wavs.dim() != 3:
                raise ValueError("BGM wavs should have a shape [B, C, T].")
            bgm_wavs = list(bgm_wavs)
        
        texts, audio_qt_embs = self._prepare_tokens_and_attributes(lyrics=lyrics, melody_wavs=melody_wavs, vocal_wavs=vocal_wavs, bgm_wavs=bgm_wavs, melody_is_wav=melody_is_wav)
        tokens = self._generate_tokens(texts, descriptions, audio_qt_embs)

        if (tokens == self.lm.eos_token_id).any():
            length = torch.nonzero(torch.eq(tokens, self.lm.eos_token_id))[:,-1].min()
            tokens = tokens[...,:length] 

        if return_tokens:
            return tokens
        else:
            out = self.generate_audio(tokens)
            return out


    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            lyrics: tp.Sequence[tp.Optional[str]],
            melody_wavs: tp.Optional[MelodyList] = None,
            vocal_wavs: tp.Optional[MelodyList] = None,
            bgm_wavs: tp.Optional[MelodyList] = None,
            melody_is_wav = True
    ) -> tp.Tuple[tp.List[str], tp.List[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        assert len(lyrics) == 1
        texts = [lyric for lyric in lyrics]
        audio_qt_embs = []
        target_melody_token_len = self.lm.cfg.prompt_len * self.frame_rate
        # import pdb; pdb.set_trace()
        if melody_wavs is None:
            melody_tokens = torch.full((1,1,target_melody_token_len), 16385, device=self.device).long()
        elif melody_wavs is not None:
            if 'prompt_audio' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(texts), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(texts)}"
            if type(melody_wavs) == list:
                melody_wavs = torch.stack(melody_wavs, dim=0)
            melody_wavs = melody_wavs.to(self.device)
            if melody_is_wav:
                melody_tokens, scale = self.audiotokenizer.encode(melody_wavs)
            else:
                melody_tokens = melody_wavs
            if melody_tokens.shape[-1] > target_melody_token_len:
                melody_tokens = melody_tokens[...,:target_melody_token_len]
            elif melody_tokens.shape[-1] < target_melody_token_len:
                melody_tokens = torch.cat([melody_tokens, torch.full((1,1,target_melody_token_len - melody_tokens.shape[-1]), 16385, device=self.device).long()], dim=-1)

        if bgm_wavs is None:
            assert vocal_wavs is None, "vocal_wavs is not None when bgm_wavs is None"
            bgm_tokens = torch.full((1,1,target_melody_token_len), 16385, device=self.device).long()
            vocal_tokens = torch.full((1,1,target_melody_token_len), 16385, device=self.device).long()
        else:
            assert vocal_wavs is not None, "vocal_wavs is None when bgm_wavs is not None"
            if type(vocal_wavs) == list:
                vocal_wavs = torch.stack(vocal_wavs, dim=0)
            if type(bgm_wavs) == list:
                bgm_wavs = torch.stack(bgm_wavs, dim=0)
            vocal_wavs = vocal_wavs.to(self.device)
            bgm_wavs = bgm_wavs.to(self.device)
            if melody_is_wav:
                vocal_tokens, bgm_tokens = self.seperate_tokenizer.encode(vocal_wavs, bgm_wavs)
            else:
                vocal_tokens = vocal_wavs
                bgm_tokens = bgm_wavs
            assert len(vocal_tokens.shape) == len(bgm_tokens.shape) == 3, \
                f"vocal and bgm tokens should have a shape [B, C, T]! " \
                f"got vocal len={vocal_tokens.shape}, and bgm len={bgm_tokens.shape}"
            assert vocal_tokens.shape[-1] == bgm_tokens.shape[-1], \
                f"vocal and bgm tokens should have the same length! " \
                f"got vocal len={vocal_tokens.shape[-1]}, and bgm len={bgm_tokens.shape[-1]}"
            if bgm_tokens.shape[-1] > target_melody_token_len:
                bgm_tokens = bgm_tokens[...,:target_melody_token_len]
            elif bgm_tokens.shape[-1] < target_melody_token_len:
                bgm_tokens = torch.cat([bgm_tokens, torch.full((1,1,target_melody_token_len - bgm_tokens.shape[-1]), 16385, device=self.device).long()], dim=-1)
            if vocal_tokens.shape[-1] > target_melody_token_len:
                vocal_tokens = vocal_tokens[...,:target_melody_token_len]
            elif vocal_tokens.shape[-1] < target_melody_token_len:
                vocal_tokens = torch.cat([vocal_tokens, torch.full((1,1,target_melody_token_len - vocal_tokens.shape[-1]), 16385, device=self.device).long()], dim=-1)
        melody_tokens = torch.cat([melody_tokens, vocal_tokens, bgm_tokens], dim=1)
        assert melody_tokens.shape[-1] == target_melody_token_len
        audio_qt_embs = melody_tokens.long()
        return texts, audio_qt_embs



    def _generate_tokens(self, 
                        texts: tp.Optional[tp.List[str]] = None,
                        descriptions: tp.Optional[tp.List[str]] = None,
                        audio_qt_embs: tp.Optional[tp.List[torch.Tensor]] = None) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(texts=texts, 
                                              descriptions=descriptions, 
                                              audio_qt_embs=audio_qt_embs, 
                                              max_gen_len=total_gen_len, 
                                              **self.generation_params)
        else:
            raise NotImplementedError(f"duration {self.duration} < max duration {self.max_duration}")
        return gen_tokens

    @torch.no_grad()
    def generate_audio(self, gen_tokens: torch.Tensor, prompt=None, vocal_prompt=None, bgm_prompt=None, chunked=False, chunk_size=128, gen_type='mixed'):
        """Generate Audio from tokens"""
        assert gen_tokens.dim() == 3
        if self.seperate_tokenizer is not None:
            gen_tokens_song = gen_tokens[:, [0], :]
            gen_tokens_vocal = gen_tokens[:, [1], :]
            gen_tokens_bgm = gen_tokens[:, [2], :]
            if gen_type == 'bgm':
                gen_tokens_vocal = torch.full_like(gen_tokens_vocal, 3142)
                if vocal_prompt is not None:
                    vocal_prompt = torch.zeros_like(vocal_prompt)
            elif gen_type == 'vocal':
                gen_tokens_bgm = torch.full_like(gen_tokens_bgm, 9670)
                if bgm_prompt is not None:
                    bgm_prompt = torch.zeros_like(bgm_prompt)
            else:
                assert gen_type == 'mixed', f"gen_type {gen_type} not supported"
            gen_audio_seperate = self.seperate_tokenizer.decode([gen_tokens_vocal, gen_tokens_bgm], vocal_prompt, bgm_prompt, chunked=chunked, chunk_size=chunk_size)
            return gen_audio_seperate
        else:
            gen_audio = self.audiotokenizer.decode(gen_tokens, prompt)
            return gen_audio

"""
Main model for using CodecLM. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from codeclm.tokenizer.audio_tokenizer import AudioTokenizer
# from .lm_llama import LMModel
from ..utils.autocast import TorchAutocast
import torch
from torch.nn import functional as F
import torchaudio
# from optim.ema import EMA
from codeclm.utils.utils import dict_from_config
from codeclm.modules.pattern import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
)
from codeclm.modules.conditioners import (
    ConditioningAttributes,
    AudioCondition,
    BaseConditioner,
    QuantizedEmbeddingConditioner,
    ConditionerProvider,
    ConditionFuser,
    QwTextConditioner,
    QwTokenizerConditioner,
    ClassifierFreeGuidanceDropoutInference,
)
import omegaconf

def get_conditioner_provider(output_dim: int, cfg: omegaconf.DictConfig, version: str = 'v1.0') -> ConditionerProvider:
    """Instantiate a conditioning model."""    
    cfg = getattr(cfg, 'conditioners')
    dict_cfg = {} if cfg is None else dict_from_config(cfg)
    conditioners: tp.Dict[str, BaseConditioner] = {}
    condition_provider_args = dict_cfg.pop('args', {})

    for cond, cond_cfg in dict_cfg.items():
        model_type = cond_cfg['model']
        model_args = cond_cfg[model_type]
        if model_type == 'QwTokenizer':
            conditioners[str(cond)] = QwTokenizerConditioner(
                output_dim=output_dim,
                **model_args
            )
        elif model_type == "QwTextTokenizer":
            conditioners[str(cond)] = QwTextConditioner(
                output_dim=output_dim,
                version=version,
                **model_args
            )
        elif model_type == "qt_embedding":
            conditioners[str(cond)] = QuantizedEmbeddingConditioner(
                dim=output_dim,
                **model_args
            )
        else:
            raise ValueError(f"Unrecognized conditioning model: {model_type}")
    conditioner = ConditionerProvider(conditioners, **condition_provider_args)
    return conditioner

def get_codebooks_pattern_provider(code_depth: int, cfg: omegaconf.DictConfig) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        'delay': DelayedPatternProvider,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name)) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(code_depth, **kwargs)

MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]

def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = getattr(cfg, 'fuser')
    fuser_methods = ['sum', 'prepend']
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser

class CodecLM_gen:
    """CodecLM main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, cfg, name: str, audiotokenizer: AudioTokenizer, 
                 max_duration: tp.Optional[float] = None):
        self.cfg = cfg
        self.name = name
        self.audiotokenizer = audiotokenizer
        self.seperate_tokenizer = None
        if max_duration is None:
            max_duration = self.cfg.max_dur
        assert max_duration is not None

        self.max_duration: float = max_duration
        # self.device = next(iter(lm.parameters())).device
        # self.device = next(iter(audiotokenizer.parameters())).device
        self.generation_params: dict = {}
        # self.set_generation_params(duration=15)  # 15 seconds by default
        self.set_generation_params(duration=15, extend_stride=self.max_duration // 2)
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        self.autocast = TorchAutocast(enabled=False)        
        self.condition_provider = get_conditioner_provider(cfg.lm.dim, self.cfg)
        codebooks_pattern_cfg = getattr(cfg, 'codebooks_pattern')
        self.pattern_provider = get_codebooks_pattern_provider(cfg.lm.code_depth, codebooks_pattern_cfg)
        self.fuser = get_condition_fuser(cfg)
        self.eos_token_id = cfg.lm.code_size



    @property
    def frame_rate(self) -> float:
        """Roughly the number of AR steps per seconds."""
        return self.audiotokenizer.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.audiotokenizer.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.audiotokenizer.channels

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
    def generate_condition(self, descriptions: tp.List[str],
                            melody_wavs: torch.Tensor = None, 
                            return_tokens: bool = False,
                            melody_is_wav: bool = True,
                            type_info: tp.List[str] = None,
                            embeded_eosp1: torch.Tensor = None,
                            ) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        if melody_wavs is not None:
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        
            # if melody_is_wav:
            #     melody_wavs = [wav.mean(dim=-2) for wav in melody_wavs]
            
        texts, audio_qt_embs = self._prepare_tokens_and_attributes(descriptions=descriptions,
                                                                        melody_wavs=melody_wavs,
                                                                        melody_is_wav=melody_is_wav)
        fused_input = self.get_condition_tensors(texts, audio_qt_embs, type_info, embeded_eosp1)

        return fused_input, audio_qt_embs


    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            melody_wavs: tp.Optional[MelodyList] = None,
            melody_is_wav = True
    ) -> tp.Tuple[tp.List[str], tp.List[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        texts = [description for description in descriptions]
        audio_qt_embs = []

        if melody_wavs is None:
            audio_qt_embs = None
        elif melody_wavs is not None:
            if 'prompt_audio' not in self.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(texts), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(texts)}"
            if type(melody_wavs) == list:
                melody_wavs = torch.stack(melody_wavs, dim=0)
            # melody_wavs = melody_wavs.to(self.device)
            if melody_is_wav:
                melody_tokens, scale = self.audiotokenizer.encode(melody_wavs)
            else:
                melody_tokens = melody_wavs
            target_melody_token_len = self.cfg.prompt_len * self.audiotokenizer.frame_rate
            if melody_tokens.shape[-1] > target_melody_token_len:
                melody_tokens = melody_tokens[...,:target_melody_token_len]
            for melody in melody_tokens:
                audio_qt_embs.append(melody.long())
        return texts, audio_qt_embs

    @torch.no_grad()
    def prepare_condition_tensors(self,
                                   batch_size = 1,
                                   text: tp.Optional[tp.List[str]] = None, 
                                   audio_qt_emb: tp.Optional[tp.List[torch.Tensor]] = None,
                                   type_info: tp.Optional[tp.List[str]] = None,
                                   prepare_null_condition = False,
                                   ):
        conditions = []
        for i in range(batch_size):
            attr = ConditioningAttributes()
            if 'description' in self.condition_provider.conditioners:
                attr["text"]["description"] = ""
                if text is not None:
                    attr["text"]["description"] = text[i]
            if 'prompt_audio' in self.condition_provider.conditioners:
                if audio_qt_emb is None:    # tokenize stage will padding to max length
                    attr["audio"]['prompt_audio'] = AudioCondition(
                        wav=torch.zeros((1, self.cfg.audio_tokenizer_code_depth, 0)).long().cuda() + 16385, 
                        length=torch.Tensor([0]).long(),
                        sample_rate=[self.cfg.sample_rate],)
                else:
                    aT = audio_qt_emb[i].shape[-1]
                    pattern = self.pattern_provider.get_pattern(aT)
                    audio_qt_seq, _, _ = pattern.build_pattern_sequence(audio_qt_emb[i][None], 
                                                                        self.eos_token_id, keep_only_valid_steps=False)   
                    attr["audio"]['prompt_audio'] = AudioCondition(
                        wav=audio_qt_seq.long().cuda(), 
                        length=torch.Tensor([audio_qt_seq.shape[-1]]).long(),
                        sample_rate=[self.cfg.sample_rate],)
            if 'type_info' in self.condition_provider.conditioners:
                attr["text"]["type_info"] = ""
                if type_info is not None:
                    attr["text"]["type_info"] = type_info[i]
            conditions.append(attr)
        if prepare_null_condition:
            cfg_inference = ClassifierFreeGuidanceDropoutInference() 
            null_conditions = cfg_inference(conditions, condition_types=["audio", "text"], 
                                            customized=None)
            conditions = conditions + null_conditions
        print("conditions", conditions)
        tokenized_conditions = self.condition_provider.tokenize(conditions)
        # import pdb; pdb.set_trace()
        condition_tensors = self.condition_provider(tokenized_conditions)
        return condition_tensors

    def get_condition_tensors(self, texts, audio_qt_embs, type_info, embeded_eosp1):
        condition_tensors = self.prepare_condition_tensors(batch_size=1, text=texts, audio_qt_emb=audio_qt_embs, type_info=type_info, prepare_null_condition=self.cfg.vllm.cfg)
        if self.cfg.vllm.cfg:
            input_ = torch.cat((embeded_eosp1, embeded_eosp1), dim=0)
        else:
            input_ = embeded_eosp1
        fused_input = self.fuser(input_, condition_tensors)
        return fused_input

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
            gen_audio = self.audiotokenizer.decode(gen_tokens, prompt, chunked=chunked, chunk_size=chunk_size)
            return gen_audio
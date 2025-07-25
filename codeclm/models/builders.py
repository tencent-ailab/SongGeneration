"""
All the functions to build the relevant models and modules
from the Hydra config.
"""

import typing as tp

import omegaconf
import torch
from codeclm.utils.utils import dict_from_config
from codeclm.modules.pattern import (
    CodebooksPatternProvider,
    DelayedPatternProvider,
)
from codeclm.modules.conditioners import (
    BaseConditioner,
    QwTokenizerConditioner,
    QwTextConditioner,
    QuantizedEmbeddingConditioner,
    ConditionerProvider,
    ConditionFuser,
)


def get_audio_tokenizer_model(checkpoint_path: str, cfg: omegaconf.DictConfig):
    from codeclm.tokenizer.audio_tokenizer import AudioTokenizer
    """Instantiate a compression model."""
    if checkpoint_path is None:
        return None
    if checkpoint_path.startswith('//pretrained/'):
        name = checkpoint_path.split('/', 3)[-1]
        return AudioTokenizer.get_pretrained(name, cfg.vae_config, cfg.vae_model, 'cuda', mode=cfg.mode)
    elif checkpoint_path == "":
        return None
    else:
        name = checkpoint_path
        return AudioTokenizer.get_pretrained(name, cfg.vae_config, cfg.vae_model, 'cuda', mode=cfg.mode)
    

def get_audio_tokenizer_model_cpu(checkpoint_path: str, cfg: omegaconf.DictConfig):
    from codeclm.tokenizer.audio_tokenizer import AudioTokenizer
    """Instantiate a compression model."""
    if checkpoint_path is None:
        return None
    if checkpoint_path.startswith('//pretrained/'):
        name = checkpoint_path.split('/', 3)[-1]
        return AudioTokenizer.get_pretrained(name, cfg.vae_config, cfg.vae_model, 'cpu', mode=cfg.mode, tango_device='cpu')
    elif checkpoint_path == "":
        return None
    else:
        name = checkpoint_path
        return AudioTokenizer.get_pretrained(name, cfg.vae_config, cfg.vae_model, 'cpu', mode=cfg.mode, tango_device='cpu')


def get_lm_model(cfg: omegaconf.DictConfig): #-> LMModel:
    """Instantiate a LM."""    
    lm_kwargs = dict_from_config(getattr(cfg, 'lm'))
    
    # n_q: number of RVQ
    code_depth = lm_kwargs['code_depth']
    q_modeling = lm_kwargs.pop('q_modeling', None)    
        
    # conditioner
    condition_provider = get_conditioner_provider(lm_kwargs["dim"], cfg)
    
    # codebook pattern: delay
    codebooks_pattern_cfg = getattr(cfg, 'codebooks_pattern')
    if codebooks_pattern_cfg.modeling is None:
        assert q_modeling is not None, \
            "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
        codebooks_pattern_cfg = omegaconf.OmegaConf.create(
            {'modeling': q_modeling, 'delay': {'delays': list(range(code_depth))}}
        )
    pattern_provider = get_codebooks_pattern_provider(code_depth, codebooks_pattern_cfg)
    
    # condition dropout
    attribute_dropout = dict_from_config(getattr(cfg, 'attribute_dropout'))
    cls_free_guidance = dict_from_config(getattr(cfg, 'classifier_free_guidance'))
    cfg_prob, cfg_coef = cls_free_guidance['training_dropout'], cls_free_guidance['inference_coef']
    
    # condition fuser
    fuser = get_condition_fuser(cfg)    
    lm_type = lm_kwargs['lm_type'] # YCY: For consistency, choose different lm.py based on lm_type
    if lm_type == 'Llama':
        from .lm_levo import LmModel
        return LmModel(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            cfg=cfg,
            **lm_kwargs
        ).to('cpu')
    else:
        raise KeyError(f"Unexpected LM model {lm_type}")


def get_conditioner_provider(output_dim: int, cfg: omegaconf.DictConfig) -> ConditionerProvider:
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


def get_condition_fuser(cfg: omegaconf.DictConfig) -> ConditionFuser:
    """Instantiate a condition fuser object."""
    fuser_cfg = getattr(cfg, 'fuser')
    fuser_methods = ['sum', 'prepend']
    fuse2cond = {k: fuser_cfg[k] for k in fuser_methods}
    kwargs = {k: v for k, v in fuser_cfg.items() if k not in fuser_methods}
    fuser = ConditionFuser(fuse2cond=fuse2cond, **kwargs)
    return fuser


def get_codebooks_pattern_provider(code_depth: int, cfg: omegaconf.DictConfig) -> CodebooksPatternProvider:
    """Instantiate a codebooks pattern provider object."""
    pattern_providers = {
        'delay': DelayedPatternProvider,
    }
    name = cfg.modeling
    kwargs = dict_from_config(cfg.get(name)) if hasattr(cfg, name) else {}
    klass = pattern_providers[name]
    return klass(code_depth, **kwargs)

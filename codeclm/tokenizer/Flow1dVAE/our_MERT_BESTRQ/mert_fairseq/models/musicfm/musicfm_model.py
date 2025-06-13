try:
    from .model.musicfm_25hz import MusicFM25Hz
except:
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model.musicfm_25hz import MusicFM25Hz
try:
    from fairseq.fairseq.dataclass import FairseqDataclass
    from fairseq.fairseq.models import BaseFairseqModel, register_model
    from fairseq.fairseq.tasks.fairseq_task import FairseqTask
except:
    from fairseq.dataclass import FairseqDataclass
    from fairseq.models import BaseFairseqModel, register_model
    from fairseq.tasks.fairseq_task import FairseqTask
    
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

from logging import getLogger

logger = getLogger(__name__)

@dataclass
class MusicFMConfig(FairseqDataclass):
    label_rate:int = field(default=25)
    num_codebooks:int = field(default=1)
    codebook_dim:int = field(default=16)
    codebook_size:int = field(default=4096)
    features:List[str] = field(default_factory=lambda:["melspec_2048"])
    hop_length:int = field(default=240)
    n_mels:int = field(default=128)
    conv_dim:int = field(default=512)
    encoder_dim:int = field(default=1024)
    encoder_depth:int = field(default=12)
    mask_hop:float = field(default=0.4)
    mask_prob:float = field(default=0.6)
    is_flash:bool = field(default=False)
    stat_path:Optional[str] = field(default=None)
    model_path:Optional[str] = field(default=None)
    w2v2_config_path:Optional[str] = field(default=None)
    use_rvq_target:bool = field(default=False)
    rvq_ckpt_path: Optional[str] = field(default=None)


SAMPLE_RATE = 24_000

@register_model("musicfm", dataclass=MusicFMConfig)
class MusicFMModel(BaseFairseqModel):
    def __init__(self, cfg: MusicFMConfig, task_cfg: FairseqTask):
        super().__init__()
        self.cfg = cfg
        self.model = MusicFM25Hz(
            num_codebooks=cfg.num_codebooks,
            codebook_dim=cfg.codebook_dim,
            codebook_size=cfg.codebook_size,
            features=cfg.features,
            n_mels=cfg.n_mels,
            conv_dim=cfg.conv_dim,
            encoder_dim=cfg.encoder_dim,
            encoder_depth=cfg.encoder_depth,
            mask_hop=cfg.mask_hop,
            mask_prob=cfg.mask_prob,
            is_flash=cfg.is_flash,
            stat_path=cfg.stat_path,
            model_path=cfg.model_path,
            w2v2_config_path=cfg.w2v2_config_path,
            use_rvq_target=cfg.use_rvq_target,
            rvq_ckpt_path=cfg.rvq_ckpt_path,
        )

    def forward(
        self,
        source: torch.Tensor, # B,L
        features_only: bool = False,
        **kwargs,
    ):
        source = source[..., :int((source.shape[-1]//(SAMPLE_RATE//self.cfg.label_rate))*(SAMPLE_RATE//self.cfg.label_rate)) ]
        # logger.info("source shape: "+str(source.shape))
        if features_only:
            _, hidden_states = self.model.get_predictions(source)
            result = {
                "layer_results": hidden_states
            }
            return result
        else:
            result = {}
            logits, hidden_emb, losses, accuracies = self.model(source)
            result["losses"] = losses
            result["accuracies"] = accuracies
            result["logits"] = logits
            result["hidden_emb"] = hidden_emb
            return result

    @classmethod
    def build_model(cls, cfg: MusicFMConfig, task: FairseqTask):
        """Build a new model instance."""

        model = MusicFMModel(cfg, task.cfg)
        import numpy as np
        s = 0
        for param in model.parameters():
            s += np.product(param.size())
        print('# of parameters: '+str(s/1024.0/1024.0))
        return model

    def get_losses(self, result, batch):
        return result['losses']
    
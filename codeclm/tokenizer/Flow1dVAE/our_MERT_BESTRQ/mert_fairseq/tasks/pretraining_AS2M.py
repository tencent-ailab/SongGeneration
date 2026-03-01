# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import sys

from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING, II

from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

try:
    from ..data.eat_data import MaeImageDataset
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from data.eat_data.mae_image_dataset import MaeImageDataset

logger = logging.getLogger(__name__)


@dataclass
class ImageMaskingConfig:
    patch_size: int = II("model.modalities.image.patch_size")
    mask_prob: float = II("model.modalities.image.mask_prob")
    mask_prob_adjust: float = II("model.modalities.image.mask_prob_adjust")
    mask_length: int = II("model.modalities.image.mask_length")
    inverse_mask: bool = II("model.modalities.image.inverse_mask")
    mask_dropout: float = II("model.modalities.image.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class MaeImagePretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    multi_data: Optional[List[str]] = None
    input_size: int = 224
    local_cache_path: Optional[str] = None
    key: str = "imgs"
    beit_transforms: bool = False
    target_transform: bool = False
    no_transform: bool = False

    rebuild_batches: bool = True
    precompute_mask_config: Optional[ImageMaskingConfig] = None
    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"
    
    audio_mae: bool = field(default=False,metadata={"help": "if set, we use image_mae way to deal with audio files."})
    h5_format: bool = field(default=False,metadata={"help": "if set, dataset will read data file in h5df format."})
    downsr_16hz: bool = field(default=False,metadata={"help": "if set, wav file's sample rate will be reduced to 16kHz."})
    target_length: int = field(default=1024,metadata={"help": "This setting will pad the audio spectrogram with zeros."})
    flexible_mask: bool = field(default=False, metadata={"help": "if true, we will using flexible inverse block mask method."})
    
    esc50_eval: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on esc50 dataset."})
    spcv2_eval: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on speech command v2 dataset."})
    AS2M_finetune: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on Audioset 2M with weighted sample."})
    spcv1_finetune: bool = field(default=False, metadata={"help": "if true, the task is to finetune model on speech commands v1  with weighted sample."})
    roll_aug: bool = field(default=False, metadata={"help": "if true, we will use roll aug in fine-tuning."})
    noise: bool = field(default=False, metadata={"help": "if true, we will add gaussian noise as augmentation during fine-tuning."})
    weights_file : str = field(default="", metadata={"help": "the path of weighted sample file"})
    num_samples: int = field(default=200000, metadata={"help": "this setting will determine the number of samples in each epoch, usually used in unbalanced training."})
    is_finetuning: bool = field(default=False, metadata={"help": "this property has been deprecated"})

    sample_rate: int = field(default=24000)
    fixed_duration: float = field(default=30.0)

    


@register_task("mae_image_pretraining", dataclass=MaeImagePretrainingConfig)
class MaeImagePretrainingTask(FairseqTask):
    """ """

    cfg: MaeImagePretrainingConfig

    @classmethod
    def setup_task(cls, cfg: MaeImagePretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg
        

        compute_mask = cfg.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:    
            mask_args = cfg.precompute_mask_config
        
        self.datasets[split] = MaeImageDataset(
            root=data_path if cfg.multi_data is None else cfg.multi_data,
            split=split,
            input_size=cfg.input_size,
            key=cfg.key,
            compute_mask=compute_mask,
            dataset_type=cfg.dataset_type,
            audio_mae=cfg.audio_mae, 
            downsr_16hz=cfg.downsr_16hz,
            h5_format=cfg.h5_format,
            esc50_eval=cfg.esc50_eval, 
            spcv2_eval=cfg.spcv2_eval,
            roll_aug=cfg.roll_aug and split == 'train',
            target_length=cfg.target_length,
            noise=cfg.noise,
            AS2M_finetune=cfg.AS2M_finetune,
            spcv1_finetune=cfg.spcv1_finetune,
            num_samples=cfg.num_samples,
            weights_file=cfg.weights_file,
            flexible_mask=cfg.flexible_mask,
            sample_rate=cfg.sample_rate,
            fixed_duration=cfg.fixed_duration,
            **mask_args,
        )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

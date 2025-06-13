# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataclasses import dataclass, field
from fairseq.data import Dictionary, HubertDataset
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

from ..data.mert_dataset import MERTDataset
from ..data.ark_dataset import ArkDataset

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label,
            append_eos=False,
            add_if_not_exist=False,
        )
class PaddedNumpyLabelEncoder(object):
    def __init__(self):
        # self.dictionary = dictionary
        pass

    def __call__(self, label):
        t = torch.IntTensor(np.asarray(label))
        t = t[t>=0] # remove padded -1 values at the end
        return t

@dataclass
class MERTPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    sharding_data: int = field(
        default=-1,
        metadata={
            "help": "set this para >1 to use sharding dataset to prevent OOM"
            "prepare data tsv and label files by adding postfix for sharding 64 like:"
            "train_28_64.tsv and train_28_64.encodec_6"
        },
    )
    load_random_data_shard: bool = field(
        default=True,
        metadata={
            "help": "whether to laod shards randomly or in order when use sharding_data"
        },
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1.0,
        metadata={"help": "label frame rate. -1.0 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys " "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    store_labels: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load all of the label into memory"}, 
    )

    numpy_memmap_label: Optional[bool] = field(
        default=False,
        metadata={"help": "whether the label file is saved as a numpy file, each line is ended with padding -1"}, 
    )

    augmentation_effects: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "a list of effects that might apply to the audios"
                "example: \"['random_mute', 'random_Gaussian', 'reverse_polarity']\" "
                "supported: random_mute,"
                "todo: "
            )
        },
    )
    augmentation_probs: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "the corresponding probabilities for the data augmentation effects"
                "example: \"[0.1, 0.5, 0.8]\" "
                "the sum is not necessarily need to be 1.0, and multiple effects can be applied to the same audio"
            )
        },
    )

    # inbatch_noise_augment_len_range: Optional[List[int]] = field(
        # default_factory=lambda: [8000, 24000],
        # default = [8000, 24000],
    inbatch_noise_augment_len_range: Optional[str] = field(
        default = "[8000, 24000]",
        metadata={
            "help": (
                "the range of length of the mix-up noise augmentation, unit in smaples"
            )
        },
    )
    # inbatch_noise_augment_number_range: Optional[List[int]] = field(
    #     default_factory=lambda: [1, 3],
        # default = [1, 3],
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
    dynamic_crops: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "used to set the maximum audio length setting, for training"
                "example: \"[1, 2, 3, 4, 5, 10]\" "
            )
        },
    )
    dynamic_crops_epoches: Optional[str] = field(
        default="[]",
        metadata={
            "help": (
                "used to set training epoches of changing the maximum audio length"
                "example: \"[1, 10, 20, 40, 80, 160,]\" "
                "then len need to be equal to len(dynamic_crops)"
            )
        },
    )

    cqt_loss_bin_dataloader: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "use this parameter to prepare cqt prediction objective in dataloader"
            )
        },
    )

    clip_secs: int = field(
        default=5,
        metadata={
            "help": "clip secs for each audio"
        }
    )


@register_task("mert_pretraining", dataclass=MERTPretrainingConfig)
class MERTPretrainingTask(FairseqTask):

    cfg: MERTPretrainingConfig

    def __init__(
        self,
        cfg: MERTPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MERTPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"        
        self.augmentation_effects = eval(self.cfg.augmentation_effects)
        self.augmentation_probs = eval(self.cfg.augmentation_probs)
        if len(self.augmentation_effects) > 0:
            assert len(self.augmentation_effects) == len(self.augmentation_probs)
            logger.info(f"Applying audio augmentation {self.augmentation_effects}, probabilities: {self.augmentation_probs}")
        
        self.inbatch_noise_augment_number_range = eval(self.cfg.inbatch_noise_augment_number_range)
        self.inbatch_noise_augment_len_range = eval(self.cfg.inbatch_noise_augment_len_range)

        self.max_sample_size = self.cfg.max_sample_size

        self.dynamic_crops = eval(self.cfg.dynamic_crops)
        self.dynamic_crops_epoches = eval(self.cfg.dynamic_crops_epoches)
        assert len(self.dynamic_crops) == len(self.dynamic_crops_epoches)
        if len(self.dynamic_crops) > 0:
            assert self.dynamic_crops_epoches[0] == 1
            
        self.cqt_loss_bin_dataloader = self.cfg.cqt_loss_bin_dataloader

        self.numpy_memmap_label = self.cfg.numpy_memmap_label
        self.store_labels = self.cfg.store_labels
        if self.numpy_memmap_label:
            assert self.store_labels

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: MERTPretrainingConfig, **kwargs
    ) -> "MERTPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if (self.cfg.label_dir is None or self.cfg.label_dir == '') else self.cfg.label_dir
        print(label_dir)
        dictionaries = [
            Dictionary.load(f"{label_dir}/dict.{label}.txt")
            for label in self.cfg.labels
        ]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None or self.cfg.label_dir=='':
            return self.cfg.data
        return self.cfg.label_dir

    def is_force_load_dataset(self, epoch, training_restore=False):
        # find the threshold that holds epoch \in [threshold, next_threshold)
        return (epoch in self.dynamic_crops_epoches) or training_restore or (self.cfg.sharding_data > 1)
        # for idx in range(len(self.dynamic_crops_epoches)):
        #     if (idx == len(self.dynamic_crops_epoches)-1) or \
        #         (epoch >= self.dynamic_crops_epoches[idx] and epoch < self.dynamic_crops_epoches[idx+1]):
        #         return True
        # return False

    def set_dynamic_crop_max_sample(self, epoch):
        """ force to set the max_sample_size config for the dynamic cropping function"""
        if epoch in self.dynamic_crops_epoches:
            for idx in range(len(self.dynamic_crops_epoches)):
                if (idx == len(self.dynamic_crops_epoches)-1) or \
                    (epoch >= self.dynamic_crops_epoches[idx] and epoch < self.dynamic_crops_epoches[idx+1]):
                        # set new cropping parameters and end loop
                        self.max_sample_size = self.dynamic_crops[idx]*self.cfg.sample_rate
                        self.cfg.max_sample_size = self.dynamic_crops[idx]*self.cfg.sample_rate
                        logger.info(f"epoch {epoch} forcely set new maximum audio length as {self.dynamic_crops[idx]}s == {self.max_sample_size} samples")
                        break
        # logger.info(f'reloading dataset for changing the sequence length')
        # self.load_dataset('train')
    #TODO : 修改数据地址
    def load_dataset(self, split: str, **kwargs) -> None:
        if len(list(filter(lambda s: s.endswith('.scp'), os.listdir(self.cfg.data)))) > 0:
            return self.load_dataset_ark(split, **kwargs)
        else:
            return self.load_dataset_mert(split, **kwargs)

    def load_dataset_ark(self, split, **kwargs):
        if 'train' not in split:
            logger.info(f'split {split} is only used for training')
            # raise ValueError(f"No support for split: {split}")
        else:
            self.datasets[split] = ArkDataset(
                wav_scp=os.path.join(self.cfg.data, f"wav_ark.scp"),
                dur_scp=os.path.join(self.cfg.data, f"dur_ark.scp"),
                sr=self.cfg.sample_rate,
            )

    def load_dataset_mert(self, split: str, **kwargs) -> None:
        if 'train' in split:
            epoch = kwargs['epoch']
            # the epoch to change crops
            if self.is_force_load_dataset(epoch):
                self.set_dynamic_crop_max_sample(epoch)
                
            # load all training data
            if self.cfg.sharding_data <= 1:            
                # manifest = f"{self.cfg.data}/{split}.tsv"
                manifest = f"{self.cfg.data}/{split}.json"

                paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]
            # load part of the training data
            else:
                if self.cfg.load_random_data_shard:
                    data_shard_idx = np.random.randint(self.cfg.sharding_data)
                else:
                    data_shard_idx = (epoch-1) % self.cfg.sharding_data # epoch start from 1
                assert data_shard_idx < self.cfg.sharding_data
                logger.info(f'loading shard {data_shard_idx} of {self.cfg.sharding_data} training data for ecpoh {epoch}')
                
                # manifest = f"{self.cfg.data}/{split}_{data_shard_idx}_{self.cfg.sharding_data}.tsv"
                manifest = f"{self.cfg.data}/{split}_{data_shard_idx}_{self.cfg.sharding_data}.json"

                paths = [f"{self.get_label_dir()}/{split}_{data_shard_idx}_{self.cfg.sharding_data}.{l}" for l in self.cfg.labels]
        else:
            # manifest = f"{self.cfg.data}/{split}.tsv"
            manifest = f"{self.cfg.data}/{split}.json"

            paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]

        if self.numpy_memmap_label:
            procs = [PaddedNumpyLabelEncoder() for dict in dicts]
        else:
            procs = [LabelEncoder(dict) for dict in dicts]

        self.datasets[split] = MERTDataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            label_paths=paths, # this containes the ensemble label sequence names
            label_rates=self.cfg.label_rate,
            pad_list=pad_list,
            eos_list=eos_list,
            label_processors=procs,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            store_labels=self.store_labels,
            npmemmap=self.numpy_memmap_label,
            random_crop=self.cfg.random_crop,
            single_target=self.cfg.single_target,
            augmentation_effects=self.augmentation_effects,
            augmentation_probs=self.augmentation_probs,
            inbatch_noise_augment_len_range=self.inbatch_noise_augment_len_range,
            inbatch_noise_augment_number_range=self.inbatch_noise_augment_number_range,
            inbatch_noise_augment_volume=self.cfg.inbatch_noise_augment_volume,
            cqt_prediction_bin=self.cqt_loss_bin_dataloader,
            clip_secs=self.cfg.clip_secs,
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

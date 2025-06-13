# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
import logging
import random
import time
import numpy as np
import os
import torch

from fairseq.data import FairseqDataset
try:
    from ..utils.data_utils import compute_block_mask_1d, compute_block_mask_2d
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from utils.data_utils import compute_block_mask_1d, compute_block_mask_2d
try:
    from .raw_audio_dataset import FileAudioDataset
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
    from raw_audio_dataset import FileAudioDataset

from shutil import copyfile

logger = logging.getLogger(__name__)


def load(path, loader, cache):
    if hasattr(caching_loader, "cache_root"):
        cache = caching_loader.cache_root

    cached_path = cache + path

    num_tries = 3
    for curr_try in range(num_tries):
        try:
            if curr_try == 2:
                return loader(path)
            if not os.path.exists(cached_path) or curr_try > 0:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                copyfile(path, cached_path)
                os.chmod(cached_path, 0o777)
            return loader(cached_path)
        except Exception as e:
            logger.warning(str(e))
            if "Errno 13" in str(e):
                caching_loader.cache_root = f"/scratch/{random.randint(0, 69420)}"
                logger.warning(f"setting cache root to {caching_loader.cache_root}")
                cached_path = caching_loader.cache_root + path
            if curr_try == (num_tries - 1):
                raise
            time.sleep(2)


def caching_loader(cache_root: str, loader):
    if cache_root is None:
        return loader

    if cache_root == "slurm_tmpdir":
        cache_root = os.environ["SLURM_TMPDIR"]
        assert len(cache_root) > 0

    if not cache_root.endswith("/"):
        cache_root += "/"

    return partial(load, loader=loader, cache=cache_root)


class MaeImageDataset(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        shuffle=True,
        key="imgs",
        compute_mask=False,
        patch_size: int = 16,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        audio_mae:bool = False,
        h5_format:bool = False,
        downsr_16hz:bool = False,
        target_length:int = 1024,
        esc50_eval:bool = False,
        spcv2_eval:bool = False,
        roll_aug: bool = False,
        noise: bool = False,
        dataset_type: str = "imagefolder",
        num_samples: int = 200000,       
        replacement:  bool = False,
        AS2M_finetune: bool = False,
        spcv1_finetune: bool =False,
        weights_file: str="",
        flexible_mask: bool = False,
        sample_rate=24000,
        fixed_duration=10,
    ):
        FairseqDataset.__init__(self)

        self.shuffle = shuffle
        self.key = key
        self.audio_mae = audio_mae
        if self.audio_mae:
            self.h5_format = h5_format
            self.downsr_16hz = downsr_16hz
            self.target_length = target_length
            self.esc50_eval = esc50_eval
            self.spcv2_eval = spcv2_eval
            self.noise = noise
            self.num_samples = num_samples
            self.replacement = replacement
            self.split = split
            self.AS2M_finetune = AS2M_finetune
            self.spcv1_finetune= spcv1_finetune
            self.weights_file = weights_file
            self.flexible_mask = flexible_mask

        self.transform_source = None
        self.transform_target = None
        self.img_shape = None
        self.roll_aug = roll_aug

        # load wav files
        mask_args = {}
        if self.audio_mae:
            min_sample_size = 10000 
            
            input_size = (self.target_length,128)
            manifest_path = os.path.join(root, "{}.jsonl".format(split))     
            self.dataset = FileAudioDataset(
                    manifest_path=manifest_path,
                    sample_rate=sample_rate,
                    fixed_duration=fixed_duration,
                    max_sample_size=sample_rate*fixed_duration,
                    min_sample_size=min_sample_size,
                    pad=False,
                    normalize=True,
                    num_buckets=0,
                    compute_mask=False,
                    h5_format=self.h5_format,
                    downsr_16hz=self.downsr_16hz,
                    wav2fbank=True,
                    target_length=self.target_length,
                    esc50_eval=self.esc50_eval,
                    spcv2_eval=self.spcv2_eval,
                    roll_mag_aug=self.roll_aug,
                    train_mode=split,
                    noise=self.noise,
                    **mask_args,
                )
            self.skipped_indices = self.dataset.skipped_indices
            
        else:
            raise Exception(f"invalid dataset type {dataset_type}")
        
            
        logger.info(f"loaded {len(self.dataset)} examples")

        self.is_compute_mask = compute_mask
        
        if type(input_size) == tuple:
            self.patches = (input_size[0] // patch_size ) * ( input_size[1] // patch_size )
            self.img_shape = (input_size[0] // patch_size,input_size[1] // patch_size )
            
        else:
            self.patches = (input_size // patch_size) ** 2  
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch

    def __getitem__(self, index):
        if self.audio_mae:
            img = self.dataset[index]['source']
        else:
            img, _ = self.dataset[index]

        source = None
        target = None

        v = {"id": index, self.key: source if source is not None else img}
        if target is not None:
            v["target"] = target

        # inverse block mask on audio patches
        if self.is_compute_mask:
            if self.mask_length == 1:
                mask = compute_block_mask_1d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                )
            else: # mask_length==5
                mask = compute_block_mask_2d(           
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                    expand_adjcent=self.expand_adjacent,
                    mask_dropout=self.mask_dropout,
                    non_overlapping=self.non_overlapping,
                    img_shape=self.img_shape,
                    flexible_mask=self.flexible_mask
                )

                if mask.shape[1] < self.patches:
                    padding = torch.zeros((mask.shape[0], self.patches - mask.shape[1]))
                    mask = torch.cat((mask, padding), dim=1)

            v["precomputed_mask"] = mask

        return v

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s[self.key] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                self.key: collated_img,
            },
        }

        if "target" in samples[0]:
            collated_target = torch.stack([s["target"] for s in samples], dim=0)
            res["net_input"]["target"] = collated_target

        if "precomputed_mask" in samples[0]:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            res["net_input"]["precomputed_mask"] = collated_mask

        return res

    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.full((len(self),), 1)

    # shuffle data (for pre-training and fine-tuning)
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle and (self.AS2M_finetune or self.spcv1_finetune) and self.split == "train" :
            weights = np.loadtxt(self.weights_file)
            normalized_weights = weights / np.sum(weights)
            weights_tensor = torch.from_numpy(normalized_weights)

            subsample_balanced_indicies = torch.multinomial(weights_tensor, self.num_samples, self.replacement)
            order = subsample_balanced_indicies.numpy()

            # order = [np.random.choice(order[0], size=len(self), replace=True, p=weights)]
            return order
        
        elif self.shuffle and self.split == "train":
            order = [np.random.permutation(len(self))]
            return order[0]

            
        else:
            order = [np.arange(len(self))]
            return order[0]

"""
Main model for using CodecLM. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings
import sys
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import numpy as np
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy
import pdb
from codeclm.models import builders
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from peft import LoraConfig, get_peft_model
from datetime import datetime
import os 
os.environ['TOKENIZERS_PARALLELISM'] = "false"


class CodecLM_PL(pl.LightningModule):
    def __init__(self, cfg, ckpt_path):
        super().__init__()

        self.cfg = cfg
        
        # 1) Build audio tokenizer (usually None during training)
        self.audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
        if self.audio_tokenizer is not None:
            for param in self.audio_tokenizer.parameters():
                param.requires_grad = False
        if "audio_tokenizer_checkpoint_sep" in self.cfg.keys():
            self.seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            for param in self.seperate_tokenizer.parameters():
                param.requires_grad = False
        else:
            self.seperate_tokenizer = None
        
        # 2) Build LM
        self.audiolm = builders.get_lm_model(self.cfg)
        print(self.audiolm)
        # 3) Load pretrained checkpoint (if any)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        missing, unexpected = self.load_state_dict(checkpoint, strict=False)
        print("successfully load pretrained model {}".format(ckpt_path))
        # 4) Build metrics
        self.val_steps = []
        self.train_slide_acc = []
        self.train_steps = []
        self.top1_acc_metric = nn.ModuleList([MulticlassAccuracy(
            self.audiolm.code_size, 
            top_k=1,
            average="micro", multidim_average="global",
            ignore_index=self.cfg.lm.code_size, # ignore EOS token prediction
        ) for _ in range(self.audiolm.code_depth)])
        self.top10_acc_metric = nn.ModuleList([MulticlassAccuracy(
            self.audiolm.code_size,
            top_k=10,
            average="micro", multidim_average="global",
            ignore_index=self.cfg.lm.code_size,
        ) for _ in range(self.audiolm.code_depth)])

        self.epoch = 0

    # TODO: move this part to loader
    def generate_mask_and_end_token(self, x, sequence_lengths, end_id=16384):
        batch_size = sequence_lengths.size(0)
        max_length = x.size(2)

        # pad one frame, if the maximum sequence length is equal to the input length
        if max_length == sequence_lengths.max():
            x = F.pad(x, (0, 1), value=end_id)
        max_length = x.size(2)

        if max_length <= sequence_lengths.max() + 1:
            sequence_lengths = sequence_lengths - (sequence_lengths.max()+1 - max_length)

        # Add end token to x according to the sequence length
        x[torch.arange(batch_size), :, sequence_lengths] = end_id
        sequence_lengths += 1

        mask = torch.arange(max_length).expand(batch_size, max_length) < sequence_lengths.unsqueeze(1)
        mask = mask.to(x.device)
        mask_3d = mask.unsqueeze(1).expand(batch_size, x.size(1), max_length)
        x = torch.where(mask_3d, x, end_id+1)
        return x, mask_3d

    def get_time(self):
        # 获取当前的日期和时间
        now = datetime.now()

        # 使用strftime函数格式化日期和时间
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        return formatted_now

class CosineLRScheduler(_LRScheduler):# 
    """Cosine LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):
        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= 0
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps
            lr = lr_ratio * lr
        elif step <= self.total_steps:
            s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * \
                (1. + math.cos(math.pi * s / self.cycle_length))
            lr = lr_ratio * lr
        else:
            lr_ratio = self.lr_min_ratio
            lr = lr_ratio * lr
        return lr

    def get_lr(self):
        return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]

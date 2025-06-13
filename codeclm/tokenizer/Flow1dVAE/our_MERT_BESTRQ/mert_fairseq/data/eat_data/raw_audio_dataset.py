# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import time
import io
try:
    import h5py
except:
    h5py = None

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from fairseq.data import FairseqDataset
try:
    from ..utils.data_utils import compute_block_mask_1d, get_buckets, get_bucketed_sizes
except:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from utils.data_utils import compute_block_mask_1d, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

import math 
from typing import Tuple
import json

logger = logging.getLogger(__name__)

def load_audio_by_json(json_path, max_keep, min_keep):
    # read json file
    n_long, n_short = 0, 0
    datas = []
    inds = []
    sizes = []
    with open(json_path) as fp:
        for ind,line in  enumerate(fp):
            data = json.loads(line)
            sz = int(data['duration'] * data['sample_rate'])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                datas.append(data)
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1 
    logger.info(
        (
            f"json_path={json_path}, "
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(datas)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return datas, inds, tot, sizes

class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask=False,
        feature_encoder_spec: str = "None",
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        corpus_key=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

        self.is_compute_mask = compute_mask
        self.feature_encoder_spec = eval(feature_encoder_spec)
        self._features_size_map = {}
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.corpus_key = corpus_key

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)


    def postprocess(self, feats, curr_sample_rate, roll_aug = False):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()
        # if self.normalize:
        #     with torch.no_grad():
        #         feats = F.layer_norm(feats, feats.shape)
        feats = feats - feats.mean()
        
        if roll_aug:
            feats = self._roll_mag_aug(feats)
        
        return feats

    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.corpus_key is not None:
            input["corpus_key"] = [self.corpus_key] * len(sources)
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self.crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self.feature_encoder_spec:
            L_in = size
            for (_, kernel_size, stride) in self.feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []

class Read_and_PadCrop_Normalized_T(torch.nn.Module):
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize


    def __call__(self, filename: str, duration: float, cur_sample_rate: int) -> Tuple[torch.Tensor, float, float, int, int]:
        if(duration<(float(self.n_samples)/self.sample_rate+1)):
            # print(duration,(float(self.n_samples)/self.sample_rate+1))
            chunk, _ = torchaudio.load(filename, frame_offset=0, num_frames=-1)
            t_start = 0.
            t_end = min(1.0, float(self.n_samples) / float(self.sample_rate) / duration)
            offset = 0
            # print('c1:',chunk.shape)
        else:
            offset = np.random.randint(0,int(duration*cur_sample_rate)-int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
            t_start = offset / float(cur_sample_rate) / duration
            t_end = t_start + float(self.n_samples) / float(self.sample_rate) / duration
            chunk, _ = torchaudio.load(filename, frame_offset=offset, num_frames=int(float(self.n_samples)/self.sample_rate*cur_sample_rate))
            # print('offset:',offset)
            # print('c0:',chunk.shape)
        # Pad with silence if necessary.
        if(chunk.shape[0]>1):
            chunk = chunk[torch.randint(chunk.shape[0], size=(1,)),:].float()
        else:
            chunk = chunk[[0],:].float()
        if(cur_sample_rate!=self.sample_rate):
            # print('a:',cur_sample_rate,chunk.shape)
            chunk = torchaudio.functional.resample(chunk, cur_sample_rate, self.sample_rate)
            # print('b:',self.sample_rate,chunk.shape)
        if chunk.shape[-1] < self.n_samples:
            chunk = torch.cat([chunk, torch.zeros((1, self.n_samples - chunk.shape[-1],))],-1)
        else:
            chunk = chunk[:,0:self.n_samples]
        seconds_start = math.floor(offset / cur_sample_rate)
        seconds_total = math.floor(duration)

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total
        )

class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        fixed_duration=None,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask=False,
        text_compression_level=TextCompressionLevel.none,
        h5_format=False,
        downsr_16hz=False,
        wav2fbank=False,
        target_length=1024,
        esc50_eval=False,
        spcv2_eval=False,
        roll_mag_aug=False,
        noise=False,
        train_mode='train',
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask=compute_mask,
            **mask_compute_kwargs,
        )

        self.text_compressor = TextCompressor(level=text_compression_level)
        self.h5_format = h5_format
        self.downsr_16hz = downsr_16hz
        self.wav2fbank = wav2fbank
        self.target_length = target_length
        self.esc50_eval = esc50_eval
        self.spcv2_eval = spcv2_eval
        self.roll_mag_aug = roll_mag_aug
        self.noise = noise
        self.train_mode = train_mode
        self.reader = Read_and_PadCrop_Normalized_T(n_samples = int(fixed_duration*sample_rate), sample_rate = sample_rate)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        # exclude data not in sample rate range     10.h5/****.wav  320000 
        self.durations = []
        self.raw_srs = []
        datas, inds, tot, sizes = load_audio_by_json(manifest_path, max_keep=None, min_keep=None)
        for data in datas:
            self.fnames.append(self.text_compressor.compress(data['path']))
            self.durations.append(data['duration'])
            self.raw_srs.append(data['sample_rate'])

        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        if self.esc50_eval:
            task_dataset = "ESC-50"
        elif self.spcv2_eval:
            task_dataset = "SPC-2"
        else:
            task_dataset = "AS"
        
        logger.info(
            f"sample rate: {sample_rate}\t"
            f"target length: {self.target_length}\t"
            f"current task: {task_dataset}\t"
        )

        # self.sizes = np.array(sizes, dtype=np.int64)
        self.sizes = np.array(sizes, dtype=np.int64)
        self.durations = np.array(self.durations)
        self.raw_srs = np.array(self.raw_srs)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

        self.set_bucket_info(num_buckets)
        # print("skipped_index: {}".format(self.skipped_indices))
        # print(len(self.skipped_indices))

    # two file format. h5_format = true -> .h5(.hdf5) ; h5_format = false -> .wav
    def __getitem__(self, index):
        import soundfile as sf

        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        path_or_fp = fn # os.path.join(self.root_dir, fn)
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])   # root/10.h5/***.wav
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)

        retry = 3
        wav = None
        for i in range(retry):
            try:
                # if self.h5_format and self.train_mode == 'train':
                #     parts = path_or_fp.split("/")
                #     path_or_fp = "/".join(parts[:-1])
                #     path_or_fp = h5py.File(path_or_fp,'r')
                #     wav = path_or_fp[parts[-1]][:]
                #     curr_sample_rate = 32000
                #     break                    
                # else:
                #     wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")
                #     break
                wav, *ignored = self.reader(fn, self.durations[index], self.raw_srs[index])
                curr_sample_rate = self.reader.sample_rate
            except Exception as e:
                logger.warning(
                    f"Failed to read {path_or_fp}: {e}. Sleeping for {1 * i}"
                )
                time.sleep(1 * i)

        if wav is None:
            raise Exception(f"Failed to load {path_or_fp}")

        if self.h5_format:
            feats = torch.tensor(wav).float()
        else:
            if not isinstance(wav, torch.Tensor):
                feats = torch.from_numpy(wav).float()
            else:
                feats = wav
            if len(feats.shape) == 2:
                feats = feats.squeeze(dim=0)
            
        if self.downsr_16hz:
            feats = torchaudio.functional.resample(feats, orig_freq=curr_sample_rate, new_freq=16000)
            curr_sample_rate = 16000
            self.sample_rate = curr_sample_rate
            
        # whether to use roll augmentation on waveform
        use_roll = self.roll_mag_aug and self.train_mode == 'train'
        
        feats = self.postprocess(feats, curr_sample_rate, use_roll)

        # convert waveform to spectrogram
        if self.wav2fbank: #这里，将wav转换为mel谱
            feats = feats.unsqueeze(dim=0)
            feats = torchaudio.compliance.kaldi.fbank(feats, htk_compat=True, sample_frequency=curr_sample_rate, use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=40).unsqueeze(dim=0)
            
            # padding 
            n_frames = feats.shape[1]
            diff = self.target_length - n_frames #时间维度上补齐至1024
            if diff > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
                feats = m(feats)
                
            elif diff < 0:
                feats = feats[0:self.target_length, :]     
                
            # global normalization for AS
            self.norm_mean = -4.268 
            self.norm_std = 4.569
            
            # global normalization for ESC-50
            if self.esc50_eval:
                self.norm_mean = -6.627
                self.norm_std = 5.359
                
            # global normalization for spcv2
            if self.spcv2_eval:
                self.norm_mean = -6.846
                self.norm_std = 5.565
                
            feats = (feats - self.norm_mean) / (self.norm_std * 2) 
            
            if self.noise and self.train_mode == 'train': 
                feats = feats + torch.rand(feats.shape[1], feats.shape[2]) * np.random.rand() / 10 # 这个加noise的方式视情况可能要换成inbatch noise
                feats = torch.roll(feats, np.random.randint(-10, 10), 1)

        v = {"id": index, "source": feats}

        if self.is_compute_mask:
            T = self._get_mask_indices_dims(feats.size(-1))
            mask = compute_block_mask_1d(
                shape=(self.clone_batch, T),
                mask_prob=self.mask_prob,
                mask_length=self.mask_length,
                mask_prob_adjust=self.mask_prob_adjust,
                inverse_mask=self.inverse_mask,
                require_same_masks=True,
                expand_adjcent=self.expand_adjacent,
                mask_dropout=self.mask_dropout,
                non_overlapping=self.non_overlapping,
            )

            v["precomputed_mask"] = mask

        return v

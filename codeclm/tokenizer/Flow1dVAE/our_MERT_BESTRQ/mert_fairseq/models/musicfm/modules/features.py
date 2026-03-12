# MIT License
#
# Copyright 2023 ByteDance Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import torchaudio
from torch import nn
import torch

class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
        is_db=False,
    ):
        super(MelSTFT, self).__init__()

        # spectrogram
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # amplitude to decibel
        self.is_db = is_db
        if is_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        # 将数据移至 CPU 处理 STFT，再移回 GPU
        device = waveform.device
        waveform_cpu = waveform.cpu()
        # 强制在 CPU 上运行
        with torch.cpu.amp.autocast(enabled=False):
            if self.is_db:
                spec = self.amplitude_to_db(self.mel_stft.to('cpu')(waveform_cpu))
            else:
                spec = self.mel_stft.to('cpu')(waveform_cpu)
        # 结果移回原设备，并将 mel_stft 移回原设备供下次使用（或者克隆一个 cpu 版的）
        spec = spec.to(device)
        self.mel_stft.to(device)
        return spec

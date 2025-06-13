from typing import Callable, Optional
import torch
from torchaudio.transforms import Spectrogram
from nnAudio.features.cqt import CQT

def _hz_to_octs(freqs, tuning=0.0, bins_per_octave=12):
    a440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return torch.log2(freqs / (a440 / 16))

def chroma_filterbank(
    sample_rate: int,
    n_freqs: int,
    n_chroma: int,
    *,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    norm: int = 2,
    base_c: bool = True,
):
    """Create a frequency-to-chroma conversion matrix. Implementation adapted from librosa.

    Args:
        sample_rate (int): Sample rate.
        n_freqs (int): Number of input frequencies.
        n_chroma (int): Number of output chroma.
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Returns:
        torch.Tensor: Chroma filter bank, with shape `(n_freqs, n_chroma)`.
    """
    # Skip redundant upper half of frequency range.
    freqs = torch.linspace(0, sample_rate // 2, n_freqs)[1:] # ->[n_freqs - 1], 均分sample_rate//2 ; 对哪些频率感兴趣
    freq_bins = n_chroma * _hz_to_octs(freqs, bins_per_octave=n_chroma, tuning=tuning) # 这些频率对应的octave坐标下的值（类似于MIDI的音高序号）
    freq_bins = torch.cat((torch.tensor([freq_bins[0] - 1.5 * n_chroma]), freq_bins))
    freq_bin_widths = torch.cat(
        (
            torch.maximum(freq_bins[1:] - freq_bins[:-1], torch.tensor(1.0)),
            torch.tensor([1]),
        )
    ) #每个波带对应的octave的宽度（至少为1）

    # (n_freqs, n_chroma)
    D = freq_bins.unsqueeze(1) - torch.arange(0, n_chroma)

    n_chroma2 = round(n_chroma / 2)

    # Project to range [-n_chroma/2, n_chroma/2 - 1] #D:[1025, 12]
    D = torch.remainder(D + n_chroma2, n_chroma) - n_chroma2

    fb = torch.exp(-0.5 * (2 * D / torch.tile(freq_bin_widths.unsqueeze(1), (1, n_chroma))) ** 2)
    fb = torch.nn.functional.normalize(fb, p=norm, dim=1) #->[1025, 12]

    if octwidth is not None:
        fb *= torch.tile(
            torch.exp(-0.5 * (((freq_bins.unsqueeze(1) / n_chroma - ctroct) / octwidth) ** 2)),
            (1, n_chroma),
        ) #->[1025, 12]

    if base_c:
        fb = torch.roll(fb, -3 * (n_chroma // 12), dims=1)

    return fb

class ChromaScale(torch.nn.Module):
    r"""Converts spectrogram to chromagram.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_freqs (int): Number of frequency bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> spectrogram_transform = transforms.Spectrogram(n_fft=1024)
        >>> spectrogram = spectrogram_transform(waveform)
        >>> chroma_transform = transforms.ChromaScale(sample_rate=sample_rate, n_freqs=1024 // 2 + 1)
        >>> chroma_spectrogram = chroma_transform(spectrogram)

    See also:
        :py:func:`torchaudio.prototype.functional.chroma_filterbank` — function used to
        generate the filter bank.
    """

    def __init__(
        self,
        sample_rate: int,
        n_freqs: int,
        *,
        n_chroma: int = 12,
        tuning: float = 0.0,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        norm: int = 2,
        base_c: bool = True,
    ):
        super().__init__()
        fb = chroma_filterbank(
            sample_rate, n_freqs, n_chroma, tuning=tuning, ctroct=ctroct, octwidth=octwidth, norm=norm, base_c=base_c
        )
        self.register_buffer("fb", fb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            specgram (torch.Tensor): Spectrogram of dimension (..., ``n_freqs``, time).

        Returns:
            torch.Tensor: Chroma spectrogram of size (..., ``n_chroma``, time).
        """
        return torch.matmul(x.transpose(-1, -2), self.fb).transpose(-1, -2) #[376, 1025] @ [1025, 12]-> [12, 376]


class ChromaSpectrogram(torch.nn.Module):
    r"""Generates chromagram for audio signal.

    .. devices:: CPU CUDA

    .. properties:: Autograd

    Composes :py:func:`torchaudio.transforms.Spectrogram` and
    and :py:func:`torchaudio.prototype.transforms.ChromaScale`.

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins.
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., torch.Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        n_chroma (int, optional): Number of chroma. (Default: ``12``)
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.ChromaSpectrogram(sample_rate=sample_rate, n_fft=400)
        >>> chromagram = transform(waveform)  # (channel, n_chroma, time)
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        *,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        n_chroma: int = 12,
        n_bins: Optional[int] = None,
        tuning: float = 0.0,
        ctroct: float = 5.0,
        octwidth: Optional[float] = 2.0,
        norm: int = 2,
        base_c: bool = True,
        use_cqt: bool = False,
    ):
        super().__init__()
        if n_bins is None:
            n_bins = n_fft // 2 + 1
        if use_cqt:
            self.spectrogram = CQT(
                sr=sample_rate, 
                hop_length=hop_length,
                n_bins=n_bins, 
                bins_per_octave=n_bins//7, 
                filter_scale=1, 
                norm=1, 
                window='hann', 
                center=True, 
                pad_mode='constant', 
                trainable=False, 
                output_format='Magnitude', 
                verbose=True,
            )
        else:
            self.spectrogram = Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad=pad,
                window_fn=window_fn,
                power=power,
                normalized=normalized,
                wkwargs=wkwargs,
                center=center,
                pad_mode=pad_mode,
                onesided=True,
            )
        self.chroma_scale = ChromaScale(
            sample_rate,
            n_bins,
            n_chroma=n_chroma,
            tuning=tuning,
            base_c=base_c,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
        )

    def forward(self, waveform: torch.Tensor, normalize=True) -> torch.Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Chromagram of size (..., ``n_chroma``, time).
        """
        spectrogram = self.spectrogram(waveform) #[1025, 376]
        chroma_spectrogram = self.chroma_scale(spectrogram)
        if normalize:
            chroma_spectrogram[chroma_spectrogram < 0] = 0.0
            chroma_spectrogram = torch.nn.functional.normalize(chroma_spectrogram, p=2, dim=-2)
        return chroma_spectrogram

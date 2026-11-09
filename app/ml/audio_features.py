"""Ekstraksi fitur akustik per segmen (numpy + scipy, tanpa librosa).

Fitur: MFCC (mean/std), Delta MFCC (mean/std), F0 (mean/std/voiced ratio, autokorelasi
ternormalisasi), Spectral Centroid (mean/std), ZCR (mean/std), log-RMS (mean/std).

Implementasi mengikuti definisi standar (HTK mel scale, log-mel -> DCT-II ortho, delta regresi
width=2). Nilainya tidak identik bit-per-bit dengan librosa (beda default window/padding),
jadi model WAJIB dilatih dan dipakai dengan extractor yang sama — ini yang dijamin modul ini.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import dct, rfft, irfft

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class FeatureConfig:
    sr: int = SAMPLE_RATE
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    n_mfcc: int = 20
    fmin: float = 20.0
    fmax: float = 7600.0
    f0_min: float = 65.0
    f0_max: float = 400.0
    voicing_threshold: float = 0.45
    preemphasis: float = 0.97

    @property
    def frame_len(self) -> int:
        return int(round(self.sr * self.frame_ms / 1000))

    @property
    def hop_len(self) -> int:
        return int(round(self.sr * self.hop_ms / 1000))

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = FeatureConfig()


def feature_names(cfg: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    names: list[str] = []
    for prefix in ("mfcc_mean", "mfcc_std", "dmfcc_mean", "dmfcc_std"):
        names += [f"{prefix}_{i}" for i in range(cfg.n_mfcc)]
    names += ["f0_mean", "f0_std", "voiced_ratio", "centroid_mean", "centroid_std",
              "zcr_mean", "zcr_std", "logrms_mean", "logrms_std"]
    return names


def feature_dim(cfg: FeatureConfig = DEFAULT_CONFIG) -> int:
    return 4 * cfg.n_mfcc + 9


def frame_signal(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size))
    n_frames = 1 + (x.size - frame_len) // hop_len
    idx = np.arange(frame_len)[None, :] + hop_len * np.arange(n_frames)[:, None]
    return x[idx]


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fmax = min(fmax, sr / 2)
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, bins.size), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = hz[m - 1], hz[m], hz[m + 1]
        up = (bins - left) / max(center - left, 1e-9)
        down = (right - bins) / max(right - center, 1e-9)
        fb[m - 1] = np.maximum(0.0, np.minimum(up, down))
    # Slaney-style area normalization
    fb *= (2.0 / (hz[2:n_mels + 2] - hz[:n_mels]))[:, None]
    return fb


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Delta regresi standar: d_t = sum_n n*(c_{t+n}-c_{t-n}) / (2*sum n^2)."""
    if feat.shape[0] == 0:
        return feat.copy()
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat)
    for n in range(1, width + 1):
        out += n * (padded[width + n: width + n + feat.shape[0]] - padded[width - n: width - n + feat.shape[0]])
    return out / denom


def cmvn(feat: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> np.ndarray:
    mean = feat.mean(axis=0) if mean is None else mean
    std = feat.std(axis=0) if std is None else std
    return (feat - mean) / np.maximum(std, 1e-8)


def _f0_autocorr(frames: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """F0 per frame via autokorelasi ternormalisasi (FFT). NaN untuk frame unvoiced."""
    frames = frames - frames.mean(axis=1, keepdims=True)
    n = frames.shape[1]
    nfft = 1 << (2 * n - 1).bit_length()
    spec = rfft(frames, n=nfft, axis=1)
    ac = irfft(spec * np.conj(spec), n=nfft, axis=1)[:, :n]
    energy = ac[:, :1]
    ac_norm = ac / np.maximum(energy, 1e-10)
    lag_min = int(cfg.sr / cfg.f0_max)
    lag_max = min(int(cfg.sr / cfg.f0_min), n - 1)
    f0 = np.full(frames.shape[0], np.nan)
    if lag_max <= lag_min:
        return f0
    region = ac_norm[:, lag_min:lag_max]
    best = np.argmax(region, axis=1)
    peak = region[np.arange(region.shape[0]), best]
    voiced = (peak >= cfg.voicing_threshold) & (energy[:, 0] > 1e-6)
    f0[voiced] = cfg.sr / (best[voiced] + lag_min)
    return f0


def frame_level_mfcc(audio: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    x = np.append(audio[0], audio[1:] - cfg.preemphasis * audio[:-1]) if audio.size > 1 else audio
    frames = frame_signal(x.astype(np.float64), cfg.frame_len, cfg.hop_len)
    window = np.hamming(cfg.frame_len)
    power = np.abs(rfft(frames * window, n=cfg.n_fft, axis=1)) ** 2 / cfg.n_fft
    fb = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    log_mel = np.log(np.maximum(power @ fb.T, 1e-10))
    return dct(log_mel, type=2, axis=1, norm="ortho")[:, : cfg.n_mfcc]


def extract_segment_features(audio: np.ndarray, cfg: FeatureConfig = DEFAULT_CONFIG,
                             apply_cmvn: bool = False) -> np.ndarray:
    """Vektor fitur fixed-length (dim = feature_dim(cfg)) untuk satu segmen audio."""
    audio = np.asarray(audio, dtype=np.float64).ravel()
    min_len = cfg.frame_len + cfg.hop_len * 4
    if audio.size < min_len:
        audio = np.pad(audio, (0, min_len - audio.size))

    mfcc = frame_level_mfcc(audio, cfg)
    if apply_cmvn:
        mfcc = cmvn(mfcc)
    dmfcc = delta(mfcc)

    raw_frames = frame_signal(audio, cfg.frame_len, cfg.hop_len)
    window = np.hamming(cfg.frame_len)
    power = np.abs(rfft(raw_frames * window, n=cfg.n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sr)
    centroid = (power @ freqs) / np.maximum(power.sum(axis=1), 1e-10)
    signs = np.signbit(raw_frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    logrms = np.log(np.sqrt(np.mean(raw_frames ** 2, axis=1)) + 1e-8)
    f0 = _f0_autocorr(raw_frames, cfg)
    voiced = ~np.isnan(f0)
    voiced_ratio = float(voiced.mean())
    f0_mean = float(np.nanmean(f0)) if voiced.any() else 0.0
    f0_std = float(np.nanstd(f0)) if voiced.sum() > 1 else 0.0

    vec = np.concatenate([
        mfcc.mean(axis=0), mfcc.std(axis=0), dmfcc.mean(axis=0), dmfcc.std(axis=0),
        [f0_mean, f0_std, voiced_ratio, centroid.mean(), centroid.std(),
         zcr.mean(), zcr.std(), logrms.mean(), logrms.std()],
    ]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def extract_many(audio: np.ndarray, spans: list[tuple[float, float]],
                 cfg: FeatureConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Fitur untuk banyak span (detik) dari satu rekaman. Output (n, dim)."""
    out = np.zeros((len(spans), feature_dim(cfg)), dtype=np.float32)
    for i, (start, end) in enumerate(spans):
        s = max(0, int(start * cfg.sr))
        e = min(audio.size, int(end * cfg.sr))
        out[i] = extract_segment_features(audio[s:e] if e > s else np.zeros(cfg.frame_len), cfg)
    return out

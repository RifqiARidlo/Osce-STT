"""Audio data augmentation untuk melatih Speaker Role Classifier (mitigasi Gap 3)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import fftconvolve, resample_poly


def add_noise(audio: np.ndarray, rng: np.random.Generator, snr_db: float,
              noise: np.ndarray | None = None) -> np.ndarray:
    if noise is None or noise.size == 0:
        # pink-ish noise: white noise di-filter 1/f sederhana
        white = rng.standard_normal(audio.size)
        spec = np.fft.rfft(white)
        freqs = np.arange(spec.size) + 1.0
        noise = np.fft.irfft(spec / np.sqrt(freqs), n=audio.size)
    elif noise.size < audio.size:
        noise = np.tile(noise, int(np.ceil(audio.size / noise.size)))[: audio.size]
    else:
        start = int(rng.integers(0, noise.size - audio.size + 1))
        noise = noise[start: start + audio.size]
    p_sig = float(np.mean(audio ** 2)) + 1e-12
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    gain = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (audio + gain * noise).astype(np.float32)


def change_gain(audio: np.ndarray, db: float) -> np.ndarray:
    return np.clip(audio * 10 ** (db / 20), -1.0, 1.0).astype(np.float32)


def pitch_speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Resampling -> pitch & tempo bergeser bersamaan (speed perturbation, à la Kaldi)."""
    frac = Fraction(factor).limit_denominator(50)
    return resample_poly(audio, frac.denominator, frac.numerator).astype(np.float32)


def simple_reverb(audio: np.ndarray, rng: np.random.Generator, sr: int = 16000, rt60: float = 0.3) -> np.ndarray:
    length = int(sr * rt60)
    t = np.arange(length) / sr
    ir = rng.standard_normal(length) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = fftconvolve(audio, ir)[: audio.size]
    peak = np.max(np.abs(wet)) + 1e-9
    return (wet / peak * min(1.0, np.max(np.abs(audio)) + 1e-9)).astype(np.float32)


def random_augment(audio: np.ndarray, rng: np.random.Generator, noise_bank: list[np.ndarray] | None = None) -> np.ndarray:
    out = audio.astype(np.float32)
    if rng.random() < 0.5:
        out = pitch_speed_perturb(out, float(rng.choice([0.92, 0.96, 1.04, 1.08])))
    if rng.random() < 0.3:
        out = simple_reverb(out, rng, rt60=float(rng.uniform(0.15, 0.5)))
    if rng.random() < 0.7:
        noise = noise_bank[int(rng.integers(len(noise_bank)))] if noise_bank else None
        out = add_noise(out, rng, snr_db=float(rng.uniform(5, 25)), noise=noise)
    out = change_gain(out, float(rng.uniform(-8, 4)))
    return out

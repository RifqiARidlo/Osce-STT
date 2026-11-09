"""Stage 1: noise reduction (spectral gating) + Voice Activity Detection.

- Spectral gating stationary (numpy/scipy). RNNoise tidak dipakai: butuh binding C tambahan dan
  di Windows sering bermasalah; spectral gating cukup untuk bising stasioner (AC, kipas).
- VAD: Silero VAD versi ONNX yang sudah dibundel di faster-whisper (tanpa torch).
  Fallback: VAD berbasis energi bila faster-whisper tidak terpasang (mis. saat unit test).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

from app.schemas.pipeline import SpeechRegion

log = logging.getLogger(__name__)
SR = 16000


def spectral_gate(audio: np.ndarray, sr: int = SR, n_std: float = 1.5, prop_decrease: float = 0.85,
                  noise_quantile: float = 0.15) -> np.ndarray:
    """Redam bising stasioner. Profil bising diestimasi dari frame berenergi terendah."""
    if audio.size < sr // 2:
        return audio
    nperseg, noverlap = 512, 384
    _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Z)
    db = 20 * np.log10(mag + 1e-10)
    frame_energy = db.mean(axis=0)
    k = max(1, int(noise_quantile * frame_energy.size))
    quiet = np.argsort(frame_energy)[:k]
    noise_mean = db[:, quiet].mean(axis=1, keepdims=True)
    noise_std = db[:, quiet].std(axis=1, keepdims=True)
    threshold = noise_mean + n_std * noise_std
    mask = (db > threshold).astype(np.float64)
    mask = uniform_filter(mask, size=(3, 5))  # smoothing frekuensi x waktu -> kurangi musical noise
    gain = mask + (1.0 - mask) * (1.0 - prop_decrease)
    _, cleaned = istft(Z * gain, fs=sr, nperseg=nperseg, noverlap=noverlap)
    cleaned = cleaned[: audio.size]
    if cleaned.size < audio.size:
        cleaned = np.pad(cleaned, (0, audio.size - cleaned.size))
    return cleaned.astype(np.float32)


def _merge_regions(flags: np.ndarray, hop_s: float, min_speech_s: float, min_silence_s: float,
                   pad_s: float, total_s: float) -> list[SpeechRegion]:
    regions: list[list[float]] = []
    start = None
    for i, active in enumerate(flags):
        t = i * hop_s
        if active and start is None:
            start = t
        elif not active and start is not None:
            regions.append([start, t])
            start = None
    if start is not None:
        regions.append([start, len(flags) * hop_s])
    merged: list[list[float]] = []
    for reg in regions:
        if merged and reg[0] - merged[-1][1] < min_silence_s:
            merged[-1][1] = reg[1]
        else:
            merged.append(reg)
    out = []
    for s, e in merged:
        if e - s >= min_speech_s:
            out.append(SpeechRegion(start=max(0.0, s - pad_s), end=min(total_s, e + pad_s)))
    return out


def energy_vad(audio: np.ndarray, sr: int = SR, min_speech_ms: int = 250, min_silence_ms: int = 300,
               pad_ms: int = 100) -> list[SpeechRegion]:
    hop = int(0.01 * sr)
    win = int(0.03 * sr)
    if audio.size < win:
        return []
    n = 1 + (audio.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    energy_db = 10 * np.log10(np.mean(audio[idx] ** 2, axis=1) + 1e-10)
    p10, p95 = np.percentile(energy_db, [10, 95])
    threshold = p10 + max(6.0, 0.3 * (p95 - p10))
    flags = energy_db > threshold
    return _merge_regions(flags, hop / sr, min_speech_ms / 1000, min_silence_ms / 1000, pad_ms / 1000,
                          audio.size / sr)


def silero_vad(audio: np.ndarray, threshold: float, min_speech_ms: int, min_silence_ms: int,
               pad_ms: int) -> list[SpeechRegion]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold, min_speech_duration_ms=min_speech_ms,
                      min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms)
    stamps = get_speech_timestamps(audio, opts)
    return [SpeechRegion(start=s["start"] / SR, end=s["end"] / SR) for s in stamps]


def detect_speech(audio: np.ndarray, backend: str = "silero", threshold: float = 0.5,
                  min_speech_ms: int = 250, min_silence_ms: int = 300, pad_ms: int = 100) -> tuple[list[SpeechRegion], str]:
    if backend == "silero":
        try:
            return silero_vad(audio, threshold, min_speech_ms, min_silence_ms, pad_ms), "silero"
        except ImportError:
            log.warning("faster-whisper (Silero VAD) tidak tersedia -> fallback energy VAD.")
        except Exception as exc:  # model ONNX gagal dimuat, dsb.
            log.warning("Silero VAD gagal (%s) -> fallback energy VAD.", exc)
    return energy_vad(audio, SR, min_speech_ms, min_silence_ms, pad_ms), "energy"

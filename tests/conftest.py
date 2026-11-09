"""Fixture uji: suara sintetis (sumber harmonik + filter formant) untuk dua 'speaker'."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import iirpeak, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
DOCTOR_VOICE = (118.0, (700, 1200, 2600))
PATIENT_VOICE = (215.0, (850, 1700, 3000))


def synth_voice(f0: float, formants: tuple[int, ...], dur: float, seed: int = 0, noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (1 + 0.04 * np.sin(2 * np.pi * (2.5 + rng.random()) * t)) * (1 + 0.02 * rng.standard_normal())
    phase = 2 * np.pi * np.cumsum(f) / SR
    src = sum(np.sin(k * phase) / k for k in range(1, 25))
    out = np.zeros_like(src)
    for fc in formants:
        b, a = iirpeak(fc * (1 + 0.02 * rng.standard_normal()), 5, fs=SR)
        out += lfilter(b, a, src)
    out *= 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t) ** 2
    out /= np.max(np.abs(out)) + 1e-9
    return (0.6 * out + noise * rng.standard_normal(t.size)).astype(np.float32)


def synth_dialogue(roles: list[str], seg_dur: float = 1.8, gap: float = 0.7, seed: int = 0,
                   doctor=DOCTOR_VOICE, patient=PATIENT_VOICE) -> tuple[np.ndarray, list[tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    parts, spans, t = [np.zeros(int(0.5 * SR), np.float32)], [], 0.5
    for i, role in enumerate(roles):
        f0, formants = doctor if role == "DOKTER" else patient
        dur = seg_dur * (0.8 + 0.4 * rng.random())
        parts.append(synth_voice(f0, formants, dur, seed=seed * 100 + i))
        spans.append((t, t + dur))
        t += dur
        parts.append(0.003 * rng.standard_normal(int(gap * SR)).astype(np.float32))
        t += gap
    return np.concatenate(parts), spans


@pytest.fixture
def doctor_enroll_audio() -> np.ndarray:
    return synth_voice(*DOCTOR_VOICE, dur=14.0, seed=1)

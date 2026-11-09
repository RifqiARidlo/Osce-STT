"""Load & normalisasi audio ke float32 mono 16 kHz.

WAV PCM 16-bit mono 16 kHz (format yang dikirim web client) dibaca via stdlib `wave`.
Format lain (webm/ogg/mp3/m4a dari upload manual) dikonversi via ffmpeg (subprocess).
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioLoadError(RuntimeError):
    pass


def _read_wav_stdlib(path: Path) -> np.ndarray | None:
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2 or wf.getnchannels() not in (1, 2):
                return None
            raw = wf.readframes(wf.getnframes())
            channels = wf.getnchannels()
    except (wave.Error, EOFError):
        return None
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    return pcm


def _read_ffmpeg(path: Path) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError("Format audio bukan WAV 16 kHz PCM16 dan ffmpeg tidak ditemukan di PATH.")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioLoadError(f"ffmpeg gagal: {proc.stderr.decode(errors='replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def load_audio(path: Path, max_minutes: float | None = None) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"File audio tidak ditemukan: {path}")
    audio = _read_wav_stdlib(path)
    if audio is None:
        audio = _read_ffmpeg(path)
    if audio.size < SAMPLE_RATE // 2:
        raise AudioLoadError("Audio terlalu pendek (< 0.5 detik).")
    if max_minutes is not None and audio.size / SAMPLE_RATE > max_minutes * 60:
        raise AudioLoadError(f"Audio melebihi batas {max_minutes} menit.")
    audio = np.nan_to_num(audio.astype(np.float32))
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio /= peak
    return audio


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

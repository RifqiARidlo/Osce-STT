"""Stage 2a: ASR Faster-Whisper + segmentasi kata -> utterance.

Satu panggilan transcribe untuk seluruh rekaman (word_timestamps=True) lalu kata-kata
dikelompokkan menjadi utterance memakai batas region VAD Stage 1, jeda antar-kata, dan tanda
baca akhir kalimat. Ini menggantikan WhisperX: word timestamps faster-whisper (cross-attention
DTW) sudah cukup untuk memotong giliran bicara pada skenario 1 mikrofon dengan SOP giliran.
"""
from __future__ import annotations

import bisect
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas.pipeline import SpeechRegion, Word

log = logging.getLogger(__name__)

# Whisper membatasi prompt ~224 token; ~700 karakter Bahasa Indonesia aman.
MAX_PROMPT_CHARS = 700


def load_glossary_prompt(path: Path) -> str:
    if not path.is_file():
        return ""
    terms = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    prompt = "Percakapan konsultasi dokter dan pasien. Istilah: " + ", ".join(terms)
    return prompt[:MAX_PROMPT_CHARS].rsplit(",", 1)[0] + "."


_nvidia_dirs_registered = False


def register_nvidia_dll_dirs() -> list[str]:
    """Windows: daftarkan DLL cuBLAS/cuDNN dari wheel pip `nvidia-*-cu12` (folder nvidia/*/bin).

    ctranslate2 memuat cuBLAS secara dinamis, jadi folder ditambahkan ke os.add_dll_directory dan PATH.
    No-op di non-Windows atau bila paket nvidia tidak terpasang.
    """
    global _nvidia_dirs_registered
    if os.name != "nt" or _nvidia_dirs_registered:
        return []
    _nvidia_dirs_registered = True
    try:
        import nvidia  # namespace package dari nvidia-cublas-cu12 / nvidia-cudnn-cu12
    except ImportError:
        return []
    dirs = sorted({str(p) for root in nvidia.__path__ for p in Path(root).glob("*/bin") if p.is_dir()})
    for d in dirs:
        os.add_dll_directory(d)
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
        log.info("DLL NVIDIA didaftarkan: %s", dirs)
    return dirs


class WhisperASR:
    """Wrapper faster-whisper. Dimuat/dilepas oleh GpuModelSlot."""

    def __init__(self, model_name: str, device: str, compute_type: str, download_root: Path,
                 language: str = "id", beam_size: int = 5) -> None:
        if device != "cpu":
            register_nvidia_dll_dirs()
        from faster_whisper import WhisperModel

        self.language = language
        self.beam_size = beam_size
        self.model_name = model_name
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type,
                                   download_root=str(download_root))
        log.info("Whisper '%s' dimuat (%s, %s)", model_name, device, compute_type)

    def transcribe(self, audio: np.ndarray, initial_prompt: str = "") -> list[Word]:
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            initial_prompt=initial_prompt or None,
            word_timestamps=True,
            beam_size=self.beam_size,
            vad_filter=True,                     # cegah halusinasi pada jeda panjang
            condition_on_previous_text=False,    # cegah loop pengulangan teks
            temperature=(0.0, 0.2, 0.4),
        )
        words: list[Word] = []
        for seg in segments:  # generator: decoding terjadi saat iterasi
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append(Word(start=float(w.start), end=float(w.end), text=text,
                                      probability=float(w.probability)))
        return words

    def close(self) -> None:
        self._model = None  # ctranslate2 melepas VRAM saat objek di-GC


@dataclass
class RawUtterance:
    start: float
    end: float
    text: str
    confidence: float


_SENT_END = (".", "?", "!")


def segment_words(words: list[Word], regions: list[SpeechRegion], gap_s: float = 0.6,
                  max_s: float = 15.0, punct_gap_s: float = 0.2) -> list[RawUtterance]:
    if not words:
        return []
    words = sorted(words, key=lambda w: w.start)
    starts = [r.start for r in regions]

    def region_of(w: Word) -> int:
        if not regions:
            return 0
        mid = 0.5 * (w.start + w.end)
        i = bisect.bisect_right(starts, mid) - 1
        if i >= 0 and regions[i].start <= mid <= regions[i].end:
            return i
        # di luar region -> region terdekat
        cand = [j for j in (i, i + 1) if 0 <= j < len(regions)]
        return min(cand, key=lambda j: min(abs(mid - regions[j].start), abs(mid - regions[j].end)))

    groups: list[list[Word]] = [[words[0]]]
    prev_region = region_of(words[0])
    for prev, cur in zip(words, words[1:]):
        reg = region_of(cur)
        gap = cur.start - prev.end
        dur = cur.end - groups[-1][0].start
        split = (
            reg != prev_region
            or gap > gap_s
            or (prev.text.endswith(_SENT_END) and gap > punct_gap_s)
            or (dur > max_s and prev.text.endswith(_SENT_END + (",",)))
            or dur > 1.5 * max_s
        )
        if split:
            groups.append([cur])
        else:
            groups[-1].append(cur)
        prev_region = reg

    out: list[RawUtterance] = []
    for g in groups:
        text = " ".join(w.text for w in g).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").strip()
        out.append(RawUtterance(start=g[0].start, end=max(w.end for w in g), text=text,
                                confidence=float(np.mean([w.probability for w in g]))))
    return out

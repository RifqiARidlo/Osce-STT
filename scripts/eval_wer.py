"""Evaluasi WER (Word Error Rate) Faster-Whisper pada rekaman mikrofon asli.

Jalur ASR sama persis dengan pipeline server: load_audio -> spectral gating (bila APP_DENOISE) ->
WhisperASR.transcribe dengan glossary prompt, memakai setelan Whisper dari `.env`.

Input: folder berisi pasangan file dengan nama sama:
    runtime/wer/sesi01.wav   rekaman (WAV apa pun; bukan 16 kHz akan di-resample)
    runtime/wer/sesi01.txt   transkrip referensi buatan manusia (ditulis apa adanya, angka pakai digit)

Output:
    runtime/wer/sesi01.hyp.txt          hasil Whisper (untuk dibandingkan manual)
    reports/wer/wer_results.json        WER per file + total + kesalahan tersering (tanpa teks penuh)

Contoh:
    python scripts/eval_wer.py                         # semua pasangan di runtime/wer/
    python scripts/eval_wer.py --dir runtime/wer --device cpu

Rekaman suara = data pribadi: simpan hanya di runtime/ (di-.gitignore), hapus setelah selesai.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.audio_io import SAMPLE_RATE, AudioLoadError, load_audio  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Normalisasi & WER (tanpa dependency tambahan)
# ---------------------------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_NUM = {w: str(i) for i, w in enumerate(
    "nol satu dua tiga empat lima enam tujuh delapan sembilan sepuluh".split())}


def normalize(text: str) -> list[str]:
    """Lowercase, samakan format angka tanda vital, buang tanda baca.

    "130/80", "130-80", "130x80", "130 per 80" -> "130 80";  "38,5" / "38 .5" -> "38_5";
    "98%" -> "98 persen";  "tiga" -> "3" (0-10, karena Whisper kadang menulis digit, kadang kata).
    Normalisasi diterapkan sama ke referensi dan hipotesis, jadi hanya salah dengar yang dihitung.
    """
    t = text.lower()
    t = re.sub(r"(\d)\s*[.,]\s*(\d)", r"\1_\2", t)     # desimal: "_" termasuk \w, tidak ikut terbuang
    t = re.sub(r"(\d)\s*(?:/|-|x|×|per)\s*(\d)", r"\1 \2", t)
    t = re.sub(r"(\d)\s*%", r"\1 persen", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)            # "500mg" -> "500 mg"
    return [_NUM.get(w, w) for w in _PUNCT.sub(" ", t).split()]


def align(ref: list[str], hyp: list[str]) -> dict:
    """Levenshtein tingkat kata. Return jumlah S/D/I dan daftar pasangan kesalahan."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    i, j, subs, dels, ins = n, m, [], [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dels.append(ref[i - 1])
            i -= 1
        else:
            ins.append(hyp[j - 1])
            j -= 1
    return {"n_ref": n, "S": len(subs), "D": len(dels), "I": len(ins),
            "subs": subs[::-1], "dels": dels[::-1], "ins": ins[::-1]}


def wer(res: dict) -> float:
    return (res["S"] + res["D"] + res["I"]) / max(res["n_ref"], 1)


# ---------------------------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------------------------
def read_any_wav(path: Path) -> np.ndarray:
    """load_audio server hanya menerima WAV 16 kHz PCM16 tanpa ffmpeg; di sini WAV lain di-resample."""
    try:
        return load_audio(path)
    except AudioLoadError:
        if path.suffix.lower() != ".wav":
            raise
    from scipy.signal import resample_poly

    with wave.open(str(path), "rb") as wf:
        sr, width, ch = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    dtype = {2: "<i2", 4: "<i4"}.get(width)
    if dtype is None:
        raise AudioLoadError(f"{path.name}: sample width {width * 8}-bit tidak didukung; ekspor ulang ke PCM16.")
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32) / float(2 ** (8 * width - 1))
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    g = np.gcd(sr, SAMPLE_RATE)
    return resample_poly(pcm, SAMPLE_RATE // g, sr // g).astype(np.float32)


def find_pairs(folder: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for ref in sorted(folder.glob("*.txt")):
        if ref.name.endswith(".hyp.txt"):
            continue
        audio = next((p for p in sorted(folder.glob(ref.stem + ".*"))
                      if p.suffix.lower() in (".wav", ".webm", ".mp3", ".m4a", ".ogg")), None)
        if audio is None:
            print(f"[peringatan] {ref.name}: tidak ada file audio bernama {ref.stem}.*, dilewati")
            continue
        pairs.append((audio, ref))
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("runtime/wer"))
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None, help="override APP_WHISPER_DEVICE")
    ap.add_argument("--no-denoise", action="store_true", help="lewati spectral gating (untuk perbandingan)")
    ap.add_argument("--out", type=Path, default=Path("reports/wer/wer_results.json"))
    args = ap.parse_args(argv)

    pairs = find_pairs(args.dir)
    if not pairs:
        print(f"Tidak ada pasangan <nama>.wav + <nama>.txt di {args.dir}. Lihat docs/TUTORIAL_TESTING.md §8.")
        return 1

    import os
    if args.device:
        os.environ["APP_WHISPER_DEVICE"] = args.device
    from app.config import get_settings
    from app.core.services import _model_factories
    from app.pipeline.stage1_preprocess import spectral_gate
    from app.pipeline.stage2_asr import load_glossary_prompt

    s = get_settings()
    denoise = s.denoise and not args.no_denoise
    prompt = load_glossary_prompt(s.glossary_path)
    t0 = time.perf_counter()
    asr = _model_factories(s)["asr"]()
    print(f"Whisper {s.whisper_model} ({os.environ.get('APP_WHISPER_DEVICE', s.whisper_device)}) dimuat "
          f"{time.perf_counter() - t0:.1f} s | denoise={denoise} | {len(pairs)} file")

    results, total = [], {"n_ref": 0, "S": 0, "D": 0, "I": 0}
    sub_counter: Counter[str] = Counter()
    del_counter: Counter[str] = Counter()
    try:
        for audio_path, ref_path in pairs:
            audio = read_any_wav(audio_path)
            clean = spectral_gate(audio) if denoise else audio
            t = time.perf_counter()
            hyp_text = " ".join(w.text for w in asr.transcribe(clean, prompt))
            asr_s = time.perf_counter() - t
            ref_path.with_name(ref_path.stem + ".hyp.txt").write_text(hyp_text + "\n", encoding="utf-8")
            res = align(normalize(ref_path.read_text(encoding="utf-8")), normalize(hyp_text))
            for k in total:
                total[k] += res[k]
            sub_counter.update(f"{r} -> {h}" for r, h in res["subs"])
            del_counter.update(res["dels"])
            dur = audio.size / SAMPLE_RATE
            row = {"file": audio_path.name, "durasi_s": round(dur, 1), "asr_s": round(asr_s, 2),
                   "rtf": round(asr_s / dur, 3), "n_kata_ref": res["n_ref"], "S": res["S"], "D": res["D"],
                   "I": res["I"], "wer": round(wer(res), 4)}
            results.append(row)
            print(f"{row['file']:<28} {dur:6.1f} s  WER {row['wer']:.1%}  (S {res['S']} D {res['D']} I {res['I']}"
                  f" / {res['n_ref']} kata)  RTF {row['rtf']}")
    finally:
        asr.close()

    summary = {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "whisper": {"model": s.whisper_model, "device": os.environ.get("APP_WHISPER_DEVICE", s.whisper_device),
                    "compute_type": s.whisper_compute_type, "beam": s.whisper_beam_size},
        "denoise": denoise,
        "per_file": results,
        "total": {**total, "wer": round(wer(total), 4)},
        "substitusi_tersering": sub_counter.most_common(20),
        "kata_hilang_tersering": del_counter.most_common(20),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTOTAL WER {wer(total):.1%} ({total['S']} S + {total['D']} D + {total['I']} I / {total['n_ref']} kata)")
    print("Substitusi tersering:", ", ".join(f"{k} ({v})" for k, v in sub_counter.most_common(8)) or "-")
    print(f"-> {args.out} | hipotesis per file: {args.dir}/<nama>.hyp.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())

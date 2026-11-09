"""Embedding speaker ECAPA-TDNN (opsional; butuh torch + speechbrain).

Jalan di CPU (model ~20 MB, cukup cepat untuk segmen pendek) supaya tidak berebut VRAM
dengan Whisper/LLM. Bila dependency tidak tersedia, `load_embedder()` mengembalikan None
dan pipeline otomatis memakai fitur handcrafted saja.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class EcapaEmbedder:
    dim = 192

    def __init__(self, savedir: Path) -> None:
        import torch  # noqa: F401
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier  # type: ignore[no-redef]
        self._torch = __import__("torch")
        self._model = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, savedir=str(savedir), run_opts={"device": "cpu"}
        )

    def embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if audio.size < sr // 4:
            audio = np.pad(audio, (0, sr // 4 - audio.size))
        with self._torch.no_grad():
            wav = self._torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb = self._model.encode_batch(wav).squeeze().cpu().numpy()
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 0 else emb.astype(np.float32)

    def embed_many(self, audio: np.ndarray, spans: list[tuple[float, float]], sr: int = 16000) -> np.ndarray:
        out = np.zeros((len(spans), self.dim), dtype=np.float32)
        for i, (start, end) in enumerate(spans):
            s, e = int(start * sr), int(end * sr)
            out[i] = self.embed(audio[s:e], sr)
        return out


def load_embedder(enabled: bool, models_dir: Path) -> EcapaEmbedder | None:
    if not enabled:
        return None
    try:
        return EcapaEmbedder(models_dir / "ecapa")
    except Exception as exc:  # ImportError, network error saat download pertama, dsb.
        log.warning("ECAPA-TDNN tidak bisa dimuat (%s). Lanjut tanpa embedding.", exc)
        return None

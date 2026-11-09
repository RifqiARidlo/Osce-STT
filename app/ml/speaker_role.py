"""[CUSTOM ML MODULE 1] Speaker Role Classifier (DOKTER vs PASIEN).

Prinsip desain (koreksi terhadap spesifikasi awal):
Fitur akustik absolut (mean MFCC, F0, dst.) mengkodekan IDENTITAS speaker dan karakter mikrofon,
bukan PERAN. Classifier yang dilatih di atas fitur absolut akan menghafal suara dokter di data latih
dan gagal pada dokter baru. Maka fitur di sini dibuat RELATIF terhadap anchor dokter yang di-enroll:

  1. |z| = |(v - anchor_mean) / anchor_scale| per dimensi  -> "seberapa jauh dari profil dokter"
  2. ringkasan jarak (mean|z|, rms z)
  3. jarak relatif-sesi (dikurangi median sesi) + persentil rank -> kompensasi domain shift
     mikrofon/ruangan, karena dokter & pasien berbagi kanal yang sama dalam satu sesi
  4. clustering 2-kelompok dalam sesi (KMeans) -> apakah segmen berada di cluster yang paling
     dekat dengan anchor
  5. (opsional) cosine similarity ECAPA-TDNN terhadap anchor + versi relatif-sesinya

Dengan begitu tugas model menjadi "speaker verification terhadap anchor", yang dapat
digeneralisasi ke dokter yang belum pernah dilihat saat training.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.ml.audio_features import DEFAULT_CONFIG, FeatureConfig, extract_segment_features, feature_dim, feature_names

log = logging.getLogger(__name__)

ModelKind = Literal["svm", "rf", "mlp", "logreg"]
Z_CLIP = 10.0
LABELS = ("DOKTER", "PASIEN")


# ---------------------------------------------------------------------------------------------
# Anchor dokter
# ---------------------------------------------------------------------------------------------
@dataclass
class DoctorAnchor:
    mean: np.ndarray
    scale: np.ndarray
    embedding: np.ndarray | None
    n_chunks: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, n_chunks=self.n_chunks,
                 embedding=self.embedding if self.embedding is not None else np.zeros(0, np.float32))

    @classmethod
    def load(cls, path: Path) -> "DoctorAnchor":
        with np.load(path) as data:
            emb = data["embedding"]
            return cls(mean=data["mean"], scale=data["scale"],
                       embedding=emb if emb.size else None, n_chunks=int(data["n_chunks"]))


def _group_slices(cfg: FeatureConfig) -> list[slice]:
    n = cfg.n_mfcc
    groups = [slice(i * n, (i + 1) * n) for i in range(4)]
    groups += [slice(4 * n + j, 4 * n + j + 1) for j in range(9)]
    return groups


def build_anchor(audio: np.ndarray, embedder: Any | None = None, cfg: FeatureConfig = DEFAULT_CONFIG,
                 chunk_s: float = 2.0, hop_s: float = 1.0) -> DoctorAnchor:
    """Bangun profil akustik dokter dari rekaman enrollment (idealnya 10-20 detik ucapan)."""
    sr = cfg.sr
    chunk, hop = int(chunk_s * sr), int(hop_s * sr)
    if audio.size < chunk:
        raise ValueError("Audio enrollment terlalu pendek (minimal 2 detik ucapan, ideal 10-15 detik).")
    vecs = np.stack([extract_segment_features(audio[s:s + chunk], cfg)
                     for s in range(0, audio.size - chunk + 1, hop)])
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    scale = np.empty_like(std)
    for sl in _group_slices(cfg):
        group_std = std[sl]
        if sl.stop - sl.start > 1:
            floor = max(0.5 * float(np.median(group_std)), 1e-3)
        else:
            floor = max(0.05 * float(abs(mean[sl][0])), 1e-3)
        scale[sl] = np.maximum(group_std, floor)
    emb = embedder.embed(audio, sr) if embedder is not None else None
    return DoctorAnchor(mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                        embedding=emb, n_chunks=int(vecs.shape[0]))


# ---------------------------------------------------------------------------------------------
# Feature builder (relatif anchor + relatif sesi)
# ---------------------------------------------------------------------------------------------
SESSION_FEATURES = ["dist_mean_abs_z", "dist_rms_z", "dist_rel_session", "dist_rank_pct",
                    "in_anchor_cluster", "cluster_margin"]
EMB_FEATURES = ["ecapa_cos", "ecapa_cos_rel_session"]


def role_feature_names(uses_embedding: bool, cfg: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    names = [f"absz_{n}" for n in feature_names(cfg)] + SESSION_FEATURES
    return names + EMB_FEATURES if uses_embedding else names


def _rank_pct(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.full(values.shape, 0.5)
    order = values.argsort().argsort()
    return order / (values.size - 1)


def build_role_features(vectors: np.ndarray, anchor: DoctorAnchor,
                        embeddings: np.ndarray | None = None, uses_embedding: bool = False) -> np.ndarray:
    """vectors: (n, D) fitur handcrafted semua utterance dalam SATU sesi."""
    vectors = np.atleast_2d(vectors).astype(np.float64)
    n = vectors.shape[0]
    z = np.clip((vectors - anchor.mean) / anchor.scale, -Z_CLIP, Z_CLIP)
    absz = np.abs(z)
    mean_abs = absz.mean(axis=1)
    rms = np.sqrt((z ** 2).mean(axis=1))
    rel = mean_abs - np.median(mean_abs)
    rank = _rank_pct(mean_abs)

    in_cluster = np.full(n, 0.5)
    margin = np.zeros(n)
    if n >= 4:
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(z)
        if len(set(km.labels_)) == 2:
            doc_cluster = int(np.argmin(np.linalg.norm(km.cluster_centers_, axis=1)))
            d_doc = np.linalg.norm(z - km.cluster_centers_[doc_cluster], axis=1)
            d_oth = np.linalg.norm(z - km.cluster_centers_[1 - doc_cluster], axis=1)
            in_cluster = (km.labels_ == doc_cluster).astype(np.float64)
            margin = (d_oth - d_doc) / np.maximum(d_oth + d_doc, 1e-9)

    feats = [absz, mean_abs[:, None], rms[:, None], rel[:, None], rank[:, None],
             in_cluster[:, None], margin[:, None]]
    if uses_embedding:
        if embeddings is None or anchor.embedding is None:
            raise ValueError("Model membutuhkan embedding ECAPA tetapi embedding/anchor embedding tidak tersedia.")
        cos = embeddings @ anchor.embedding / np.maximum(
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(anchor.embedding), 1e-9)
        feats += [cos[:, None], (cos - np.median(cos))[:, None]]
    return np.hstack(feats).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------------------------
def make_estimator(kind: ModelKind, random_state: int = 42):
    if kind == "svm":
        # SVC(probability=True) deprecated di sklearn >= 1.9 -> kalibrasi Platt eksplisit.
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", CalibratedClassifierCV(SVC(C=1.0, gamma="scale", class_weight="balanced",
                                                            random_state=random_state),
                                                        method="sigmoid", cv=3, ensemble=False))])
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                      n_jobs=-1, random_state=random_state)
    if kind == "mlp":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                               max_iter=600, random_state=random_state))])
    if kind == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    raise ValueError(f"Model tidak dikenal: {kind}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SpeakerRoleClassifier:
    """Wrapper inference. Tanpa model terlatih -> heuristik berbasis jarak anchor (ditandai)."""

    def __init__(self, bundle: dict | None = None) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path) -> "SpeakerRoleClassifier":
        if path.is_file():
            bundle = joblib.load(path)
            log.info("Speaker role model dimuat: %s (%s)", path, bundle.get("model_kind"))
            return cls(bundle)
        log.warning("Speaker role model belum ada di %s -> pakai heuristik anchor.", path)
        return cls(None)

    @staticmethod
    def save_bundle(path: Path, model: Any, kind: str, uses_embedding: bool,
                    metrics: dict | None = None, cfg: FeatureConfig = DEFAULT_CONFIG) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model, "model_kind": kind, "uses_embedding": uses_embedding,
            "feature_names": role_feature_names(uses_embedding, cfg), "feature_config": cfg.to_dict(),
            "labels": list(LABELS), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics_summary": metrics or {},
        }, path)

    @property
    def is_trained(self) -> bool:
        return self.bundle is not None

    @property
    def uses_embedding(self) -> bool:
        return bool(self.bundle and self.bundle["uses_embedding"])

    @property
    def description(self) -> str:
        if not self.bundle:
            return "heuristic-anchor-distance"
        return f"{self.bundle['model_kind']}@{self.bundle['trained_at']}"

    def predict(self, vectors: np.ndarray, anchor: DoctorAnchor,
                embeddings: np.ndarray | None = None) -> list[tuple[str, float, str]]:
        """Return list (label, confidence, source)."""
        if len(vectors) == 0:
            return []
        if self.bundle is not None:
            if self.uses_embedding and (embeddings is None or anchor.embedding is None):
                log.warning("Model dilatih dengan ECAPA, tapi embedding tidak tersedia -> heuristik.")
            else:
                X = build_role_features(vectors, anchor, embeddings, self.uses_embedding)
                model = self.bundle["model"]
                proba = model.predict_proba(X)
                classes = list(model.classes_)
                out = []
                for row in proba:
                    k = int(np.argmax(row))
                    out.append((str(classes[k]), float(row[k]), "model"))
                return out
        return self._heuristic(vectors, anchor, embeddings)

    @staticmethod
    def _heuristic(vectors: np.ndarray, anchor: DoctorAnchor,
                   embeddings: np.ndarray | None) -> list[tuple[str, float, str]]:
        X = build_role_features(vectors, anchor, None, False)
        names = SESSION_FEATURES
        base = X.shape[1] - len(names)
        rel = X[:, base + names.index("dist_rel_session")]
        in_cluster = X[:, base + names.index("in_anchor_cluster")]
        mean_abs = X[:, base + names.index("dist_mean_abs_z")]
        mad = float(np.median(np.abs(mean_abs - np.median(mean_abs)))) + 1e-6
        p_doc = _sigmoid(-rel / mad)
        if len(vectors) >= 4:
            p_doc = 0.5 * p_doc + 0.5 * np.where(in_cluster > 0.5, 0.85, 0.15)
        if embeddings is not None and anchor.embedding is not None:
            cos = embeddings @ anchor.embedding
            p_doc = 0.5 * p_doc + 0.5 * _sigmoid((cos - np.median(cos)) / 0.05)
        return [("DOKTER", float(p), "heuristic") if p >= 0.5 else ("PASIEN", float(1 - p), "heuristic")
                for p in p_doc]


def default_feature_dim() -> int:
    return feature_dim(DEFAULT_CONFIG)

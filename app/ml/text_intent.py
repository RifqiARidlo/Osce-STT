"""[CUSTOM ML MODULE 2] Clinical Utterance Classifier (S/O/A/P/CHITCHAT).

Fitur:
  - TF-IDF kata n-gram (1,2)               -> kata kunci klinis
  - TF-IDF karakter n-gram (3,5) char_wb   -> robust terhadap salah eja / error ASR
  - One-hot peran pembicara (DOKTER/PASIEN) -> prior kuat (mis. O & A hampir selalu dari dokter)
  - (opsional) sentence embedding IndoBERT  -> via sentence-transformers bila terpasang

Classifier: Logistic Regression (default, probabilistik & interpretable) atau LinearSVC terkalibrasi.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from app.schemas.pipeline import INTENT_LABELS

log = logging.getLogger(__name__)

IntentModelKind = Literal["logreg", "linearsvc"]
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s/.,%-]", " ", text)
    return _WS.sub(" ", text).strip()


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Transformer sklearn untuk sentence embedding (lazy load, tidak ikut di-pickle)."""

    def __init__(self, model_name: str = "indobenchmark/indobert-base-p1", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

    def _model(self):
        if getattr(self, "_st", None) is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Mode embedding butuh `pip install sentence-transformers`.") from exc
            self._st = SentenceTransformer(self.model_name, device="cpu")
        return self._st

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        texts = list(X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X)
        return self._model().encode(texts, batch_size=self.batch_size, normalize_embeddings=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_st", None)
        return state


ALL_FEATURES = ("word", "char", "role")


def build_pipeline(kind: IntentModelKind = "logreg", use_embedding: bool = False,
                   embedding_model: str = "indobenchmark/indobert-base-p1",
                   features: tuple[str, ...] = ALL_FEATURES) -> Pipeline:
    available: dict[str, tuple[str, Any, Any]] = {
        "word": ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50000), "text"),
        "char": ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                         max_features=80000), "text"),
        "role": ("role", OneHotEncoder(handle_unknown="ignore"), ["role"]),
    }
    unknown = set(features) - set(available)
    if unknown:
        raise ValueError(f"Fitur tidak dikenal: {unknown}")
    transformers: list[tuple[str, Any, Any]] = [available[f] for f in features]
    if use_embedding:
        transformers.append(("emb", SentenceEmbedder(embedding_model), ["text"]))
    features = ColumnTransformer(transformers)
    if kind == "logreg":
        clf: Any = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    elif kind == "linearsvc":
        clf = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)
    else:
        raise ValueError(f"Model tidak dikenal: {kind}")
    return Pipeline([("features", features), ("clf", clf)])


def to_frame(texts: Sequence[str], roles: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({"text": [normalize_text(t) for t in texts], "role": list(roles)})


# --------------------------- fallback rule-based ---------------------------------------------
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("O_PEMERIKSAAN", re.compile(r"\b(tensi|tekanan darah|nadi|suhu|saturasi|spo2|respirasi|frekuensi napas|"
                                 r"auskultasi|ronkhi|wheezing|vesikuler|perkusi|palpasi|nyeri tekan|"
                                 r"mmhg|kali per menit|derajat|hasil pemeriksaan|pemeriksaan fisik)\b")),
    ("A_DIAGNOSIS", re.compile(r"\b(diagnosis|diagnosa|kemungkinan|curiga|suspek|mengarah ke|"
                               r"kesimpulan|terkena|mengalami (infeksi|radang)|didiagnosis)\b")),
    ("P_EDUKASI_TERAPI", re.compile(r"\b(obat|minum|resep|tablet|kapsul|sirup|mg|dosis|kontrol|"
                                    r"istirahat|banyak minum|hindari|jangan|dirujuk|rujuk|cek lab|"
                                    r"rontgen|kali sehari|sesudah makan|sebelum makan)\b")),
    ("S_KELUHAN", re.compile(r"\b(sakit|nyeri|demam|panas|batuk|pilek|sesak|pusing|mual|muntah|diare|"
                             r"lemas|gatal|bengkak|sejak|berapa lama|keluhan|kambuh|riwayat|"
                             r"dahak|mencret|nggliyeng|kemeng|mules|meriang)\b")),
]


def rule_based_intent(text: str, role: str) -> tuple[str, float]:
    t = normalize_text(text)
    for label, pattern in _RULES:
        if pattern.search(t):
            if label in ("O_PEMERIKSAAN", "A_DIAGNOSIS") and role == "PASIEN":
                continue
            return label, 0.5
    return "CHITCHAT", 0.4


class IntentClassifier:
    def __init__(self, bundle: dict | None = None) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        if path.is_file():
            bundle = joblib.load(path)
            log.info("Intent model dimuat: %s (%s)", path, bundle.get("model_kind"))
            return cls(bundle)
        log.warning("Intent model belum ada di %s -> pakai rule-based fallback.", path)
        return cls(None)

    @staticmethod
    def save_bundle(path: Path, pipeline: Pipeline, kind: str, use_embedding: bool,
                    metrics: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": pipeline, "model_kind": kind, "use_embedding": use_embedding,
            "labels": list(INTENT_LABELS),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics_summary": metrics or {},
        }, path)

    @property
    def is_trained(self) -> bool:
        return self.bundle is not None

    @property
    def description(self) -> str:
        if not self.bundle:
            return "rule-based-fallback"
        return f"{self.bundle['model_kind']}@{self.bundle['trained_at']}"

    def predict(self, texts: Sequence[str], roles: Sequence[str]) -> list[tuple[str, float, dict[str, float], str]]:
        """Return list (label, confidence, probs_per_class, source)."""
        if not texts:
            return []
        if self.bundle is None:
            out = []
            for text, role in zip(texts, roles):
                label, conf = rule_based_intent(text, role)
                out.append((label, conf, {label: conf}, "heuristic"))
            return out
        pipe: Pipeline = self.bundle["pipeline"]
        proba = pipe.predict_proba(to_frame(texts, roles))
        classes = [str(c) for c in pipe.classes_]
        out = []
        for row in proba:
            k = int(np.argmax(row))
            out.append((classes[k], float(row[k]), {c: round(float(p), 4) for c, p in zip(classes, row)}, "model"))
        return out

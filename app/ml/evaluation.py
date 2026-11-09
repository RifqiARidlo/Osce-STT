"""Helper metrik untuk Academic ML Evaluation Panel."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

log = logging.getLogger(__name__)


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    labels = list(labels)
    present = [lab for lab in labels if lab in set(y_true) | set(y_pred)]
    report = classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=present)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)),
        "labels": present,
        "per_class": {lab: {k: float(report[lab][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for lab in present},
        "confusion_matrix": cm.astype(int).tolist(),
    }


def summarize_cv(scores: Sequence[float]) -> dict:
    arr = np.asarray(scores, dtype=float)
    return {"scores": [round(float(s), 4) for s in arr], "mean": float(arr.mean()), "std": float(arr.std())}


def save_metrics(out_dir: Path, metrics: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _try_plot_confusion(out_dir, metrics.get("test", {}))
    return path


def _try_plot_confusion(out_dir: Path, test_metrics: dict) -> None:
    """PNG confusion matrix untuk slide presentasi (opsional; butuh matplotlib)."""
    if "confusion_matrix" not in test_metrics:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.asarray(test_metrics["confusion_matrix"])
    labels = test_metrics["labels"]
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

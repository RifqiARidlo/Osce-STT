"""Training & evaluasi Custom ML Module 2 (Clinical Utterance Classifier).

Protokol evaluasi:
  - Split train/test BERBASIS GRUP (template_id untuk data sintetis, job_id untuk data verifikasi)
    -> mencegah kebocoran kalimat hampir-identik dari template yang sama.
  - Opsi --test-source human_verified: seluruh data verifikasi manusia dipakai sebagai test set
    (protokol paling jujur: latih di sintetis, uji di percakapan nyata).
  - GroupKFold CV di train set, perbandingan dengan baseline (majority & rule-based), ablation fitur.
  - Model final di-refit pada seluruh data (default) setelah evaluasi.

Contoh:
    python scripts/train_text_intent.py --data data/synthetic/text_intent.jsonl runtime/training_data/text_intent.jsonl
    python scripts/train_text_intent.py --data data/synthetic/*.jsonl --test-source human_verified --data runtime/training_data/text_intent.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.evaluation import compute_metrics, save_metrics, summarize_cv  # noqa: E402
from app.ml.text_intent import IntentClassifier, build_pipeline, rule_based_intent, to_frame  # noqa: E402
from app.schemas.pipeline import INTENT_LABELS  # noqa: E402


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if not path.is_file():
            print(f"[peringatan] {path} tidak ada, dilewati")
            continue
        for i, line in enumerate(path.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("label") not in INTENT_LABELS or r.get("role") not in ("DOKTER", "PASIEN") or not r.get("text"):
                continue
            r["group"] = r.get("template_id") or r.get("job_id") or f"{path.stem}_{i}"
            rows.append(r)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", action="extend", required=True)
    ap.add_argument("--model", choices=["logreg", "linearsvc"], default="logreg")
    ap.add_argument("--embedding", action="store_true", help="tambahkan sentence embedding IndoBERT")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--test-source", default="", help="mis. human_verified -> semua baris sumber ini jadi test set")
    ap.add_argument("--no-refit", action="store_true", help="simpan model dari train split saja")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models/text_intent"))
    args = ap.parse_args(argv)

    rows = load_rows(args.data)
    if len(rows) < 50:
        print(f"Data terlalu sedikit ({len(rows)} baris). Jalankan generate_synthetic_text.py dulu.")
        return 1
    groups = np.array([r["group"] for r in rows])
    y = np.array([r["label"] for r in rows])
    X = to_frame([r["text"] for r in rows], [r["role"] for r in rows])

    if args.test_source:
        test_mask = np.array([r.get("source") == args.test_source for r in rows])
        if test_mask.sum() < 20 or (~test_mask).sum() < 50:
            print(f"Data sumber '{args.test_source}' = {test_mask.sum()} baris; terlalu sedikit untuk test set.")
            return 1
        train_idx, test_idx = np.where(~test_mask)[0], np.where(test_mask)[0]
        split_note = f"Test set = seluruh data '{args.test_source}' ({len(test_idx)} baris); train = sumber lain."
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        split_note = (f"Split berbasis grup ({len(set(groups[test_idx]))} grup/template di test tidak pernah dilihat "
                      "saat training). Bila data mayoritas sintetis, skor ini batas atas optimistis.")
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    print(f"train={len(train_idx)} test={len(test_idx)} | {split_note}")

    n_groups_train = len(set(groups[train_idx]))
    cv = None
    if n_groups_train >= 5:
        scores = cross_val_score(build_pipeline(args.model, args.embedding), Xtr, ytr, groups=groups[train_idx],
                                 cv=GroupKFold(n_splits=5), scoring="f1_macro", n_jobs=1)
        cv = summarize_cv(scores)
        print(f"GroupKFold-5 macro F1: {cv['mean']:.3f} ± {cv['std']:.3f}")

    pipe = build_pipeline(args.model, args.embedding).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    test_metrics = compute_metrics(yte, pred, INTENT_LABELS)
    print(f"TEST accuracy={test_metrics['accuracy']:.3f} macro_f1={test_metrics['macro_f1']:.3f}")

    comparison: dict[str, dict] = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    comparison["baseline_majority"] = compute_metrics(yte, dummy.predict(Xte), INTENT_LABELS)
    rule_pred = [rule_based_intent(t, r)[0] for t, r in zip(Xte["text"], Xte["role"])]
    comparison["baseline_rule_keyword"] = compute_metrics(yte, rule_pred, INTENT_LABELS)
    other = "linearsvc" if args.model == "logreg" else "logreg"
    comparison[other] = compute_metrics(yte, build_pipeline(other).fit(Xtr, ytr).predict(Xte), INTENT_LABELS)
    comparison[f"{args.model} (dipilih)"] = test_metrics
    comparison = {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]} for k, v in comparison.items()}

    ablation: dict[str, float] = {}
    for name, feats in {"word+char+role": ("word", "char", "role"), "word+char (tanpa role)": ("word", "char"),
                        "word saja": ("word",), "char saja": ("char",)}.items():
        p = build_pipeline(args.model, features=feats).fit(Xtr, ytr).predict(Xte)
        ablation[name] = compute_metrics(yte, p, INTENT_LABELS)["macro_f1"]
    print("Ablation:", {k: round(v, 3) for k, v in ablation.items()})

    final = pipe if args.no_refit else build_pipeline(args.model, args.embedding).fit(X, y)
    sources: dict[str, int] = {}
    for r in rows:
        sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1
    metrics = {
        "model_kind": args.model + ("+indobert" if args.embedding else ""),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_summary": ", ".join(f"{k}={v}" for k, v in sources.items()),
        "split_note": split_note + ("" if args.no_refit else " Model tersimpan di-refit pada seluruh data."),
        "test": test_metrics, "cv": cv, "comparison": comparison, "ablation": ablation,
    }
    IntentClassifier.save_bundle(args.out / "model.joblib", final, metrics["model_kind"], args.embedding,
                                 {"test_macro_f1": test_metrics["macro_f1"]})
    save_metrics(args.out, metrics)
    print(f"Model -> {args.out / 'model.joblib'} | metrics -> {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Training & evaluasi Custom ML Module 1 (Speaker Role Classifier DOKTER vs PASIEN).

Sumber data (boleh digabung):
  --data      : runtime/training_data/speaker_role.jsonl (otomatis terisi saat sesi diverifikasi di UI)
  --manifest  : JSONL rekaman berlabel manual, satu baris per sesi:
                {"audio": "rekaman/s01.wav", "doctor_id": "<id enrollment>",
                 "segments": [{"start": 0.5, "end": 3.2, "role": "DOKTER"}, ...]}
                -> mendukung augmentasi audio (--augment N salinan per sesi).

Protokol evaluasi:
  - Fitur dibangun per SESI (fitur relatif-sesi butuh seluruh utterance sesi tersebut).
  - Split berbasis DOKTER bila >= 3 dokter (menguji generalisasi ke dokter yang belum pernah dilihat),
    selain itu berbasis SESI. Salinan augmentasi selalu berada di grup yang sama dengan aslinya.
  - Perbandingan SVM / RF / MLP / LogReg + baseline heuristik anchor, ablation fitur relatif vs absolut.

Contoh:
    python scripts/train_speaker_role.py --data runtime/training_data/speaker_role.jsonl --model svm
    python scripts/train_speaker_role.py --manifest data/labeled/manifest.jsonl --augment 2 --model rf
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.audio_io import SAMPLE_RATE, load_audio  # noqa: E402
from app.ml.audio_features import extract_many  # noqa: E402
from app.ml.augment import random_augment  # noqa: E402
from app.ml.evaluation import compute_metrics, save_metrics, summarize_cv  # noqa: E402
from app.ml.speaker_role import (LABELS, DoctorAnchor, SpeakerRoleClassifier, build_role_features,  # noqa: E402
                                 make_estimator)


@dataclass
class Session:
    session_id: str
    doctor_id: str
    group: str
    vectors: np.ndarray
    embeddings: np.ndarray | None
    labels: list[str]


def load_active_learning(path: Path) -> list[Session]:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("label") in LABELS:
                by_session[r["session_id"]].append(r)
    sessions = []
    for sid, rows in by_session.items():
        rows.sort(key=lambda r: r["utt_id"])
        embs = [r.get("embedding") for r in rows]
        sessions.append(Session(
            session_id=sid, doctor_id=rows[0]["doctor_id"], group=sid,
            vectors=np.asarray([r["vector"] for r in rows], dtype=np.float32),
            embeddings=np.asarray(embs, dtype=np.float32) if all(e is not None for e in embs) else None,
            labels=[r["label"] for r in rows]))
    return sessions


def load_manifest(path: Path, n_augment: int, seed: int) -> list[Session]:
    rng = np.random.default_rng(seed)
    sessions = []
    for i, line in enumerate(path.open(encoding="utf-8")):
        if not line.strip():
            continue
        item = json.loads(line)
        audio_path = (path.parent / item["audio"]).resolve() if not Path(item["audio"]).is_absolute() else Path(item["audio"])
        audio = load_audio(audio_path)  # mentah, konsisten dengan pipeline inference
        segs = [s for s in item["segments"] if s["role"] in LABELS and s["end"] > s["start"]]
        spans = [(float(s["start"]), float(s["end"])) for s in segs]
        labels = [s["role"] for s in segs]
        sid = item.get("session_id", f"{path.stem}_{i}")
        sessions.append(Session(sid, item["doctor_id"], sid, extract_many(audio, spans), None, labels))
        for k in range(n_augment):
            aug_vectors = []
            for start, end in spans:
                seg = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
                aug = random_augment(seg, rng)
                aug_vectors.append(extract_many(aug, [(0.0, aug.size / SAMPLE_RATE)])[0])
            sessions.append(Session(f"{sid}_aug{k}", item["doctor_id"], sid, np.stack(aug_vectors), None, labels))
        print(f"  manifest sesi {sid}: {len(spans)} segmen (+{n_augment} augmentasi)")
    return sessions


def build_matrix(sessions: list[Session], anchors: dict[str, DoctorAnchor], mode: str, use_emb: bool):
    X, y, groups, doctors = [], [], [], []
    for s in sessions:
        if mode == "relative":
            feats = build_role_features(s.vectors, anchors[s.doctor_id], s.embeddings, use_emb)
        else:  # absolute: fitur akustik mentah (pembanding / ablation)
            feats = s.vectors
        X.append(feats)
        y += s.labels
        groups += [s.group] * len(s.labels)
        doctors += [s.doctor_id] * len(s.labels)
    return np.vstack(X), np.asarray(y), np.asarray(groups), np.asarray(doctors)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="*", default=[])
    ap.add_argument("--manifest", type=Path, nargs="*", default=[])
    ap.add_argument("--anchors-dir", type=Path, default=Path("runtime/anchors"))
    ap.add_argument("--augment", type=int, default=0)
    ap.add_argument("--model", choices=["svm", "rf", "mlp", "logreg"], default="svm")
    ap.add_argument("--use-embedding", action="store_true", help="pakai fitur ECAPA (harus tersedia di semua data)")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("models/speaker_role"))
    args = ap.parse_args(argv)

    sessions: list[Session] = []
    for p in args.data:
        if p.is_file():
            sessions += load_active_learning(p)
    for p in args.manifest:
        sessions += load_manifest(p, args.augment, args.seed)
    anchors: dict[str, DoctorAnchor] = {}
    for s in sessions:
        if s.doctor_id not in anchors:
            path = args.anchors_dir / f"{s.doctor_id}.npz"
            if not path.is_file():
                print(f"Anchor dokter {s.doctor_id} tidak ditemukan di {path}")
                return 1
            anchors[s.doctor_id] = DoctorAnchor.load(path)
    if args.use_embedding and any(s.embeddings is None for s in sessions):
        print("--use-embedding diminta tapi sebagian sesi tidak punya embedding ECAPA.")
        return 1

    X, y, groups, doctors = build_matrix(sessions, anchors, "relative", args.use_embedding)
    n_doctors = len(set(doctors))
    print(f"{len(sessions)} sesi, {len(y)} segmen, {n_doctors} dokter, distribusi: "
          f"{ {lab: int((y == lab).sum()) for lab in LABELS} }")
    if len(set(groups)) < 3 or len(set(y)) < 2:
        print("Data belum cukup: minimal 3 sesi berbeda dengan kedua label.")
        return 1

    split_groups = doctors if n_doctors >= 3 else groups
    split_note = ("Split berbasis DOKTER: dokter di test set tidak pernah dilihat saat training."
                  if n_doctors >= 3 else
                  "Split berbasis SESI (dokter < 3): generalisasi ke dokter baru BELUM teruji.")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, y, split_groups))
    print(split_note, f"train={len(tr)} test={len(te)}")

    cv = None
    n_cv = min(5, len(set(split_groups[tr])))
    if n_cv >= 3:
        scores = []
        for a, b in GroupKFold(n_splits=n_cv).split(X[tr], y[tr], split_groups[tr]):
            if len(set(y[tr][a])) < 2:
                continue
            m = make_estimator(args.model, args.seed).fit(X[tr][a], y[tr][a])
            scores.append(compute_metrics(y[tr][b], m.predict(X[tr][b]), LABELS)["macro_f1"])
        cv = summarize_cv(scores) if scores else None

    model = make_estimator(args.model, args.seed).fit(X[tr], y[tr])
    test_metrics = compute_metrics(y[te], model.predict(X[te]), LABELS)
    print(f"TEST accuracy={test_metrics['accuracy']:.3f} macro_f1={test_metrics['macro_f1']:.3f}")

    comparison: dict[str, dict] = {}
    for kind in ("svm", "rf", "mlp", "logreg"):
        m = model if kind == args.model else make_estimator(kind, args.seed).fit(X[tr], y[tr])
        res = compute_metrics(y[te], m.predict(X[te]), LABELS)
        comparison[kind + (" (dipilih)" if kind == args.model else "")] = {"accuracy": res["accuracy"], "macro_f1": res["macro_f1"]}
    # baseline heuristik (tanpa training) per sesi test
    test_groups = set(split_groups[te])
    h_true, h_pred = [], []
    heuristic = SpeakerRoleClassifier(None)
    for s in sessions:
        key = s.doctor_id if n_doctors >= 3 else s.group
        if key in test_groups:
            h_pred += [p[0] for p in heuristic.predict(s.vectors, anchors[s.doctor_id], s.embeddings)]
            h_true += s.labels
    hres = compute_metrics(h_true, h_pred, LABELS)
    comparison["baseline_heuristik_anchor"] = {"accuracy": hres["accuracy"], "macro_f1": hres["macro_f1"]}

    # ablation: fitur absolut (mentah) vs relatif-anchor
    Xa, _, _, _ = build_matrix(sessions, anchors, "absolute", False)
    abs_model = Pipeline([("s", StandardScaler()), ("c", SVC(class_weight="balanced"))]).fit(Xa[tr], y[tr])
    ablation = {
        "relatif anchor+sesi (dipakai)": test_metrics["macro_f1"],
        "absolut (MFCC/F0 mentah, SVM)": compute_metrics(y[te], abs_model.predict(Xa[te]), LABELS)["macro_f1"],
    }
    print("Perbandingan:", {k: round(v["macro_f1"], 3) for k, v in comparison.items()})
    print("Ablation:", {k: round(v, 3) for k, v in ablation.items()})

    final = make_estimator(args.model, args.seed).fit(X, y)
    metrics = {
        "model_kind": args.model + ("+ecapa" if args.use_embedding else ""),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_summary": f"{len(sessions)} sesi, {n_doctors} dokter, {len(y)} segmen, augmentasi={args.augment}",
        "split_note": split_note + " Model tersimpan di-refit pada seluruh data.",
        "test": test_metrics, "cv": cv, "comparison": comparison, "ablation": ablation,
    }
    SpeakerRoleClassifier.save_bundle(args.out / "model.joblib", final, args.model, args.use_embedding,
                                      {"test_macro_f1": test_metrics["macro_f1"]})
    save_metrics(args.out, metrics)
    print(f"Model -> {args.out / 'model.joblib'} | metrics -> {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

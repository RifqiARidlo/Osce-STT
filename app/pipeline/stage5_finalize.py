"""Stage 5 (bagian manusia): verifikasi, ekspor ke Bank Kasus OSCE, dan Active Learning.

Saat reviewer menekan "Verifikasi & Ekspor":
  1. Rangkuman divalidasi ulang terhadap schema template (nilai enum tetap terkunci).
  2. PII scrubber dijalankan sekali lagi atas teks hasil edit manusia.
  3. Dual-view disimpan: transkrip berurutan (simulasi komunikasi) + rangkuman (kunci jawaban/rubrik).
  4. Label hasil koreksi manusia ditulis ke training_data/*.jsonl untuk re-training berkala.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from app.config import Settings
from app.nlp.pii import scrub
from app.schemas.pipeline import ReviewPayload
from app.schemas.templates import get_template
from app.storage.db import Repository, utcnow

log = logging.getLogger(__name__)
_append_lock = threading.Lock()


class VerificationError(ValueError):
    pass


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _append_lock, path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_review(template_id: str, payload: ReviewPayload) -> dict:
    spec = get_template(template_id)
    try:
        return spec.model.model_validate(payload.summary).model_dump()
    except Exception as exc:
        raise VerificationError(f"Rangkuman tidak sesuai schema template: {exc}") from exc


def verify_and_export(settings: Settings, repo: Repository, job_id: str, payload: ReviewPayload) -> dict:
    job = repo.get_job(job_id)
    if job is None:
        raise VerificationError("Job tidak ditemukan.")
    if job["status"] not in ("done", "verified"):
        raise VerificationError(f"Job berstatus '{job['status']}', belum bisa diverifikasi.")
    if repo.case_for_job(job_id) is not None:
        raise VerificationError("Job ini sudah diekspor ke Bank Kasus OSCE.")
    result = job["result"] or {}
    original = {u["id"]: u for u in result.get("utterances", [])}

    summary = validate_review(job["template_id"], payload)
    doctor = repo.get_doctor(job["doctor_id"]) if job["doctor_id"] else None
    doctor_names = [doctor["name"]] if doctor else []

    transcript: list[dict] = []
    text_rows: list[dict] = []
    for u in payload.utterances:
        text, _ = scrub(u.text, job["extra_names"], doctor_names)
        transcript.append({"id": u.id, "start": u.start, "end": u.end, "role": u.role,
                           "intent": u.intent, "text": text})
        orig = original.get(u.id, {})
        text_rows.append({
            "text": text, "role": u.role, "label": u.intent, "source": "human_verified",
            "job_id": job_id, "model_label": orig.get("intent"),
            "corrected": orig.get("intent") != u.intent, "created_at": utcnow(),
        })

    # --- Active learning: audio features per utterance (ID utterance = index vektor) ---
    feat_file = settings.features_dir / f"{job_id}.npz"
    audio_rows: list[dict] = []
    if feat_file.is_file() and doctor is not None:
        with np.load(feat_file) as data:
            vectors = data["vectors"]
            embeddings = data["embeddings"] if data["embeddings"].size else None
        for u in payload.utterances:
            if 0 <= u.id < len(vectors):
                orig = original.get(u.id, {})
                audio_rows.append({
                    "session_id": job_id, "doctor_id": doctor["id"], "utt_id": u.id, "label": u.role,
                    "model_label": orig.get("role"), "vector": [round(float(x), 6) for x in vectors[u.id]],
                    "embedding": ([round(float(x), 6) for x in embeddings[u.id]] if embeddings is not None else None),
                })

    _append_jsonl(settings.training_dir / "text_intent.jsonl", text_rows)
    if audio_rows:
        _append_jsonl(settings.training_dir / "speaker_role.jsonl", audio_rows)

    metadata = {
        "doctor_id": job["doctor_id"], "duration_s": result.get("duration_s"),
        "models": result.get("models", {}), "summary_source": result.get("summary_source"),
        "n_corrections_intent": sum(r["corrected"] for r in text_rows),
        "n_corrections_role": sum(1 for r in audio_rows if r["model_label"] != r["label"]),
        "verified_at": utcnow(),
    }
    repo.update_job(job_id, review={"utterances": [u.model_dump() for u in payload.utterances], "summary": summary})
    case = repo.create_case(job_id, job["template_id"], job["doctor_id"], transcript, summary, metadata)
    log.info("Job %s diekspor sebagai kasus OSCE %s", job_id, case["id"])
    return case

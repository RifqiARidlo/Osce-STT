"""Voice Enrollment dokter."""
from __future__ import annotations

import asyncio
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_services, save_upload
from app.core.audio_io import SAMPLE_RATE, AudioLoadError, load_audio
from app.core.services import Services
from app.ml.speaker_role import build_anchor
from app.pipeline.stage1_preprocess import detect_speech, spectral_gate

router = APIRouter(prefix="/api/doctors", tags=["doctors"])
MIN_SPEECH_S = 8.0


@router.get("")
def list_doctors(svc: Services = Depends(get_services)) -> list[dict]:
    return [{"id": d["id"], "name": d["name"], "enroll_seconds": d["enroll_seconds"],
             "created_at": d["created_at"]} for d in svc.repo.list_doctors()]


def _enroll(svc: Services, path, name: str, doctor_id: str) -> dict:
    s = svc.settings
    try:
        audio = load_audio(path, max_minutes=2)
    finally:
        path.unlink(missing_ok=True)  # rekaman enrollment tidak disimpan; hanya profil fitur
    clean = spectral_gate(audio) if s.denoise else audio
    regions, _ = detect_speech(clean, s.vad_backend, s.vad_threshold, s.vad_min_speech_ms,
                               s.vad_min_silence_ms, s.vad_speech_pad_ms)
    if not regions:
        raise ValueError("Tidak ada suara terdeteksi pada rekaman enrollment.")
    # VAD pada audio ter-denoise, tetapi fitur diambil dari audio mentah (konsisten dengan pipeline sesi).
    speech = np.concatenate([audio[int(r.start * SAMPLE_RATE): int(r.end * SAMPLE_RATE)] for r in regions])
    speech_s = speech.size / SAMPLE_RATE
    if speech_s < MIN_SPEECH_S:
        raise ValueError(f"Durasi ucapan terdeteksi {speech_s:.1f}s; minimal {MIN_SPEECH_S:.0f}s. Ulangi rekaman.")
    anchor = build_anchor(speech, svc.embedder)
    anchor_path = s.anchors_dir / f"{doctor_id}.npz"
    anchor.save(anchor_path)
    doc = svc.repo.create_doctor(name, anchor_path, round(speech_s, 2), doctor_id)
    return {"id": doc["id"], "name": doc["name"], "enroll_seconds": doc["enroll_seconds"],
            "n_chunks": anchor.n_chunks, "has_embedding": anchor.embedding is not None}


@router.post("")
async def enroll_doctor(name: str = Form(..., min_length=2, max_length=100), audio: UploadFile = File(...),
                        svc: Services = Depends(get_services)) -> dict:
    doctor_id = uuid.uuid4().hex[:12]
    path = await save_upload(audio, svc.settings.uploads_dir / f"enroll_{doctor_id}", 20)
    try:
        return await asyncio.to_thread(_enroll, svc, path, name.strip(), doctor_id)
    except (AudioLoadError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc

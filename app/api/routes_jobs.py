"""Submit rekaman, status antrean, review & verifikasi."""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.deps import get_services, save_upload
from app.core.services import Services
from app.pipeline.stage5_finalize import VerificationError, validate_review, verify_and_export
from app.schemas.pipeline import ReviewPayload
from app.schemas.templates import TEMPLATES

router = APIRouter(prefix="/api", tags=["jobs"])


def _parse_json_field(raw: str, default, name: str):
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Field {name} bukan JSON valid.") from exc
    if not isinstance(value, type(default)):
        raise HTTPException(400, f"Field {name} harus bertipe {type(default).__name__}.")
    return value


@router.post("/jobs")
async def submit_job(request: Request, audio: UploadFile = File(...), template_id: str = Form("soap_umum"),
                     doctor_id: str = Form(""), extra_names: str = Form("[]"), locked: str = Form("{}"),
                     svc: Services = Depends(get_services)) -> dict:
    if template_id not in TEMPLATES:
        raise HTTPException(400, f"Template tidak dikenal: {template_id}")
    if doctor_id and svc.repo.get_doctor(doctor_id) is None:
        raise HTTPException(404, "Profil dokter tidak ditemukan.")
    names = [str(n) for n in _parse_json_field(extra_names, [], "extra_names")][:20]
    locked_values = _parse_json_field(locked, {}, "locked")
    model_fields = TEMPLATES[template_id].model.model_fields
    unknown = set(locked_values) - set(model_fields)
    if unknown:
        raise HTTPException(400, f"Parameter terkunci tidak dikenal untuk template ini: {sorted(unknown)}")

    job_id = uuid.uuid4().hex[:12]
    path = await save_upload(audio, svc.settings.uploads_dir / job_id, svc.settings.max_upload_mb)
    client_ip = request.client.host if request.client else None
    svc.repo.create_job(job_id, doctor_id or None, template_id, path, names, locked_values, client_ip)
    position = await svc.queue.submit(job_id)
    return {"job_id": job_id, "position": position}


@router.get("/jobs")
def list_jobs(limit: int = 30, svc: Services = Depends(get_services)) -> list[dict]:
    return svc.repo.list_jobs(min(max(limit, 1), 200))


@router.get("/jobs/{job_id}")
def get_job(job_id: str, light: bool = False, svc: Services = Depends(get_services)) -> dict:
    """light=true -> tanpa result/review (untuk polling status yang sering)."""
    job = svc.repo.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job tidak ditemukan.")
    job.pop("audio_path", None)
    job.pop("client_ip", None)
    if light:
        job.pop("result", None)
        job.pop("review", None)
    job["position"] = svc.queue.position(job_id)
    case = svc.repo.case_for_job(job_id)
    job["case_id"] = case["id"] if case else None
    return job


@router.get("/queue")
def queue_status(svc: Services = Depends(get_services)) -> dict:
    snap = svc.queue.snapshot()
    snap["gpu_loaded"] = svc.ctx.slot.loaded
    return snap


@router.put("/jobs/{job_id}/review")
def save_review(job_id: str, payload: ReviewPayload, svc: Services = Depends(get_services)) -> dict:
    job = svc.repo.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job tidak ditemukan.")
    if job["status"] not in ("done",):
        raise HTTPException(409, f"Job berstatus '{job['status']}', tidak bisa diedit.")
    try:
        summary = validate_review(job["template_id"], payload)
    except VerificationError as exc:
        raise HTTPException(422, str(exc)) from exc
    svc.repo.update_job(job_id, review={"utterances": [u.model_dump() for u in payload.utterances],
                                        "summary": summary})
    return {"ok": True}


@router.post("/jobs/{job_id}/verify")
def verify_job(job_id: str, payload: ReviewPayload, svc: Services = Depends(get_services)) -> dict:
    try:
        case = verify_and_export(svc.settings, svc.repo, job_id, payload)
    except VerificationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"case_id": case["id"]}

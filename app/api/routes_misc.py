"""Template, Bank Kasus OSCE, metrik ML, health."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from app.api.deps import ffmpeg_available, get_services
from app.core.services import Services
from app.schemas.templates import TEMPLATES, inline_json_schema

router = APIRouter(prefix="/api", tags=["misc"])


def _lockable_fields(schema: dict[str, Any]) -> list[str]:
    """Field kategorikal top-level (enum) -> bisa dikunci pengguna sebelum merekam."""
    return [name for name, prop in schema.get("properties", {}).items() if "enum" in prop]


@router.get("/health")
def health(svc: Services = Depends(get_services)) -> dict:
    s = svc.settings
    return {
        "status": "ok",
        "llm_backend": s.llm_backend,
        "llm_model": s.ollama_model if s.llm_backend == "ollama" else s.llamacpp_model_path,
        "whisper_model": s.whisper_model,
        "gpu_swap_policy": s.gpu_swap_policy,
        "gpu_loaded": svc.ctx.slot.loaded,
        "speaker_model": svc.ctx.speaker_clf.description,
        "intent_model": svc.ctx.intent_clf.description,
        "ecapa": svc.embedder is not None,
        "ffmpeg": ffmpeg_available(),
        "auth": bool(s.access_token),
    }


@router.get("/templates")
def list_templates() -> list[dict]:
    out = []
    for spec in TEMPLATES.values():
        schema = inline_json_schema(spec.model)
        out.append({"id": spec.id, "title": spec.title, "description": spec.description,
                    "schema": schema, "lockable": _lockable_fields(schema)})
    return out


@router.get("/cases")
def list_cases(svc: Services = Depends(get_services)) -> list[dict]:
    return [{"id": c["id"], "job_id": c["job_id"], "template_id": c["template_id"],
             "keluhan_utama": c["summary"].get("keluhan_utama", "-"), "n_utterances": len(c["transcript"]),
             "created_at": c["created_at"]} for c in svc.repo.list_cases()]


@router.get("/cases/export.jsonl", response_class=PlainTextResponse)
def export_cases(svc: Services = Depends(get_services)) -> PlainTextResponse:
    lines = [json.dumps(c, ensure_ascii=False) for c in svc.repo.list_cases()]
    return PlainTextResponse("\n".join(lines), media_type="application/x-ndjson",
                             headers={"Content-Disposition": "attachment; filename=osce_case_bank.jsonl"})


@router.get("/cases/{case_id}")
def get_case(case_id: str, svc: Services = Depends(get_services)) -> dict:
    case = svc.repo.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Kasus tidak ditemukan.")
    return case


@router.get("/ml/metrics")
def ml_metrics(svc: Services = Depends(get_services)) -> dict:
    out: dict[str, Any] = {}
    for name in ("speaker_role", "text_intent"):
        path = svc.settings.models_dir / name / "metrics.json"
        out[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    counts = {}
    for fname in ("text_intent.jsonl", "speaker_role.jsonl"):
        path = svc.settings.training_dir / fname
        counts[fname] = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    out["active_learning_samples"] = counts
    out["loaded"] = {"speaker": svc.ctx.speaker_clf.description, "intent": svc.ctx.intent_clf.description}
    return out


@router.post("/ml/reload")
def ml_reload(svc: Services = Depends(get_services)) -> dict:
    """Muat ulang model setelah re-training tanpa restart server."""
    return svc.reload_classifiers()

"""Stage 4: Constrained Clinical Standardization & Summarization.

Alur: utterance (non-CHITCHAT) + teks hasil lexicon mapper -> prompt -> LLM dengan JSON Schema
(constrained decoding) -> validasi Pydantic -> verifikasi `sumber` (ID utterance harus ada) ->
bila gagal: retry dengan pesan error -> bila tetap gagal: fallback rule-based (tidak pernah crash).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from app.llm.backends import LLMBackend, LLMError
from app.schemas.pipeline import Utterance
from app.schemas.templates import TD, EvidenceItem, TemplateSpec, evidence_fields, llm_json_schema

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Anda adalah asisten dokumentasi klinis berbahasa Indonesia.
Tugas: menyusun rangkuman terstandar dari potongan percakapan dokter-pasien.
ATURAN WAJIB:
1. Gunakan HANYA informasi yang tertulis dalam utterance. Dilarang menambah diagnosis, obat, dosis,
   atau hasil pemeriksaan yang tidak diucapkan.
2. Jika informasi tidak ada, isi "Tidak disebutkan" (untuk pilihan) atau biarkan daftar kosong.
3. Setiap butir naratif wajib mencantumkan `sumber` berisi ID utterance [angka] pendukungnya.
4. Tulis dengan istilah medis baku yang ringkas. Istilah dalam kurung adalah ucapan asli pasien.
5. Label kelas (S_KELUHAN, O_PEMERIKSAAN, A_DIAGNOSIS, P_EDUKASI_TERAPI) adalah prediksi model dan
   bisa keliru; utamakan isi kalimat.
6. Field rubrik bernilai true hanya jika dokter benar-benar menanyakan/melakukan hal tersebut."""


def build_user_prompt(spec: TemplateSpec, utterances: list[Utterance], locked: dict[str, Any],
                      schema: dict[str, Any] | None = None) -> str:
    lines = [f"Template: {spec.title} — {spec.description}"]
    if locked:
        lines.append("Parameter yang sudah dikunci pengguna (salin apa adanya): "
                     + json.dumps(locked, ensure_ascii=False))
    lines.append("\nUtterance (format: [ID] PERAN/KELAS: teks):")
    for u in utterances:
        if u.intent == "CHITCHAT":
            continue
        text = u.text_normalized or u.text
        lines.append(f"[{u.id}] {u.role}/{u.intent}: {text}")
    if schema is not None:
        # Grammar hanya membatasi bentuk output; model perlu melihat nama, enum & deskripsi field.
        lines.append("\nJSON Schema output:\n" + json.dumps(schema, ensure_ascii=False))
    lines.append("\nKeluarkan JSON sesuai schema.")
    return "\n".join(lines)


def _validate_sources(summary: BaseModel, valid_ids: set[int], warnings: list[str]) -> BaseModel:
    for field in evidence_fields(type(summary)):
        items: list[EvidenceItem] = getattr(summary, field)
        for item in items:
            bad = [i for i in item.sumber if i not in valid_ids]
            if bad:
                item.sumber = [i for i in item.sumber if i in valid_ids]
                warnings.append(f"{field}: sumber tidak valid {bad} dihapus")
            if not item.sumber:
                warnings.append(f"{field}: butir tanpa sumber -> wajib dicek manual: '{item.teks[:60]}'")
    return summary


def _apply_locked(summary: BaseModel, locked: dict[str, Any]) -> BaseModel:
    if not locked:
        return summary
    data = summary.model_dump()
    data.update({k: v for k, v in locked.items() if k in data and v not in (None, "", TD)})
    return type(summary).model_validate(data)


def rule_based_summary(spec: TemplateSpec, utterances: list[Utterance], locked: dict[str, Any]) -> BaseModel:
    data: dict[str, list[dict]] = {}
    for u in utterances:
        field = spec.intent_field_map.get(u.intent)
        if field:
            data.setdefault(field, []).append({"teks": u.text_normalized or u.text, "sumber": [u.id]})
    summary = spec.model.model_validate(data)
    complaints = [u for u in utterances if u.intent == "S_KELUHAN" and u.role == "PASIEN"]
    if complaints and "keluhan_utama" in spec.model.model_fields:
        summary = summary.model_copy(update={"keluhan_utama": complaints[0].text_normalized or complaints[0].text})
    return _apply_locked(summary, locked)


def summarize(spec: TemplateSpec, utterances: list[Utterance], llm: LLMBackend | None,
              locked: dict[str, Any] | None = None, max_retries: int = 1) -> tuple[dict, str, list[str]]:
    """Return (summary_dict, source['llm'|'fallback_rule'], warnings)."""
    locked = locked or {}
    warnings: list[str] = []
    valid_ids = {u.id for u in utterances}
    clinical = [u for u in utterances if u.intent != "CHITCHAT"]
    if llm is None or not clinical:
        if not clinical:
            warnings.append("Tidak ada utterance klinis (semua CHITCHAT).")
        return rule_based_summary(spec, utterances, locked).model_dump(), "fallback_rule", warnings

    schema = llm_json_schema(spec.model)
    user_prompt = build_user_prompt(spec, utterances, locked, schema)
    last_error = ""
    for attempt in range(max_retries + 1):
        prompt = user_prompt if not last_error else (
            f"{user_prompt}\n\nOutput sebelumnya tidak valid: {last_error[:400]}\nPerbaiki dan keluarkan JSON valid.")
        try:
            raw = llm.generate_json(SYSTEM_PROMPT, prompt, schema)
            summary = spec.model.model_validate_json(raw)
            summary = _apply_locked(summary, locked)
            summary = _validate_sources(summary, valid_ids, warnings)
            return summary.model_dump(), "llm", warnings
        except ValidationError as exc:
            last_error = str(exc)
            log.warning("Output LLM tidak valid (percobaan %d): %s", attempt + 1, last_error[:200])
        except LLMError as exc:
            warnings.append(f"LLM gagal: {exc}")
            break
    warnings.append("Rangkuman memakai fallback rule-based (LLM tidak memberi output valid).")
    return rule_based_summary(spec, utterances, locked).model_dump(), "fallback_rule", warnings

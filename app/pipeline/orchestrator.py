"""Orkestrator 5-stage pipeline. Dipanggil oleh JobQueue di worker thread tunggal (sekuensial)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config import Settings
from app.core.audio_io import SAMPLE_RATE, load_audio
from app.core.model_slot import GpuModelSlot
from app.ml.audio_features import extract_many
from app.ml.speaker_role import DoctorAnchor, SpeakerRoleClassifier
from app.ml.text_intent import IntentClassifier
from app.nlp.lexicon import MedicalLexicon
from app.nlp.pii import scrub
from app.pipeline.stage1_preprocess import detect_speech, spectral_gate
from app.pipeline.stage2_asr import segment_words
from app.pipeline.stage4_summarize import summarize
from app.schemas.pipeline import PipelineResult, Utterance
from app.schemas.templates import get_template
from app.storage.db import Repository, utcnow

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    settings: Settings
    repo: Repository
    slot: GpuModelSlot
    speaker_clf: SpeakerRoleClassifier
    intent_clf: IntentClassifier
    lexicon: MedicalLexicon
    glossary_prompt: str
    embedder: Any | None = None
    has_llm: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            self.timings[name] = round(time.perf_counter() - t0, 3)


def features_path(settings: Settings, job_id: str) -> Path:
    return settings.features_dir / f"{job_id}.npz"


def run_pipeline(ctx: PipelineContext, job_id: str) -> PipelineResult:
    s = ctx.settings
    job = ctx.repo.get_job(job_id)
    if job is None:
        raise ValueError(f"Job {job_id} tidak ditemukan")
    spec = get_template(job["template_id"])
    doctor = ctx.repo.get_doctor(job["doctor_id"]) if job["doctor_id"] else None
    timer = _Timer()
    warnings: list[str] = []

    def stage(name: str) -> None:
        ctx.repo.update_job(job_id, stage=name)

    # ---------------- Stage 1 ----------------
    stage("stage1_preprocess")
    audio = timer.run("load_audio", lambda: load_audio(Path(job["audio_path"]), s.max_audio_minutes))
    duration = audio.size / SAMPLE_RATE
    clean = timer.run("denoise", lambda: spectral_gate(audio) if s.denoise else audio)
    regions, vad_name = timer.run("vad", lambda: detect_speech(
        clean, s.vad_backend, s.vad_threshold, s.vad_min_speech_ms, s.vad_min_silence_ms, s.vad_speech_pad_ms))
    if not regions:
        raise RuntimeError("Tidak ada suara terdeteksi (VAD kosong). Periksa mikrofon / volume.")

    # ---------------- Stage 2: ASR ----------------
    stage("stage2_asr")
    with ctx.slot.use("asr") as asr:
        words = timer.run("asr", lambda: asr.transcribe(clean, ctx.glossary_prompt))
        asr_name = getattr(asr, "model_name", "asr")
    raw_utts = segment_words(words, regions, s.utterance_gap_s, s.utterance_max_s)
    if not raw_utts:
        raise RuntimeError("ASR tidak menghasilkan teks.")

    # ---------------- Stage 2: Speaker role ----------------
    stage("stage2_speaker")
    spans = [(u.start, u.end) for u in raw_utts]
    # Fitur speaker diambil dari audio MENTAH (bukan hasil denoise): spectral gating mengubah MFCC/centroid
    # secara tidak konsisten antar-segmen (tergantung SNR lokal) sehingga merusak jarak ke anchor.
    # Enrollment memakai aturan yang sama (routes_doctors._enroll).
    vectors = timer.run("audio_features", lambda: extract_many(audio, spans))
    embeddings = None
    if ctx.embedder is not None:
        embeddings = timer.run("ecapa", lambda: ctx.embedder.embed_many(audio, spans))
    if doctor is not None:
        anchor = DoctorAnchor.load(Path(doctor["anchor_path"]))
        roles = timer.run("speaker_clf", lambda: ctx.speaker_clf.predict(vectors, anchor, embeddings))
    else:
        warnings.append("Tidak ada profil dokter dipilih: semua peran = PASIEN (confidence 0). Koreksi manual.")
        roles = [("PASIEN", 0.0, "heuristic")] * len(raw_utts)
    np.savez_compressed(features_path(s, job_id), vectors=vectors,
                        embeddings=embeddings if embeddings is not None else np.zeros((0,), np.float32))

    # ---------------- PII scrub (sedini mungkin: sebelum apa pun disimpan) ----------------
    doctor_names = [doctor["name"]] if doctor else []
    utterances: list[Utterance] = []
    pii_hits: list[str] = []
    for i, (ru, (role, rconf, rsrc)) in enumerate(zip(raw_utts, roles)):
        text, hits = scrub(ru.text, job["extra_names"], doctor_names)
        pii_hits += hits
        utterances.append(Utterance(id=i, start=round(ru.start, 2), end=round(ru.end, 2), text=text,
                                    asr_confidence=round(ru.confidence, 3), role=role,  # type: ignore[arg-type]
                                    role_confidence=round(rconf, 4), role_source=rsrc))  # type: ignore[arg-type]

    # ---------------- Stage 3 ----------------
    stage("stage3_intent")
    preds = timer.run("intent_clf", lambda: ctx.intent_clf.predict(
        [u.text for u in utterances], [u.role for u in utterances]))
    for u, (label, conf, probs, src) in zip(utterances, preds):
        u.intent, u.intent_confidence, u.intent_probs, u.intent_source = label, round(conf, 4), probs, src  # type: ignore[assignment]
        u.text_normalized, u.lexicon_hits = ctx.lexicon.apply(u.text)

    # ---------------- Stage 4 ----------------
    stage("stage4_summarize")
    if ctx.has_llm:
        with ctx.slot.use("llm") as llm:
            summary, source, w = timer.run("llm", lambda: summarize(
                spec, utterances, llm, job["locked"], s.llm_max_retries))
            llm_name = getattr(llm, "name", "llm")
    else:
        summary, source, w = summarize(spec, utterances, None, job["locked"])
        llm_name = "none"
    warnings += w

    # ---------------- Stage 5 (bagian otomatis) ----------------
    stage("stage5_finalize")
    if not s.keep_raw_audio:
        try:
            Path(job["audio_path"]).unlink(missing_ok=True)
        except OSError as exc:
            warnings.append(f"Gagal menghapus audio mentah: {exc}")

    return PipelineResult(
        job_id=job_id, template_id=spec.id, duration_s=round(duration, 2), utterances=utterances,
        summary=summary, summary_source=source,  # type: ignore[arg-type]
        pii_hits=pii_hits, warnings=warnings, timings=timer.timings,
        models={"vad": vad_name, "asr": asr_name, "speaker": ctx.speaker_clf.description,
                "intent": ctx.intent_clf.description, "llm": llm_name,
                **{f"load_{k}": f"{v:.1f}s" for k, v in ctx.slot.load_times.items()}},
    )


def process_job(ctx: PipelineContext, job_id: str) -> None:
    """Entry point worker: menjalankan pipeline + mencatat status ke DB. Tidak melempar exception."""
    ctx.repo.update_job(job_id, status="processing", started_at=utcnow())
    try:
        result = run_pipeline(ctx, job_id)
        ctx.repo.update_job(job_id, status="done", stage="done", finished_at=utcnow(),
                            result=result.model_dump(), review=None)
        log.info("Job %s selesai (%s)", job_id, result.timings)
    except Exception as exc:
        log.exception("Job %s gagal", job_id)
        ctx.repo.update_job(job_id, status="failed", stage="failed", error=str(exc)[:1000], finished_at=utcnow())

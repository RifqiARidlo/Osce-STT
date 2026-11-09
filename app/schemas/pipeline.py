"""Model data internal pipeline & API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SpeakerRole = Literal["DOKTER", "PASIEN"]
IntentLabel = Literal["S_KELUHAN", "O_PEMERIKSAAN", "A_DIAGNOSIS", "P_EDUKASI_TERAPI", "CHITCHAT"]
INTENT_LABELS: tuple[str, ...] = ("S_KELUHAN", "O_PEMERIKSAAN", "A_DIAGNOSIS", "P_EDUKASI_TERAPI", "CHITCHAT")
SPEAKER_LABELS: tuple[str, ...] = ("DOKTER", "PASIEN")

JobStatus = Literal["queued", "processing", "done", "failed", "verified"]
PipelineStage = Literal["queued", "stage1_preprocess", "stage2_asr", "stage2_speaker",
                        "stage3_intent", "stage4_summarize", "stage5_finalize", "done", "failed"]


class SpeechRegion(BaseModel):
    start: float
    end: float


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float = 1.0


class Utterance(BaseModel):
    id: int
    start: float
    end: float
    text: str                       # sudah di-scrub PII
    text_normalized: str = ""       # setelah lexicon mapper (input LLM)
    asr_confidence: float = 1.0
    role: SpeakerRole = "PASIEN"
    role_confidence: float = 0.0
    role_source: Literal["model", "heuristic", "human"] = "heuristic"
    intent: IntentLabel = "CHITCHAT"
    intent_confidence: float = 0.0
    intent_probs: dict[str, float] = Field(default_factory=dict)
    intent_source: Literal["model", "heuristic", "human"] = "heuristic"
    lexicon_hits: list[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    job_id: str
    template_id: str
    duration_s: float
    utterances: list[Utterance]
    summary: dict[str, Any]
    summary_source: Literal["llm", "fallback_rule"]
    pii_hits: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)


class ReviewPayload(BaseModel):
    utterances: list[Utterance]
    summary: dict[str, Any]

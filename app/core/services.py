"""Komposisi dependency (dibangun sekali saat startup server)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.config import Settings
from app.core.job_queue import JobQueue
from app.core.model_slot import GpuModelSlot
from app.llm.backends import LlamaCppBackend, OllamaBackend
from app.ml.speaker_embedding import load_embedder
from app.ml.speaker_role import SpeakerRoleClassifier
from app.ml.text_intent import IntentClassifier
from app.nlp.lexicon import MedicalLexicon
from app.pipeline.orchestrator import PipelineContext, process_job
from app.pipeline.stage2_asr import WhisperASR, load_glossary_prompt
from app.storage.db import Repository

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    repo: Repository
    ctx: PipelineContext
    queue: JobQueue
    embedder: Any | None

    def reload_classifiers(self) -> dict[str, str]:
        self.ctx.speaker_clf = SpeakerRoleClassifier.load(self.settings.speaker_model_path)
        self.ctx.intent_clf = IntentClassifier.load(self.settings.intent_model_path)
        return {"speaker": self.ctx.speaker_clf.description, "intent": self.ctx.intent_clf.description}


def _model_factories(s: Settings) -> dict[str, Callable[[], Any]]:
    def make_asr() -> WhisperASR:
        device = s.whisper_device
        compute = s.whisper_compute_type
        if device == "cpu" and compute.endswith("float16"):
            compute = "int8"  # float16 tidak didukung CPU
        return WhisperASR(s.whisper_model, device, compute, s.models_dir / "whisper", s.language,
                          s.whisper_beam_size)

    def make_llm() -> Any:
        if s.llm_backend == "ollama":
            return OllamaBackend(s.ollama_url, s.ollama_model, s.llm_ctx, s.llm_timeout_s,
                                 unload_after=s.gpu_swap_policy == "swap")
        if s.llm_backend == "llamacpp":
            return LlamaCppBackend(s.llamacpp_model_path, s.llamacpp_n_gpu_layers, s.llm_ctx)
        raise RuntimeError("LLM backend = none")

    return {"asr": make_asr, "llm": make_llm}


def build_services(s: Settings) -> Services:
    s.ensure_dirs()
    repo = Repository(s.db_path)
    embedder = load_embedder(s.use_ecapa, s.models_dir)
    ctx = PipelineContext(
        settings=s,
        repo=repo,
        slot=GpuModelSlot(_model_factories(s), s.gpu_swap_policy),
        speaker_clf=SpeakerRoleClassifier.load(s.speaker_model_path),
        intent_clf=IntentClassifier.load(s.intent_model_path),
        lexicon=MedicalLexicon.from_file(s.lexicon_path),
        glossary_prompt=load_glossary_prompt(s.glossary_path),
        embedder=embedder,
        has_llm=s.llm_backend != "none",
    )
    queue = JobQueue(lambda job_id: process_job(ctx, job_id))
    return Services(settings=s, repo=repo, ctx=ctx, queue=queue, embedder=embedder)

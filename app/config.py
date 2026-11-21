"""Konfigurasi aplikasi dari environment variable (tanpa dependency pydantic-settings)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(path: Path) -> None:
    """Parser .env minimal: KEY=VALUE per baris, tidak menimpa env yang sudah ada."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings(BaseModel):
    # --- Path ---
    data_dir: Path
    models_dir: Path
    db_path: Path

    # --- Server / keamanan ---
    access_token: str = ""  # kosong = tanpa token (hanya untuk LAN tertutup)
    max_upload_mb: int = 200
    max_audio_minutes: float = 40.0
    keep_raw_audio: bool = False  # audio mentah = PII (suara). Default dihapus setelah diproses.

    # --- Stage 1 ---
    denoise: bool = True
    vad_backend: Literal["silero", "energy"] = "silero"
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 300
    vad_min_speech_ms: int = 250
    vad_speech_pad_ms: int = 100

    # --- Stage 2 ---
    whisper_model: str = "small"
    whisper_device: Literal["cuda", "cpu", "auto"] = "cuda"
    whisper_compute_type: str = "int8_float16"  # GPU. CPU: "int8"
    whisper_beam_size: int = 5
    language: str = "id"
    utterance_gap_s: float = 0.6
    utterance_max_s: float = 15.0
    use_ecapa: bool = False  # butuh torch + speechbrain

    # --- Stage 4 ---
    llm_backend: Literal["ollama", "llamacpp", "deepseek", "none"] = "ollama"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"
    llamacpp_model_path: str = ""
    llamacpp_n_gpu_layers: int = -1
    deepseek_api_key: str = Field(default="", repr=False)  # rahasia: jangan dicetak ke log
    deepseek_model: str = "deepseek-chat"
    deepseek_url: str = "https://api.deepseek.com"
    llm_ctx: int = 8192
    llm_timeout_s: int = 300
    llm_max_retries: int = 1

    # --- GPU ---
    gpu_swap_policy: Literal["swap", "coexist"] = "swap"

    speaker_model_path: Path = Field(default=Path())
    intent_model_path: Path = Field(default=Path())

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def anchors_dir(self) -> Path:
        return self.data_dir / "anchors"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def training_dir(self) -> Path:
        return self.data_dir / "training_data"

    @property
    def lexicon_path(self) -> Path:
        return PROJECT_ROOT / "data" / "lexicon" / "medical_lexicon.json"

    @property
    def glossary_path(self) -> Path:
        return PROJECT_ROOT / "data" / "glossary" / "medical_glossary.txt"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.uploads_dir, self.anchors_dir,
                  self.features_dir, self.training_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(_env("APP_DATA_DIR", str(PROJECT_ROOT / "runtime")))
    models_dir = Path(_env("APP_MODELS_DIR", str(PROJECT_ROOT / "models")))
    settings = Settings(
        data_dir=data_dir,
        models_dir=models_dir,
        db_path=Path(_env("APP_DB_PATH", str(data_dir / "osce.sqlite3"))),
        access_token=_env("APP_ACCESS_TOKEN", ""),
        max_upload_mb=int(_env("APP_MAX_UPLOAD_MB", "200")),
        max_audio_minutes=float(_env("APP_MAX_AUDIO_MINUTES", "40")),
        keep_raw_audio=_env_bool("APP_KEEP_RAW_AUDIO", False),
        denoise=_env_bool("APP_DENOISE", True),
        vad_backend=_env("APP_VAD_BACKEND", "silero"),  # type: ignore[arg-type]
        vad_threshold=float(_env("APP_VAD_THRESHOLD", "0.5")),
        vad_min_silence_ms=int(_env("APP_VAD_MIN_SILENCE_MS", "300")),
        whisper_model=_env("APP_WHISPER_MODEL", "small"),
        whisper_device=_env("APP_WHISPER_DEVICE", "cuda"),  # type: ignore[arg-type]
        whisper_compute_type=_env("APP_WHISPER_COMPUTE_TYPE", "int8_float16"),
        whisper_beam_size=int(_env("APP_WHISPER_BEAM_SIZE", "5")),
        language=_env("APP_LANGUAGE", "id"),
        utterance_gap_s=float(_env("APP_UTTERANCE_GAP_S", "0.6")),
        utterance_max_s=float(_env("APP_UTTERANCE_MAX_S", "15")),
        use_ecapa=_env_bool("APP_USE_ECAPA", False),
        llm_backend=_env("APP_LLM_BACKEND", "ollama"),  # type: ignore[arg-type]
        ollama_url=_env("APP_OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=_env("APP_OLLAMA_MODEL", "qwen2.5:3b"),
        llamacpp_model_path=_env("APP_LLAMACPP_MODEL_PATH", ""),
        llamacpp_n_gpu_layers=int(_env("APP_LLAMACPP_N_GPU_LAYERS", "-1")),
        deepseek_api_key=_env("APP_DEEPSEEK_API_KEY", ""),
        deepseek_model=_env("APP_DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_url=_env("APP_DEEPSEEK_URL", "https://api.deepseek.com"),
        llm_ctx=int(_env("APP_LLM_CTX", "8192")),
        llm_timeout_s=int(_env("APP_LLM_TIMEOUT_S", "300")),
        gpu_swap_policy=_env("APP_GPU_SWAP_POLICY", "swap"),  # type: ignore[arg-type]
        speaker_model_path=Path(_env("APP_SPEAKER_MODEL_PATH",
                                     str(models_dir / "speaker_role" / "model.joblib"))),
        intent_model_path=Path(_env("APP_INTENT_MODEL_PATH",
                                    str(models_dir / "text_intent" / "model.joblib"))),
    )
    return settings

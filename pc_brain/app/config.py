import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used before dependencies are installed.
    def load_dotenv(*args, **kwargs):
        return False


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    robot_base_url: str
    request_timeout: float
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_think: bool
    llm_timeout: float
    stt_provider: str
    stt_model: str
    stt_device: str
    stt_compute_type: str
    tts_provider: str
    tts_model: str
    tts_language: str
    tts_device: str
    voice_id: str
    data_dir: Path
    warm_models: bool

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("ROBIT_DATA_DIR", "./data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir

    return Settings(
        robot_base_url=os.getenv("ROBIT_BASE_URL", "http://192.168.4.1").rstrip("/"),
        request_timeout=_float_env("ROBIT_REQUEST_TIMEOUT", 2.0),
        llm_provider=os.getenv("ROBIT_LLM_PROVIDER", "ollama"),
        llm_base_url=os.getenv("ROBIT_LLM_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=os.getenv("ROBIT_LLM_MODEL", "gemma4:e4b"),
        llm_think=_bool_env("ROBIT_LLM_THINK", False),
        llm_timeout=_float_env("ROBIT_LLM_TIMEOUT", 30.0),
        stt_provider=os.getenv("ROBIT_STT_PROVIDER", "faster_whisper"),
        stt_model=os.getenv("ROBIT_STT_MODEL", "base"),
        stt_device=os.getenv("ROBIT_STT_DEVICE", "cuda"),
        stt_compute_type=os.getenv("ROBIT_STT_COMPUTE_TYPE", "float16"),
        tts_provider=os.getenv("ROBIT_TTS_PROVIDER", "xtts"),
        tts_model=os.getenv(
            "ROBIT_TTS_MODEL",
            "tts_models/multilingual/multi-dataset/xtts_v2",
        ),
        tts_language=os.getenv("ROBIT_TTS_LANGUAGE", "en"),
        tts_device=os.getenv("ROBIT_TTS_DEVICE", "cuda"),
        voice_id=os.getenv("ROBIT_VOICE_ID", "default"),
        data_dir=data_dir,
        warm_models=_bool_env("ROBIT_WARM_MODELS", True),
    )


settings = load_settings()

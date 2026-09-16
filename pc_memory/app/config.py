"""Settings for pc_memory (MEMORY_* env vars, mirrors pc_brain/app/config.py)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LLM_MODEL = "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
# Dedicated embedding endpoint (Bekko a25m via llama-server --embedding).
# Kept separate from the chat LLM so retrieval degrades independently.
DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:8093/v1"
DEFAULT_EMBED_MODEL = "bekko"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used before dependencies are installed.

    def load_dotenv(*args, **kwargs) -> bool:
        return False


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    port: int
    db_path: Path
    llm_base_url: str
    llm_model: str
    llm_timeout: float
    context_budget_chars: int
    max_hops: int
    beam: int
    embed_base_url: str
    embed_model: str


def load_settings() -> Settings:
    load_dotenv()
    # Default DB lives under pc_memory/data/ (resolved relative to the package root).
    db_path = Path(os.getenv("MEMORY_DB_PATH", "data/memory.db")).expanduser()
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path

    return Settings(
        port=_int_env("MEMORY_PORT", 8092),
        db_path=db_path,
        llm_base_url=os.getenv("MEMORY_LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/"),
        llm_model=os.getenv("MEMORY_LLM_MODEL", DEFAULT_LLM_MODEL),
        llm_timeout=_float_env("MEMORY_LLM_TIMEOUT", 30.0),
        context_budget_chars=_int_env("MEMORY_CONTEXT_BUDGET_CHARS", 6000),
        max_hops=_int_env("MEMORY_MAX_HOPS", 3),
        beam=_int_env("MEMORY_BEAM", 6),
        embed_base_url=os.getenv(
            "MEMORY_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL
        ).rstrip("/"),
        embed_model=os.getenv("MEMORY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
    )


settings = load_settings()

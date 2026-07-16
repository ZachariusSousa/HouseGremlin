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


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    robot_base_url: str
    request_timeout: float
    robot_request_retries: int
    robot_retry_backoff_seconds: float
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_think: bool
    llm_timeout: float
    realtime_ws_url: str
    realtime_voice: str
    realtime_instructions: str
    robot_llm_max_speed: int
    robot_llm_max_duration_ms: int
    data_dir: Path
    warm_models: bool


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("ROBIT_DATA_DIR", "./data")).expanduser()
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir

    return Settings(
        robot_base_url=os.getenv("ROBIT_BASE_URL", "http://robit.local").rstrip("/"),
        request_timeout=_float_env("ROBIT_REQUEST_TIMEOUT", 2.0),
        robot_request_retries=_int_env("ROBIT_REQUEST_RETRIES", 2),
        robot_retry_backoff_seconds=_float_env("ROBIT_RETRY_BACKOFF_SECONDS", 0.15),
        llm_provider=os.getenv("ROBIT_LLM_PROVIDER", "openai_compatible"),
        llm_base_url=os.getenv("ROBIT_LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
        llm_model=os.getenv("ROBIT_LLM_MODEL", "gemma4:e4b"),
        llm_think=_bool_env("ROBIT_LLM_THINK", False),
        llm_timeout=_float_env("ROBIT_LLM_TIMEOUT", 30.0),
        realtime_ws_url=os.getenv("ROBIT_REALTIME_WS_URL", "ws://localhost:7861/v1/realtime"),
        realtime_voice=os.getenv("ROBIT_REALTIME_VOICE", "serena"),
        realtime_instructions=os.getenv(
            "ROBIT_REALTIME_INSTRUCTIONS",
            (
                "You are Robit, a small helpful home robot. Talk like Rocky from Project Hail Mary. "
                "Be concise and use plain spoken text only. You can call the robot_action tool for "
                "safe movement, head, or emergency-stop actions when the user asks. You may also "
                "select a temporary emotional eye expression when a message genuinely warrants it; "
                "when the user explicitly asks for an expression, call robot_action with the eyes field and "
                "do not claim it changed without that tool call. Operational eye states are automatic."
            ),
        ),
        robot_llm_max_speed=_int_env("ROBIT_LLM_MAX_SPEED", 180),
        robot_llm_max_duration_ms=_int_env("ROBIT_LLM_MAX_DURATION_MS", 1000),
        data_dir=data_dir,
        warm_models=_bool_env("ROBIT_WARM_MODELS", True),
    )


settings = load_settings()

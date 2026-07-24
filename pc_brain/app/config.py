import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_E4B_MODEL = "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
DEFAULT_E4B_BASE_URL = "http://127.0.0.1:8081/v1"

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
    robot_llm_default_speed: int
    robot_llm_max_duration_ms: int
    data_dir: Path
    warm_models: bool
    vision_enabled: bool = True
    vision_base_url: str = DEFAULT_E4B_BASE_URL
    vision_model: str = DEFAULT_E4B_MODEL
    vision_request_timeout_seconds: float = 30.0
    vision_max_output_tokens: int = 320
    vision_image_tokens: int = 140
    vision_awareness_interval_seconds: float = 5.0
    vision_snapshot_ttl_seconds: float = 10.0
    vision_world_window_seconds: float = 60.0
    vision_change_threshold: float = 0.03
    vision_blur_threshold: float = 20.0
    camera_frame_interval_seconds: float = 5.0
    camera_rotate_degrees: int = 180
    tracking_enabled: bool = True
    tracking_base_url: str = "http://127.0.0.1:8091"
    tracking_request_timeout_seconds: float = 2.0
    tracking_confidence: float = 0.40
    tracking_pan_sign: int = 1
    tracking_tilt_sign: int = 1


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
        llm_base_url=os.getenv("ROBIT_LLM_BASE_URL", DEFAULT_E4B_BASE_URL).rstrip("/"),
        llm_model=os.getenv("ROBIT_LLM_MODEL", DEFAULT_E4B_MODEL),
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
        robot_llm_default_speed=_int_env("ROBIT_LLM_DEFAULT_SPEED", 170),
        robot_llm_max_duration_ms=_int_env("ROBIT_LLM_MAX_DURATION_MS", 1000),
        data_dir=data_dir,
        warm_models=_bool_env("ROBIT_WARM_MODELS", True),
        vision_enabled=_bool_env("ROBIT_VISION_ENABLED", True),
        vision_base_url=os.getenv("ROBIT_VISION_BASE_URL", DEFAULT_E4B_BASE_URL).rstrip("/"),
        vision_model=os.getenv(
            "ROBIT_VISION_MODEL",
            os.getenv("ROBIT_REALTIME_MODEL", DEFAULT_E4B_MODEL),
        ),
        vision_request_timeout_seconds=_float_env("ROBIT_VISION_REQUEST_TIMEOUT_SECONDS", 30.0),
        vision_max_output_tokens=_int_env("ROBIT_VISION_MAX_OUTPUT_TOKENS", 320),
        vision_image_tokens=_int_env("ROBIT_VISION_IMAGE_TOKENS", 140),
        vision_awareness_interval_seconds=_float_env("ROBIT_VISION_AWARENESS_INTERVAL_SECONDS", 5.0),
        vision_snapshot_ttl_seconds=_float_env("ROBIT_VISION_SNAPSHOT_TTL_SECONDS", 10.0),
        vision_world_window_seconds=_float_env("ROBIT_VISION_WORLD_WINDOW_SECONDS", 60.0),
        vision_change_threshold=_float_env("ROBIT_VISION_CHANGE_THRESHOLD", 0.03),
        vision_blur_threshold=_float_env("ROBIT_VISION_BLUR_THRESHOLD", 20.0),
        camera_frame_interval_seconds=_float_env("ROBIT_CAMERA_FRAME_INTERVAL_SECONDS", 5.0),
        camera_rotate_degrees=_int_env("ROBIT_CAMERA_ROTATE_DEGREES", 180),
        tracking_enabled=_bool_env("ROBIT_TRACKING_ENABLED", True),
        tracking_base_url=os.getenv("ROBIT_TRACKING_BASE_URL", "http://127.0.0.1:8091").rstrip("/"),
        tracking_request_timeout_seconds=_float_env("ROBIT_TRACKING_REQUEST_TIMEOUT_SECONDS", 2.0),
        tracking_confidence=_float_env("ROBIT_TRACKING_CONFIDENCE", 0.40),
        tracking_pan_sign=_int_env("ROBIT_TRACKING_PAN_SIGN", 1),
        tracking_tilt_sign=_int_env("ROBIT_TRACKING_TILT_SIGN", 1),
    )


settings = load_settings()

from pathlib import Path

from app.config import Settings
from app.llm import OllamaChatClient


def test_ollama_payload_disables_thinking():
    settings = Settings(
        robot_base_url="http://robot",
        request_timeout=2.0,
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_model="gemma4:e4b",
        llm_think=False,
        llm_timeout=30.0,
        stt_provider="faster_whisper",
        stt_model="base",
        stt_device="cuda",
        stt_compute_type="float16",
        tts_provider="xtts",
        tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
        tts_language="en",
        tts_device="cuda",
        voice_id="default",
        data_dir=Path("data"),
        warm_models=True,
    )

    payload = OllamaChatClient(settings)._payload("hello")

    assert payload["model"] == "gemma4:e4b"
    assert payload["think"] is False
    assert payload["stream"] is False

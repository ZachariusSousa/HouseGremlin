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
        tts_provider="chatterbox_turbo",
        tts_model="ResembleAI/chatterbox-turbo",
        tts_language="en",
        tts_device="cuda",
        tts_temperature=0.8,
        tts_top_p=0.95,
        tts_top_k=1000,
        tts_repetition_penalty=1.2,
        tts_norm_loudness=True,
        voice_id="default",
        data_dir=Path("data"),
        warm_models=True,
    )

    payload = OllamaChatClient(settings)._payload("hello")

    assert payload["model"] == "gemma4:e4b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 60
    assert payload["options"]["temperature"] == 0.4

from pathlib import Path

from app.config import Settings
from app.llm import OpenAICompatibleChatClient


def settings_for_test() -> Settings:
    return Settings(
        robot_base_url="http://robot",
        request_timeout=2.0,
        robot_request_retries=2,
        robot_retry_backoff_seconds=0.15,
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="gemma4:e4b",
        llm_think=False,
        llm_timeout=30.0,
        realtime_ws_url="ws://localhost:7861/v1/realtime",
        realtime_voice="serena",
        realtime_instructions="test realtime instructions",
        robot_llm_max_speed=180,
        robot_llm_max_duration_ms=1000,
        data_dir=Path("data"),
        warm_models=True,
    )


def test_openai_compatible_payload_disables_thinking():
    payload = OpenAICompatibleChatClient(settings_for_test())._payload("hello")

    assert payload["model"] == "gemma4:e4b"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 60
    assert payload["temperature"] == 0.4
    assert payload["think"] is False
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {"role": "user", "content": "hello"}


def test_openai_compatible_response_text_reads_chat_completions_shape():
    body = {"choices": [{"message": {"content": " hello robit "}}]}

    assert OpenAICompatibleChatClient._response_text(body) == "hello robit"

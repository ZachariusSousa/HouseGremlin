from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

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
        robot_llm_default_speed=170,
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


def install_scripted_http_client(monkeypatch, outcomes, requests):
    class ScriptedAsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, headers, json):
            requests.append({"url": url, "headers": headers, "json": json})
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr("app.llm.httpx.AsyncClient", ScriptedAsyncClient)


def chat_response(content, *, status_code=200, error=None):
    request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    body = error or {"choices": [{"message": {"content": content}}]}
    return httpx.Response(status_code, json=body, request=request)


@pytest.mark.anyio
async def test_action_chat_requests_strict_schema_with_bounded_robot_actions(
    monkeypatch,
):
    requests = []
    install_scripted_http_client(
        monkeypatch,
        [chat_response('{"response":"turning","action":null}')],
        requests,
    )

    result = await OpenAICompatibleChatClient(settings_for_test()).action_chat("turn left")

    assert result.response == '{"response":"turning","action":null}'
    response_format = requests[0]["json"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "robit_action"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"response", "action", "vision_question"}
    robot_action = schema["$defs"]["RobotActionRequest"]
    assert robot_action["additionalProperties"] is False
    assert set(robot_action["properties"]) == {"movement", "head", "eyes"}
    movement = schema["$defs"]["MovementAction"]
    assert movement["additionalProperties"] is False
    assert movement["properties"]["direction"]["enum"] == [
        "forward",
        "reverse",
        "left",
        "right",
        "stop",
    ]


@pytest.mark.anyio
async def test_action_chat_retries_once_without_schema_when_provider_rejects_capability(
    monkeypatch,
):
    requests = []
    install_scripted_http_client(
        monkeypatch,
        [
            chat_response(
                "",
                status_code=400,
                error={"error": {"message": "response_format json_schema is not supported"}},
            ),
            chat_response(
                '{"response":"turning","action":{"movement":{"direction":"left"}}}'
            ),
        ],
        requests,
    )

    result = await OpenAICompatibleChatClient(settings_for_test()).action_chat("turn left")

    assert result.response == (
        '{"response":"turning","action":{"movement":{"direction":"left"}}}'
    )
    assert len(requests) == 2
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"
    assert "response_format" not in requests[1]["json"]


@pytest.mark.anyio
async def test_action_chat_does_not_fallback_when_provider_reports_invalid_schema(
    monkeypatch,
):
    requests = []
    install_scripted_http_client(
        monkeypatch,
        [
            chat_response(
                "",
                status_code=400,
                error={
                    "error": {
                        "message": "response_format contains unsupported schema keyword anyOf"
                    }
                },
            )
        ],
        requests,
    )

    with pytest.raises(HTTPException) as caught:
        await OpenAICompatibleChatClient(settings_for_test()).action_chat("turn left")

    assert caught.value.status_code == 502
    assert len(requests) == 1


@pytest.mark.anyio
async def test_warmup_surfaces_timeout_without_retrying(monkeypatch):
    requests = []
    timeout_request = httpx.Request(
        "POST", "http://localhost:11434/v1/chat/completions"
    )
    install_scripted_http_client(
        monkeypatch,
        [httpx.ReadTimeout("model timed out", request=timeout_request)],
        requests,
    )

    with pytest.raises(HTTPException, match="model timed out") as caught:
        await OpenAICompatibleChatClient(settings_for_test()).warmup()

    assert caught.value.status_code == 502
    assert len(requests) == 1


@pytest.mark.anyio
async def test_action_chat_timeout_does_not_trigger_schema_fallback(monkeypatch):
    requests = []
    timeout_request = httpx.Request(
        "POST", "http://localhost:11434/v1/chat/completions"
    )
    install_scripted_http_client(
        monkeypatch,
        [httpx.ReadTimeout("action timed out", request=timeout_request)],
        requests,
    )

    with pytest.raises(HTTPException, match="action timed out") as caught:
        await OpenAICompatibleChatClient(settings_for_test()).action_chat("turn left")

    assert caught.value.status_code == 502
    assert len(requests) == 1
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (400, "generic bad request"),
        (401, "invalid bearer token"),
        (415, "response_format json_schema is not supported"),
        (500, "provider failed internally"),
        (501, "response_format json_schema is not supported"),
    ],
)
async def test_action_chat_does_not_fallback_for_non_capability_http_errors(
    monkeypatch,
    status_code,
    message,
):
    requests = []
    install_scripted_http_client(
        monkeypatch,
        [chat_response("", status_code=status_code, error={"error": message})],
        requests,
    )

    with pytest.raises(HTTPException) as caught:
        await OpenAICompatibleChatClient(settings_for_test()).action_chat("turn left")

    assert caught.value.status_code == 502
    assert len(requests) == 1
    assert requests[0]["json"]["response_format"]["type"] == "json_schema"

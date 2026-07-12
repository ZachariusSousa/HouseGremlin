import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app import main
from app.tts import SynthesisStreamEvent
from app.voices import VoiceStore


@dataclass
class FakeChatResult:
    response: str
    model: str


@dataclass
class FakeTranscriptionResult:
    text: str
    language: str | None = "en"
    duration_seconds: float | None = 1.0


@dataclass
class FakeSynthesisResult:
    audio_url: str
    audio_urls: list[str]
    voice_id: str
    spoken_text: str = "reply to hello"
    tts_input_chars: int = 14
    active_reference_count: int = 1


class FakeLlmClient:
    async def chat(self, text: str):
        return FakeChatResult(response=f"reply to {text}", model="gemma4:e4b")

    async def action_chat(self, text: str):
        return FakeChatResult(response='{"response":"ok","action":null}', model="gemma4:e4b")


class FakeActionLlmClient:
    def __init__(self, response: str):
        self.response = response

    async def action_chat(self, text: str):
        return FakeChatResult(response=self.response, model="gemma4:e4b")


class FakeTranscriber:
    def transcribe(self, path):
        return FakeTranscriptionResult(text="hello robit")


class FakeTts:
    def runtime_info(self):
        return {"provider": "chatterbox_streaming", "model_loaded": False}

    def synthesize(self, text: str, voice_id: str):
        return FakeSynthesisResult(
            audio_url="/audio/fake.wav",
            audio_urls=["/audio/fake.wav"],
            voice_id=voice_id,
            spoken_text=text,
            tts_input_chars=len(text),
        )

    def synthesize_stream(self, text: str, voice_id: str):
        yield SynthesisStreamEvent(
            event={
                "type": "chunk",
                "chunk_index": 1,
                "audio_url": "/audio/chunk.wav",
                "voice_id": voice_id,
                "first_latency_seconds": 1.25,
            }
        )
        yield SynthesisStreamEvent(
            event={
                "type": "final",
                "audio_url": "/audio/final.wav",
                "audio_urls": ["/audio/chunk.wav"],
                "voice_id": voice_id,
                "spoken_text": text,
                "tts_input_chars": len(text),
                "active_reference_count": 1,
                "total_chunks": 1,
            },
            final_result=FakeSynthesisResult(
                audio_url="/audio/final.wav",
                audio_urls=["/audio/chunk.wav"],
                voice_id=voice_id,
                spoken_text=text,
                tts_input_chars=len(text),
            ),
        )


class FakeVoiceStore:
    def list_voice_ids(self):
        return ["default", "narrator"]

    def sample_count(self, voice_id: str):
        return {"default": 1, "narrator": 2}[voice_id]


def test_voices_endpoint_lists_available_voices(monkeypatch):
    monkeypatch.setattr(main, "voice_store", FakeVoiceStore())

    response = TestClient(main.app).get("/voices")

    assert response.status_code == 200
    assert response.json()["voices"] == ["default", "narrator"]
    assert response.json()["voice_details"] == [
        {"voice_id": "default", "sample_count": 1},
        {"voice_id": "narrator", "sample_count": 2},
    ]
    assert response.json()["default_voice_id"] == "default"


def test_voice_store_counts_multiple_reference_clips(tmp_path):
    voice_dir = tmp_path / "voices" / "narrator"
    voice_dir.mkdir(parents=True)
    (voice_dir / "reference.wav").write_bytes(b"fake")
    (voice_dir / "reference-extra.wav").write_bytes(b"fake")
    (voice_dir / "notes.txt").write_text("ignored")

    store = VoiceStore(SimpleNamespace(voices_dir=tmp_path / "voices"))

    assert store.list_voice_ids() == ["narrator"]
    assert store.sample_count("narrator") == 2


def test_chat_endpoint_uses_configured_model(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())

    response = TestClient(main.app).post("/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["response"] == "reply to hello"


def test_robot_camera_urls_derive_from_configured_robot_base_url(monkeypatch):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(robot_base_url="http://172.22.1.126", request_timeout=2.0),
    )

    response = TestClient(main.app).get("/robot/camera")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "robot_base_url": "http://172.22.1.126",
        "page_url": "http://172.22.1.126/camera",
        "capture_url": "http://172.22.1.126/camera/capture",
        "stream_url": "http://172.22.1.126:81/stream",
    }


def test_robot_drive_uses_single_api_move_call(monkeypatch):
    calls = []

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "movement": body["direction"], "speed": body["speed"]}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/robot/drive", json={"move": "left", "speed": 90})

    assert response.status_code == 200
    assert calls == [("/api/move", {"direction": "left", "speed": 90})]
    assert response.json()["movement"] == "left"


def test_robot_head_uses_json_api_head_call(monkeypatch):
    calls = []

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "pan": body["pan"], "tilt": body["tilt"]}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/robot/head", json={"pan": 110, "tilt": 80})

    assert response.status_code == 200
    assert calls == [("/api/head", {"pan": 110, "tilt": 80})]
    assert response.json()["pan"] == 110
    assert response.json()["tilt"] == 80


def test_robot_request_retries_transient_http_errors(monkeypatch):
    calls = 0

    class FlakyRobotClient:
        async def request(self, method, url, params=None, json=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("temporary robot connection drop")
            return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot",
            request_timeout=2.0,
            robot_request_retries=1,
            robot_retry_backoff_seconds=0.0,
        ),
    )
    monkeypatch.setattr(main, "robot_http_client", FlakyRobotClient())

    response = TestClient(main.app).get("/robot/status")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert calls == 2


def test_robot_action_clamps_movement_speed_and_duration(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(robot_llm_max_speed=180, robot_llm_max_duration_ms=1000),
    )

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "path": path, "body": body}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post(
        "/robot/action",
        json={"movement": {"direction": "forward", "speed": 240, "duration_ms": 2000}},
    )

    assert response.status_code == 200
    assert calls == [("/api/move", {"direction": "forward", "speed": 180, "duration_ms": 1000})]
    assert response.json()["action"]["movement"] == {
        "direction": "forward",
        "speed": 180,
        "duration_ms": 1000,
    }


def test_robot_action_rejects_invalid_movement_direction():
    response = TestClient(main.app).post(
        "/robot/action",
        json={"movement": {"direction": "diagonal", "speed": 120, "duration_ms": 300}},
    )

    assert response.status_code == 422


def test_chat_action_executes_strict_json_action(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(robot_llm_max_speed=180, robot_llm_max_duration_ms=1000),
    )
    monkeypatch.setattr(
        main,
        "llm_client",
        FakeActionLlmClient(
            '{"response":"moving now","action":{"movement":{"direction":"left","speed":120,"duration_ms":300}}}'
        ),
    )

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/chat/action", json={"text": "move left"})

    assert response.status_code == 200
    assert response.json()["response"] == "moving now"
    assert response.json()["parse_error"] is None
    assert calls == [("/api/move", {"direction": "left", "speed": 120, "duration_ms": 300})]


def test_chat_action_normalizes_fractional_llm_speed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(robot_llm_max_speed=180, robot_llm_max_duration_ms=1000),
    )
    monkeypatch.setattr(
        main,
        "llm_client",
        FakeActionLlmClient(
            '{"response":"slow move","action":{"movement":{"direction":"forward","speed":0.3,"duration_ms":300.4}}}'
        ),
    )

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/chat/action", json={"text": "move slowly"})

    assert response.status_code == 200
    assert response.json()["parse_error"] is None
    assert calls == [("/api/move", {"direction": "forward", "speed": 54, "duration_ms": 300})]


def test_chat_action_invalid_robot_action_returns_safe_error(monkeypatch):
    called = False
    monkeypatch.setattr(
        main,
        "llm_client",
        FakeActionLlmClient(
            '{"response":"moving","action":{"movement":{"direction":"sideways","speed":120,"duration_ms":300}}}'
        ),
    )

    async def fake_robot_post(path, body=None):
        nonlocal called
        called = True
        return {"ok": True}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/chat/action", json={"text": "move sideways"})

    assert response.status_code == 200
    assert response.json()["action_result"] is None
    assert response.json()["parse_error"].startswith("LLM returned an invalid robot action")
    assert called is False


def test_chat_action_handles_chat_only_json(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeActionLlmClient('{"response":"hello","action":null}'))

    response = TestClient(main.app).post("/chat/action", json={"text": "say hello"})

    assert response.status_code == 200
    assert response.json()["response"] == "hello"
    assert response.json()["action_result"] is None
    assert response.json()["parse_error"] is None


def test_chat_action_invalid_json_does_not_move(monkeypatch):
    called = False
    monkeypatch.setattr(main, "llm_client", FakeActionLlmClient("plain text reply"))

    async def fake_robot_post(path, body=None):
        nonlocal called
        called = True
        return {"ok": True}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/chat/action", json={"text": "move maybe"})

    assert response.status_code == 200
    assert response.json()["response"] == "plain text reply"
    assert response.json()["action_result"] is None
    assert response.json()["parse_error"]
    assert called is False


def test_health_reports_chatterbox_runtime(monkeypatch):
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["tts_runtime"]["provider"] == "chatterbox_streaming"


def test_voice_roundtrip_endpoint_streams_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "transcriber", FakeTranscriber())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/voice/roundtrip",
        files={"audio": ("sample.wav", b"fake wav bytes", "audio/wav")},
        data={"voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "chunk"
    assert events[1]["type"] == "final"
    assert events[1]["transcript"] == "hello robit"
    assert events[1]["model"] == "gemma4:e4b"
    assert events[1]["audio_url"] == "/audio/final.wav"


def test_chat_speak_endpoint_streams_with_mocked_services(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/chat/speak",
        json={"text": "hello", "voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "response"
    assert events[0]["response"] == "reply to hello"
    assert events[0]["model"] == "gemma4:e4b"
    assert events[1]["type"] == "final"
    assert events[1]["response"] == "reply to hello"
    assert events[1]["model"] == "gemma4:e4b"
    assert events[1]["audio_url"] == "/audio/fake.wav"
    assert events[1]["voice_id"] == "default"
    assert events[1]["active_reference_count"] == 1


def test_voice_synthesize_endpoint_streams_with_mocked_tts(monkeypatch):
    monkeypatch.setattr(main, "tts", FakeTts())

    response = TestClient(main.app).post(
        "/voice/synthesize",
        json={"text": "hello", "voice_id": "default"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "chunk"
    assert events[0]["audio_url"] == "/audio/chunk.wav"
    assert events[1]["type"] == "final"
    assert events[1]["audio_url"] == "/audio/final.wav"
    assert events[1]["total_chunks"] == 1

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.brain_models import SceneSnapshot, WorldState
from app.frame_broker import CameraFrame
from app.vision import VisionQueryResult


@dataclass
class FakeChatResult:
    response: str
    model: str


class FakeLlmClient:
    async def chat(self, text: str):
        user_text = text.rsplit("User request: ", 1)[-1]
        return FakeChatResult(response=f"reply to {user_text}", model="gemma4:e4b")

    async def action_chat(self, text: str):
        return FakeChatResult(response='{"response":"ok","action":null}', model="gemma4:e4b")


class FakeActionLlmClient:
    def __init__(self, response: str):
        self.response = response

    async def action_chat(self, text: str):
        return FakeChatResult(response=self.response, model="gemma4:e4b")


class FakeVisionService:
    enabled = True
    unavailable_reason = None

    def __init__(self):
        now = datetime.now(timezone.utc)
        self.snapshot = SceneSnapshot(
            frame_id="frame-vision",
            observed_at=now,
            trigger="explicit",
            summary="A chair is visible.",
            entities=[],
            novelty=1.0,
            uncertainty=0.1,
            model="fake/vision",
            latency_ms=10.0,
            expires_at=now + timedelta(seconds=30),
        )

    async def query(self, question, fresh=True):
        return VisionQueryResult(self.snapshot, fresh)

    def current_snapshot(self):
        return self.snapshot

    def world_state(self):
        return WorldState(snapshot_ids=[self.snapshot.frame_id], summary=self.snapshot.summary)

    def latest_payload(self):
        return {
            "available": True,
            "enabled": True,
            "reason": None,
            "stale": False,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "world_state": self.world_state().model_dump(mode="json"),
        }


def test_operator_console_is_not_browser_cached():
    response = TestClient(main.app).get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_text_model_prompt_always_contains_live_visual_context(monkeypatch):
    vision = FakeVisionService()
    monkeypatch.setattr(main, "vision_service", vision)

    prompt = main.prompt_with_live_scene("Hello Robit")

    assert "LIVE VISUAL CONTEXT" in prompt
    assert "A chair is visible." in prompt
    assert "User request: Hello Robit" in prompt


def test_chat_endpoint_uses_configured_model(monkeypatch):
    monkeypatch.setattr(main, "llm_client", FakeLlmClient())

    response = TestClient(main.app).post("/chat", json={"text": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "gemma4:e4b"
    assert response.json()["response"] == "reply to hello"


def test_health_reports_realtime_config(monkeypatch):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot",
            llm_provider="openai_compatible",
            llm_base_url="http://localhost:11434/v1",
            llm_model="gemma4:e4b",
            realtime_ws_url="ws://localhost:7861/v1/realtime",
            realtime_voice="serena",
            realtime_instructions="test instructions",
        ),
    )

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["realtime"] == {
        "ws_url": "ws://testserver/v1/realtime",
        "gateway_path": "/v1/realtime",
        "voice": "serena",
        "instructions": f"test instructions {main.REALTIME_EYE_POLICY}",
    }
    assert "tts_runtime" not in response.json()


def test_removed_voice_endpoints_are_not_registered():
    client = TestClient(main.app)

    assert client.get("/voices").status_code == 404
    assert client.post("/voices").status_code == 404
    assert client.post("/voice/transcribe").status_code == 404
    assert client.post("/voice/synthesize", json={"text": "hello"}).status_code == 404
    assert client.post("/voice/roundtrip").status_code == 404
    assert client.post("/chat/speak", json={"text": "hello"}).status_code == 404


def test_robot_camera_urls_derive_from_configured_robot_base_url(monkeypatch):
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://172.22.1.126",
            request_timeout=2.0,
            camera_frame_interval_seconds=5.0,
        ),
    )

    response = TestClient(main.app).get("/robot/camera")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "robot_base_url": "http://172.22.1.126",
        "page_url": "http://172.22.1.126/camera",
        "capture_url": "http://172.22.1.126:81/capture",
        "stream_url": "http://172.22.1.126:81/stream",
        "frame_interval_seconds": 5.0,
    }


def test_camera_capture_returns_shared_frame_headers(monkeypatch):
    frame = CameraFrame("frame-1", datetime.now(timezone.utc), b"jpeg-data")

    class FakeBroker:
        async def get_frame(self, force_fresh=False):
            return frame

    monkeypatch.setattr(main, "frame_broker", FakeBroker())
    response = TestClient(main.app).get("/robot/camera/capture")

    assert response.status_code == 200
    assert response.content == b"jpeg-data"
    assert response.headers["x-robit-frame-id"] == "frame-1"


@pytest.mark.anyio
async def test_frame_broker_fetches_from_dedicated_camera_server(monkeypatch):
    calls = []

    async def fake_fetch(path, base_url=None):
        calls.append((path, base_url))
        return b"jpeg", "image/jpeg"

    monkeypatch.setattr(main, "robot_fetch_bytes", fake_fetch)

    content, media_type = await main.fetch_robot_camera_frame()

    assert content == b"jpeg"
    assert media_type == "image/jpeg"
    expected_base = main.camera_urls()["capture_url"].removesuffix("/capture")
    assert calls == [("/capture", expected_base)]


def test_perception_endpoints_return_typed_snapshot(monkeypatch):
    fake = FakeVisionService()
    monkeypatch.setattr(main, "vision_service", fake)

    latest = TestClient(main.app).get("/perception/latest")
    query = TestClient(main.app).post("/perception/query", json={"question": "What do you see?", "fresh": True})

    assert latest.status_code == 200
    assert latest.json()["snapshot"]["summary"] == "A chair is visible."
    assert query.status_code == 200
    assert query.json()["fresh"] is True


def test_perception_query_returns_503_when_vision_has_no_result(monkeypatch):
    class UnavailableVisionService(FakeVisionService):
        async def query(self, question, fresh=True):
            from app.vision import VisionUnavailable

            raise VisionUnavailable("multimodal projector unavailable")

    monkeypatch.setattr(main, "vision_service", UnavailableVisionService())

    response = TestClient(main.app).post(
        "/perception/query",
        json={"question": "What do you see?", "fresh": True},
    )

    assert response.status_code == 503
    assert "multimodal projector unavailable" in response.json()["detail"]


def test_text_visual_question_never_executes_action(monkeypatch):
    fake_vision = FakeVisionService()
    fake_llm = FakeActionLlmClient(
        '{"response":"I will look.","vision_question":"What do you see?",'
        '"action":{"movement":{"direction":"forward"}}}'
    )

    async def grounded_chat(text):
        return FakeChatResult(response="I can see a chair.", model="gemma4:e4b")

    fake_llm.chat = grounded_chat
    monkeypatch.setattr(main, "vision_service", fake_vision)
    monkeypatch.setattr(main, "llm_client", fake_llm)
    response = TestClient(main.app).post("/chat/action", json={"text": "What do you see?"})

    assert response.status_code == 200
    assert response.json()["response"] == "I can see a chair."
    assert response.json()["action"] is None
    assert response.json()["vision"]["snapshot"]["frame_id"] == "frame-vision"


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


def test_robot_action_executes_supported_eye_expression(monkeypatch):
    calls = []

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "path": path, "body": body}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post(
        "/robot/action",
        json={"eyes": {"expression": "cute", "duration_ms": 1500}},
    )

    assert response.status_code == 200
    assert calls == [("/api/eyes", {"expression": "cute", "duration_ms": 1500})]


def test_robot_action_rejects_unknown_eye_expression():
    response = TestClient(main.app).post(
        "/robot/action",
        json={"eyes": {"expression": "laser_beams"}},
    )

    assert response.status_code == 422


def test_model_eye_policy_rejects_operational_expressions():
    with pytest.raises(ValueError, match="operational"):
        main.validate_action_payload({"eyes": {"expression": "fault"}})

    assert main.validate_action_payload({"eyes": {"expression": "happy"}}) == {
        "eyes": {"expression": "happy"},
        "emergency_stop": False,
    }


@pytest.mark.anyio
async def test_model_eye_action_queues_mood_without_waiting_for_firmware(monkeypatch):
    selected = []

    class FakeEyeController:
        coordinator = main.get_brain_coordinator()

        def select_mood(self, expression, duration_ms, source, correlation_id):
            selected.append((expression, duration_ms, source, correlation_id))
            return self.coordinator.state.eyes

    monkeypatch.setattr(main, "eye_controller", FakeEyeController())

    result = await main.execute_voice_model_action_payload(
        {"eyes": {"expression": "cute", "duration_ms": 1200}}
    )

    assert result["executed"][0]["type"] == "eyes.mood"
    assert result["executed"][0]["queued"] is True
    assert selected[0][:3] == ("cute", 1200, main.EventSource.voice_model)


def test_robot_action_rejects_empty_and_unknown_fields():
    client = TestClient(main.app)
    assert client.post("/robot/action", json={}).status_code == 422
    assert client.post("/robot/action", json={"movement": {"direction": "left", "surprise": True}}).status_code == 422


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

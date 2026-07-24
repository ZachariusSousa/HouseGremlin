from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.brain_models import SceneSnapshot, WorldState
from app.frame_broker import CameraFrame
from app.tracking import TrackingStatus
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


class FakeTrackingDetector:
    available = True
    reason = None

    async def probe(self):
        return True


class FakeTrackingService:
    detector = FakeTrackingDetector()

    def __init__(self):
        self.mode = "track"
        self.head = (90, 90)

    def status(self):
        return TrackingStatus(
            available=True,
            enabled=self.mode != "off",
            state="searching" if self.mode == "track" else "off",
            mode=self.mode,
            head={"pan": self.head[0], "tilt": self.head[1]},
            effective_camera_fps=3.0 if self.mode == "track" else 0.2,
            backend="cuda/bfloat16",
            model="Roboflow/rf-detr-nano",
        )

    async def enable(self):
        self.mode = "track"
        return self.status()

    async def stop(self, reason="requested"):
        self.mode = "off"
        return self.status()

    def suspend_for_manual_control(self, *args, **kwargs):
        return None

    def sync_head_position(self, pan, tilt):
        self.head = (
            self.head[0] if pan is None else pan,
            self.head[1] if tilt is None else tilt,
        )

    def emergency_disable(self):
        self.mode = "off"


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
        "effective_fps": 0.2,
    }


def test_camera_capture_returns_shared_frame_headers(monkeypatch):
    frame = CameraFrame("frame-1", datetime.now(timezone.utc), b"jpeg-data")
    calls = []

    class FakeBroker:
        async def get_frame(self, force_fresh=False):
            calls.append(force_fresh)
            return frame

    monkeypatch.setattr(main, "frame_broker", FakeBroker())
    response = TestClient(main.app).get("/robot/camera/capture?fresh=true")

    assert response.status_code == 200
    assert response.content == b"jpeg-data"
    assert response.headers["x-robit-frame-id"] == "frame-1"
    assert calls == [True]


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


def test_tracking_api_and_deterministic_text_command(monkeypatch):
    tracking = FakeTrackingService()
    monkeypatch.setattr(main, "tracking_service", tracking)
    client = TestClient(main.app)

    initial = client.get("/tracking/status")
    stopped = client.post("/tracking/stop")
    command = client.post("/chat/action", json={"text": "Track me"})
    stopped_by_text = client.post("/chat/action", json={"text": "Stop tracking"})
    stopped_by_negative_request = client.post("/chat/action", json={"text": "Please don't track me"})
    restarted_by_text = client.post("/chat/action", json={"text": "Start tracking me"})
    stopped_again = client.post("/chat/action", json={"text": "Stop looking at me"})
    negated_start = client.post("/chat/action", json={"text": "Don't start tracking me"})
    negated_switch = client.post("/chat/action", json={"text": "Don't turn tracking on"})
    restarted_by_switch = client.post("/chat/action", json={"text": "Turn tracking on"})
    started = client.post("/tracking/start")

    assert initial.status_code == 200
    assert initial.json()["mode"] == "track"
    assert initial.json()["effective_camera_fps"] == 3.0
    assert stopped.status_code == 200
    assert stopped.json()["mode"] == "off"
    assert started.status_code == 200
    assert started.json()["mode"] == "track"
    assert command.status_code == 200
    assert command.json()["model"] == "deterministic-tracking"
    assert command.json()["action_result"]["command"] == "track"
    assert stopped_by_text.status_code == 200
    assert stopped_by_text.json()["action_result"]["command"] == "off"
    assert stopped_by_text.json()["action_result"]["status"]["mode"] == "off"
    assert stopped_by_negative_request.status_code == 200
    assert stopped_by_negative_request.json()["action_result"]["command"] == "off"
    assert restarted_by_text.status_code == 200
    assert restarted_by_text.json()["action_result"]["command"] == "track"
    assert stopped_again.status_code == 200
    assert stopped_again.json()["action_result"]["command"] == "off"
    assert negated_start.status_code == 200
    assert negated_start.json()["action_result"]["command"] == "off"
    assert negated_switch.status_code == 200
    assert negated_switch.json()["action_result"]["command"] == "off"
    assert restarted_by_switch.status_code == 200
    assert restarted_by_switch.json()["action_result"]["command"] == "track"


def test_combined_tracking_and_motor_stop_prioritizes_emergency_stop(monkeypatch):
    tracking = FakeTrackingService()
    monkeypatch.setattr(main, "tracking_service", tracking)
    calls = []

    async def fake_emergency_stop():
        calls.append("emergency-stop")
        return {"ok": True}

    monkeypatch.setattr(main, "robot_emergency_stop_request", fake_emergency_stop)

    response = TestClient(main.app).post(
        "/chat/action",
        json={"text": "Stop tracking and stop moving"},
    )

    assert response.status_code == 200
    assert response.json()["model"] == "deterministic-safety"
    assert response.json()["action"] == {"emergency_stop": True}
    assert calls == ["emergency-stop"]
    assert tracking.mode == "off"


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
    tracking = FakeTrackingService()
    monkeypatch.setattr(main, "tracking_service", tracking)

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "pan": body["pan"], "tilt": body["tilt"]}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post("/robot/head", json={"pan": 110, "tilt": 80})

    assert response.status_code == 200
    assert calls == [("/api/head", {"pan": 110, "tilt": 80})]
    assert response.json()["pan"] == 110
    assert response.json()["tilt"] == 80
    assert tracking.head == (110, 80)


def test_relative_head_action_resyncs_tracking_from_robot_response(monkeypatch):
    tracking = FakeTrackingService()
    monkeypatch.setattr(main, "tracking_service", tracking)

    async def fake_robot_post(path, body=None):
        assert path == "/api/head"
        assert body == {"pan_delta": 15, "tilt_delta": -5}
        return {"ok": True, "pan": 105, "tilt": 85}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post(
        "/robot/action",
        json={"head": {"pan_delta": 15, "tilt_delta": -5}},
    )

    assert response.status_code == 200
    assert tracking.head == (105, 85)


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


@pytest.mark.anyio
async def test_transport_rechecks_tracking_authorization_after_waiting_for_lock(monkeypatch):
    lock = asyncio.Lock()
    monkeypatch.setattr(main, "robot_request_lock", lock)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot",
            request_timeout=2.0,
            robot_request_retries=0,
            robot_retry_backoff_seconds=0.0,
        ),
    )
    authorized = True
    calls = []

    class FakeRobotClient:
        is_closed = False

        async def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(main, "robot_http_client", FakeRobotClient())
    await lock.acquire()

    pending = asyncio.create_task(
        main.robot_post(
            "/api/move",
            {"direction": "right", "speed": 70, "duration_ms": 150},
            authorization=lambda: authorized,
        )
    )
    await asyncio.sleep(0)
    authorized = False
    lock.release()

    assert await pending == {"ok": False, "skipped": "tracking authorization expired"}
    assert calls == []


@pytest.mark.anyio
async def test_slow_camera_fetch_does_not_block_robot_control_requests(monkeypatch):
    control_lock = asyncio.Lock()
    camera_lock = asyncio.Lock()
    monkeypatch.setattr(main, "robot_request_lock", control_lock)
    monkeypatch.setattr(main, "robot_camera_request_lock", camera_lock)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot",
            request_timeout=2.0,
            robot_request_retries=0,
            robot_retry_backoff_seconds=0.0,
        ),
    )
    calls = []

    class FakeRobotClient:
        is_closed = False

        async def get(self, url):
            calls.append(("GET", url))
            return httpx.Response(
                200,
                content=b"jpeg",
                headers={"content-type": "image/jpeg"},
                request=httpx.Request("GET", url),
            )

        async def request(self, method, url, **kwargs):
            calls.append((method, url))
            return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(main, "robot_http_client", FakeRobotClient())
    await camera_lock.acquire()
    pending_camera = asyncio.create_task(main.robot_fetch_bytes("/capture", "http://robot:81"))
    await asyncio.sleep(0)

    control_result = await asyncio.wait_for(
        main.robot_post("/api/head", {"pan": 95, "tilt": 90}),
        timeout=0.1,
    )

    assert control_result == {"ok": True}
    assert calls == [("POST", "http://robot/api/head")]
    camera_lock.release()
    assert await pending_camera == (b"jpeg", "image/jpeg")


@pytest.mark.anyio
async def test_continuous_tracking_head_command_uses_direct_control_path(monkeypatch):
    calls = []

    class AuthorizedTracking:
        def motion_authorized(self, generation, *, body, allow_search=False):
            return generation == 7 and not body and not allow_search

    async def fake_robot_post(path, body=None, authorization=None):
        assert authorization is not None and authorization()
        calls.append((path, body))
        return {"ok": True, "pan": body["pan"], "tilt": body["tilt"]}

    monkeypatch.setattr(main, "tracking_service", AuthorizedTracking())
    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    result = await main.tracking_head_command(108, 76, 7)

    assert result == {"ok": True, "pan": 108, "tilt": 76}
    assert calls == [("/api/head", {"pan": 108, "tilt": 76})]


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


def test_robot_action_uses_170_default_when_speed_is_omitted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_llm_max_speed=180,
            robot_llm_default_speed=170,
            robot_llm_max_duration_ms=1000,
        ),
    )

    async def fake_robot_post(path, body=None):
        calls.append((path, body))
        return {"ok": True, "path": path, "body": body}

    monkeypatch.setattr(main, "robot_post", fake_robot_post)

    response = TestClient(main.app).post(
        "/robot/action",
        json={"movement": {"direction": "forward", "duration_ms": 700}},
    )

    assert response.status_code == 200
    assert calls == [("/api/move", {"direction": "forward", "speed": 170, "duration_ms": 700})]


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

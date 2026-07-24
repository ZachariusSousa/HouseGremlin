from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.brain_models import ConversationState, SceneEntity
from app.config import Settings
from app.coordinator import BrainCoordinator
from app.frame_broker import CameraFrame, FrameBroker
from app.journal import EventJournal
from app.vision import (
    FrameQuality,
    LlamaServerVisionAdapter,
    VisionOutput,
    VisionService,
    VisionUnavailable,
)


def vision_settings(tmp_path: Path) -> Settings:
    return Settings(
        robot_base_url="http://robot",
        request_timeout=2.0,
        robot_request_retries=0,
        robot_retry_backoff_seconds=0.0,
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:11434/v1",
        llm_model="gemma4:e4b",
        llm_think=False,
        llm_timeout=30.0,
        realtime_ws_url="ws://localhost:7861/v1/realtime",
        realtime_voice="serena",
        realtime_instructions="instructions",
        robot_llm_max_speed=180,
        robot_llm_default_speed=170,
        robot_llm_max_duration_ms=1000,
        data_dir=tmp_path,
        warm_models=False,
        vision_enabled=True,
    )


@pytest.mark.anyio
async def test_frame_broker_shares_one_frame_inside_interval():
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        return f"jpeg-{calls}".encode(), "image/jpeg"

    broker = FrameBroker(fetch, interval_seconds=0.01)
    first = await broker.get_frame()
    second = await broker.get_frame()

    assert first.frame_id == second.frame_id
    assert calls == 1
    fresh = await broker.get_frame(force_fresh=True)
    assert fresh.frame_id != first.frame_id
    assert calls == 2


@pytest.mark.anyio
async def test_frame_broker_rate_leases_raise_and_restore_shared_rate():
    async def fetch():
        return b"jpeg", "image/jpeg"

    broker = FrameBroker(fetch, interval_seconds=5.0, max_fps=3.0)
    assert broker.effective_fps == pytest.approx(0.2)
    assert broker.effective_interval_seconds == pytest.approx(5.0)

    broker.set_rate_lease("conversation", 2.0)
    broker.set_rate_lease("tracking", 3.0)
    assert broker.effective_fps == pytest.approx(3.0)

    broker.release_rate_lease("tracking")
    assert broker.effective_fps == pytest.approx(2.0)
    broker.release_rate_lease("conversation")
    assert broker.effective_fps == pytest.approx(0.2)


class FakeAdapter:
    model_name = "fake/vision"

    def __init__(self):
        self.fail = False

    async def infer(self, image, question):
        if self.fail:
            raise ValueError("bad model output")
        return VisionOutput(
            summary="A chair is visible.",
            entities=[SceneEntity(label="chair", confidence=0.9)],
            uncertainty=0.1,
        )


@pytest.mark.anyio
async def test_explicit_vision_query_creates_typed_snapshot_and_cached_fallback(monkeypatch, tmp_path):
    frames = 0

    async def fetch():
        nonlocal frames
        frames += 1
        return b"jpeg", "image/jpeg"

    broker = FrameBroker(fetch, interval_seconds=0.001)
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    adapter = FakeAdapter()
    service = VisionService(vision_settings(tmp_path), coordinator, broker, adapter)
    observed = []
    service.subscribe_snapshot(observed.append)
    monkeypatch.setattr(
        "app.vision.inspect_frame",
        lambda *args, **kwargs: FrameQuality(object(), object(), 100.0, 1.0, False, True),
    )

    result = await service.query("What do you see?", fresh=True)
    assert result.fresh is True
    assert result.snapshot.summary == "A chair is visible."
    assert observed == [result.snapshot]
    assert service.world_state().entities[0].label == "chair"
    assert any(event.event_type == "perception.snapshot.created" for event in coordinator.journal.list_events())

    adapter.fail = True
    fallback = await service.query("What changed?", fresh=True)
    assert fallback.fresh is False
    assert fallback.snapshot.frame_id == result.snapshot.frame_id
    assert "bad model output" in fallback.warning

    result.snapshot.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert service.world_state().summary == "unknown"
    assert service.world_state().entities == []
    with pytest.raises(VisionUnavailable, match="bad model output"):
        await service.query("What is visible now?", fresh=True)


@pytest.mark.anyio
async def test_unchanged_frame_refreshes_snapshot_and_notifies_voice_context(monkeypatch, tmp_path):
    async def fetch():
        return b"jpeg", "image/jpeg"

    broker = FrameBroker(fetch, interval_seconds=0.001)
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    service = VisionService(vision_settings(tmp_path), coordinator, broker, FakeAdapter())
    observed = []
    service.subscribe_snapshot(observed.append)
    quality = FrameQuality(object(), object(), 100.0, 0.0, False, False)
    first_frame = CameraFrame("frame-first", datetime.now(timezone.utc), b"jpeg")
    first = await service._infer(first_frame, quality, "Describe it", "awareness")
    next_frame = CameraFrame("frame-next", datetime.now(timezone.utc), b"jpeg")

    assert await service._carry_forward_snapshot(next_frame, quality) is True

    assert service.latest.frame_id == "frame-next"
    assert service.latest.summary == first.summary
    assert service.latest.latency_ms == 0.0
    assert observed == [first, service.latest]


def test_scene_entity_rejects_non_normalized_bounding_box():
    with pytest.raises(ValueError):
        SceneEntity(label="chair", confidence=0.9, bounding_box=(0.0, 0.0, 2.0, 1.0))


def test_awareness_pauses_while_conversation_is_active(tmp_path):
    async def fetch():
        return b"jpeg", "image/jpeg"

    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    service = VisionService(vision_settings(tmp_path), coordinator, FrameBroker(fetch), FakeAdapter())
    service._last_awareness_at = 10.0

    assert service.awareness_ready(now=15.0) is True
    coordinator.transition("corr-speaking", conversation=ConversationState.speaking)
    assert service.awareness_ready(now=20.0) is False


def test_inspect_frame_rejects_unchanged_and_blurred_frames():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from io import BytesIO

    from app.vision import inspect_frame

    image = Image.new("RGB", (320, 240), color="gray")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    first = inspect_frame(buffer.getvalue(), None, 180, 0.03, 20.0)
    second = inspect_frame(buffer.getvalue(), first.preview, 180, 0.03, 20.0)

    assert first.blurred is True
    assert second.changed is False
    assert np.array_equal(first.preview, second.preview)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("request failed", request=None, response=None)

    def json(self):
        return self.payload


class FakeAsyncClient:
    def __init__(self, responses, requests, **kwargs):
        self.responses = responses
        self.requests = requests

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        self.requests.append(("GET", url, None))
        return self.responses.pop(0)

    async def post(self, url, json):
        self.requests.append(("POST", url, json))
        return self.responses.pop(0)


@pytest.mark.anyio
async def test_llama_server_adapter_probes_and_sends_schema_and_base64_image(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    requests = []
    responses = [
        FakeResponse({
            "data": [{"id": "e4b"}],
            "models": [{"name": "e4b", "capabilities": ["completion", "multimodal"]}],
        }),
        FakeResponse({
            "choices": [{
                "message": {
                    "content": '{"summary":"A chair.","entities":[],"uncertainty":0.1}'
                }
            }]
        }),
    ]
    monkeypatch.setattr(
        "app.vision.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, requests, **kwargs),
    )
    adapter = LlamaServerVisionAdapter("http://localhost:8081/v1/", "e4b", 5.0, 123)

    await adapter.probe()
    output = await adapter.infer(Image.new("RGB", (8, 8), "white"), "What is visible?")

    assert output.summary == "A chair."
    assert requests[0][:2] == ("GET", "http://localhost:8081/v1/models")
    request = requests[1][2]
    assert request["max_tokens"] == 123
    assert request["cache_prompt"] is False
    assert request["response_format"]["type"] == "json_schema"
    assert "stateless visual parser" in request["messages"][0]["content"]
    assert "Robit camera frame" not in str(request["messages"])
    image_url = request["messages"][1]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")


@pytest.mark.anyio
async def test_llama_server_adapter_rejects_missing_multimodal_capability(monkeypatch):
    responses = [FakeResponse({"data": [{"id": "e4b", "capabilities": {"vision": False}}]})]
    monkeypatch.setattr(
        "app.vision.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, [], **kwargs),
    )
    adapter = LlamaServerVisionAdapter("http://localhost:8081/v1", "e4b", 5.0, 123)

    with pytest.raises(VisionUnavailable, match="without multimodal"):
        await adapter.probe()


@pytest.mark.anyio
async def test_llama_server_adapter_rejects_malformed_structured_output(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    responses = [FakeResponse({"choices": [{"message": {"content": "not json"}}]})]
    monkeypatch.setattr(
        "app.vision.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(responses, [], **kwargs),
    )
    adapter = LlamaServerVisionAdapter("http://localhost:8081/v1", "e4b", 5.0, 123)

    with pytest.raises(ValueError):
        await adapter.infer(Image.new("RGB", (8, 8), "white"), "What is visible?")


@pytest.mark.anyio
async def test_vision_service_disables_cleanly_when_probe_fails(tmp_path):
    class UnavailableAdapter(FakeAdapter):
        async def probe(self):
            raise VisionUnavailable("no multimodal projector")

    async def fetch():
        return b"jpeg", "image/jpeg"

    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    service = VisionService(vision_settings(tmp_path), coordinator, FrameBroker(fetch), UnavailableAdapter())

    await service.start()

    assert service.enabled is False
    assert service.latest_payload()["reason"] == "no multimodal projector"
    with pytest.raises(VisionUnavailable, match="no multimodal projector"):
        await service.query("What is visible?")

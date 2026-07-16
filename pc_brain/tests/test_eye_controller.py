import asyncio

import pytest

from app.brain_models import BodyState, ConversationState, EventSource
from app.coordinator import BrainCoordinator
from app.eye_controller import EyeController
from app.journal import EventJournal


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_operational_states_override_and_restore_selected_mood(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    calls = []

    async def post(path, body):
        calls.append((path, body))
        return {"ok": True}

    controller = EyeController(coordinator, post, heartbeat_interval_seconds=None)
    await controller.start()
    await controller._queue.join()
    calls.clear()
    controller.set_voice_session_active(True)
    coordinator.transition("corr-eyes", EventSource.browser, conversation=ConversationState.listening)
    controller.select_mood("happy", 10000, EventSource.voice_model, "corr-eyes")
    assert coordinator.state.eyes.base_expires_at is None
    coordinator.transition("corr-eyes", EventSource.voice_model, conversation=ConversationState.formulating)
    coordinator.transition("corr-eyes", EventSource.voice_model, conversation=ConversationState.speaking)
    coordinator.transition("corr-eyes", EventSource.voice_model, conversation=ConversationState.idle)
    await controller._queue.join()

    assert coordinator.state.eyes.base_expression == "happy"
    assert coordinator.state.eyes.effective_expression == "happy"
    assert coordinator.state.eyes.base_expires_at is not None
    assert calls[-1] == ("/api/eyes", {"expression": "happy", "duration_ms": 0})
    events = coordinator.journal.list_events(correlation_id="corr-eyes")
    changed = [event.payload["expression"] for event in events if event.event_type == "eyes.expression.changed"]
    assert changed == ["listening", "thinking", "speaking", "happy"]
    await controller.shutdown()


@pytest.mark.anyio
async def test_fault_has_priority_and_recovery_restores_mood(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))

    async def post(path, body):
        return {"ok": True}

    controller = EyeController(coordinator, post, heartbeat_interval_seconds=None)
    await controller.start()
    controller.select_mood("content", 10000, EventSource.text_model, "corr-fault")
    controller.set_server_fault("realtime_upstream", True, "corr-fault")
    assert coordinator.state.eyes.effective_expression == "fault"

    coordinator.transition("corr-fault", EventSource.system, body=BodyState.fault, safety="fault")
    controller.set_server_fault("realtime_upstream", False, "corr-fault")
    assert coordinator.state.eyes.effective_expression == "fault"

    coordinator.transition("corr-fault", EventSource.system, body=BodyState.stationary, safety="normal")
    assert coordinator.state.eyes.effective_expression == "content"
    await controller.shutdown()


@pytest.mark.anyio
async def test_mood_expires_to_neutral(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))

    async def post(path, body):
        return {"ok": True}

    controller = EyeController(
        coordinator,
        post,
        maximum_mood_duration_ms=100,
        heartbeat_interval_seconds=None,
    )
    await controller.start()
    controller.select_mood("cute", 20, EventSource.text_model, "corr-expiry")
    assert coordinator.state.eyes.effective_expression == "cute"
    await asyncio.sleep(0.04)

    assert coordinator.state.eyes.base_expression == "neutral"
    assert coordinator.state.eyes.effective_expression == "neutral"
    assert any(event.event_type == "eyes.mood.expired" for event in coordinator.journal.list_events())
    await controller.shutdown()


@pytest.mark.anyio
async def test_duplicate_effective_state_is_not_resent(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    calls = []

    async def post(path, body):
        calls.append(body["expression"])
        return {"ok": True}

    controller = EyeController(coordinator, post, heartbeat_interval_seconds=None)
    await controller.start()
    await controller._queue.join()
    calls.clear()
    controller.set_voice_session_active(True)

    coordinator.transition("corr-dedupe", EventSource.browser, conversation=ConversationState.listening)
    coordinator.transition("corr-dedupe", EventSource.browser, conversation=ConversationState.listening)
    await controller._queue.join()

    assert calls == ["listening"]
    await controller.shutdown()


@pytest.mark.anyio
async def test_firmware_heartbeat_recovery_forces_resync(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    heartbeat_seen = asyncio.Event()
    calls = []

    async def post(path, body):
        calls.append((path, body))
        return {"ok": True}

    async def heartbeat(path, body):
        heartbeat_seen.set()
        return {"ok": True, "heartbeat_recovered": True}

    controller = EyeController(
        coordinator,
        post,
        heartbeat_post=heartbeat,
        heartbeat_interval_seconds=60,
    )
    await controller.start()
    await asyncio.wait_for(heartbeat_seen.wait(), 1)
    await controller._queue.join()

    assert calls[-1] == ("/api/eyes", {"expression": "neutral", "duration_ms": 0})
    assert any(event.event_type == "eyes.heartbeat.recovered" for event in coordinator.journal.list_events())
    await controller.shutdown()


def test_model_cannot_select_operational_expression(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))

    async def post(path, body):
        return {"ok": True}

    controller = EyeController(coordinator, post, heartbeat_interval_seconds=None)
    with pytest.raises(ValueError, match="operational"):
        controller.select_mood("fault", 1000, EventSource.voice_model, "corr-rejected")
    assert any(event.event_type == "eyes.mood.rejected" for event in coordinator.journal.list_events())

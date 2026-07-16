import json
import asyncio
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from app.brain_models import ConversationState, EventSource
from app.coordinator import BrainCoordinator
from app.journal import EventJournal
from app.realtime_gateway import RealtimeGateway, explicit_eye_expression


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeUpstream:
    def __init__(self):
        self.sent = []

    async def send(self, value):
        self.sent.append(json.loads(value))


class FakeBrowser:
    def __init__(self, block=False):
        self.sent = []
        self.accepted = False
        self.closed = None
        self.block = block
        self.release = asyncio.Event()

    async def accept(self):
        self.accepted = True

    async def send_json(self, value):
        self.sent.append(value)

    async def receive_json(self):
        if self.block:
            await self.release.wait()
        raise WebSocketDisconnect()

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)
        self.release.set()


@pytest.mark.anyio
async def test_voice_tool_executes_in_gateway_once(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    executed = []

    async def execute(action):
        executed.append(action)
        return {"ok": True, "action": action}

    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        execute,
    )
    upstream = FakeUpstream()
    gateway._upstream = upstream
    event = {
        "type": "response.function_call_arguments.done",
        "call_id": "call_1",
        "name": "robot_action",
        "arguments": '{"movement":{"direction":"left"}}',
    }

    assert await gateway._handle_upstream_event(event) is False
    assert await gateway._handle_upstream_event(event) is False
    assert executed == [{"movement": {"direction": "left"}}]
    assert upstream.sent[-2]["item"]["type"] == "function_call_output"
    assert upstream.sent[-1] == {"type": "response.create"}


def test_explicit_eye_expression_only_matches_commands():
    assert explicit_eye_expression("Can you show me you're happy?") == "happy"
    assert explicit_eye_expression("Give me sleepy eyes") == "sleepy"
    assert explicit_eye_expression("Try the embarrassed expression") == "cute"
    assert explicit_eye_expression("Are you happy?") is None
    assert explicit_eye_expression("Try it again", "concerned") == "concerned"


@pytest.mark.anyio
async def test_explicit_voice_eye_command_executes_without_model_tool(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    executed = []

    async def execute(action):
        executed.append(action)
        return {"ok": True}

    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        execute,
    )
    gateway._upstream = FakeUpstream()

    await gateway._handle_upstream_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Can you show me happy eyes?",
        }
    )
    await gateway._handle_upstream_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Try it again",
        }
    )

    assert executed == [
        {"eyes": {"expression": "happy"}},
        {"eyes": {"expression": "happy"}},
    ]
    assert any(event.payload.get("reason") == "Explicit voice eye request" for event in coordinator.journal.list_events())


@pytest.mark.anyio
async def test_malformed_voice_tool_is_rejected_without_execution(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    executed = False

    async def execute(action):
        nonlocal executed
        executed = True
        return {"ok": True}

    def validate(action):
        raise ValueError("invalid direction")

    gateway = RealtimeGateway("ws://sidecar", "serena", "instructions", coordinator, validate, execute)
    gateway._upstream = FakeUpstream()
    await gateway._handle_upstream_event(
        {
            "type": "response.function_call_arguments.done",
            "call_id": "call_bad",
            "name": "robot_action",
            "arguments": '{"movement":{"direction":"sideways"}}',
        }
    )

    assert executed is False
    events = coordinator.journal.list_events()
    assert any(event.event_type == "action.rejected" for event in events)


@pytest.mark.anyio
async def test_response_done_releases_operational_overlay_to_idle(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    coordinator.transition("corr-response", EventSource.voice_model, conversation=ConversationState.speaking)
    gateway = RealtimeGateway("ws://sidecar", "serena", "instructions", coordinator, lambda action: action, None)

    assert await gateway._handle_upstream_event({"type": "response.done"}) is True
    assert coordinator.state.conversation == ConversationState.idle


@pytest.mark.anyio
async def test_tool_response_stays_formulating_until_spoken_followup_finishes(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    coordinator.transition("corr-tool", EventSource.voice_model, conversation=ConversationState.formulating)

    async def execute(action):
        return {"ok": True}

    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        execute,
    )
    gateway._upstream = FakeUpstream()
    await gateway._handle_upstream_event(
        {
            "type": "response.function_call_arguments.done",
            "call_id": "call-eyes",
            "name": "robot_action",
            "arguments": '{"eyes":{"expression":"happy"}}',
        }
    )

    await gateway._handle_upstream_event({"type": "response.done"})
    assert coordinator.state.conversation == ConversationState.formulating

    await gateway._handle_upstream_event({"type": "response.done"})
    assert coordinator.state.conversation == ConversationState.idle


@pytest.mark.anyio
async def test_new_upstream_session_receives_server_tools_and_recent_history(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    coordinator.record_turn("user", "remember this", EventSource.browser, "corr-history")
    gateway = RealtimeGateway("ws://sidecar", "serena", "instructions", coordinator, lambda action: action, None)
    upstream = FakeUpstream()
    gateway._upstream = upstream

    assert await gateway._handle_upstream_event({"type": "session.created"}) is True

    session = upstream.sent[0]["session"]
    assert session["audio"]["output"]["voice"] == "serena"
    assert session["tools"][0]["name"] == "robot_action"
    eye_enum = session["tools"][0]["parameters"]["properties"]["eyes"]["properties"]["expression"]["enum"]
    assert "happy" in eye_enum
    assert {"fault", "listening", "thinking", "speaking"}.isdisjoint(eye_enum)
    assert upstream.sent[1]["item"]["content"][0]["text"] == "remember this"


def test_browser_contains_no_voice_tool_execution_path():
    page = (Path(__file__).resolve().parents[2] / "web_control" / "index.html").read_text(encoding="utf-8")
    assert "realtimeToolDefinitions" not in page
    assert "executeRealtimeTool" not in page
    assert 'postJson("/robot/action"' not in page
    assert 'getJson("/brain/state")' in page
    assert "Voice active in another tab" in page
    assert "Voice reconnecting" in page


@pytest.mark.anyio
async def test_browser_disconnect_keeps_upstream_and_new_browser_replaces_old(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    gateway = RealtimeGateway("ws://sidecar", "serena", "instructions", coordinator, lambda action: action, None)
    upstream = FakeUpstream()

    async def ensure_upstream():
        gateway._upstream = upstream
        return upstream

    gateway._ensure_upstream = ensure_upstream
    first = FakeBrowser(block=True)
    second = FakeBrowser()
    first_task = asyncio.create_task(gateway.handle_browser(first))
    while gateway._client is not first:
        await asyncio.sleep(0)
    await gateway.handle_browser(second)
    await first_task

    assert first.closed[0] == 4001
    assert gateway._upstream is upstream
    assert first.sent[0]["type"] == "robit.session.snapshot"
    assert second.sent[0]["type"] == "robit.session.snapshot"


@pytest.mark.anyio
async def test_browser_disconnect_clears_voice_eye_overlay(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    voice_states = []
    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        None,
        voice_session_handler=voice_states.append,
    )
    upstream = FakeUpstream()

    async def ensure_upstream():
        gateway._upstream = upstream
        return upstream

    gateway._ensure_upstream = ensure_upstream
    await gateway.handle_browser(FakeBrowser())

    assert voice_states == [True, False]


@pytest.mark.anyio
async def test_next_request_recreates_missing_upstream(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    created = []

    async def connector(url, **kwargs):
        upstream = FakeUpstream()
        created.append(upstream)
        return upstream

    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        None,
        connector=connector,
    )
    gateway._pump_upstream = lambda: asyncio.sleep(3600)
    await gateway._send_upstream({"type": "response.create"})
    gateway._upstream_task.cancel()
    await asyncio.gather(gateway._upstream_task, return_exceptions=True)
    gateway._upstream = None
    await gateway._send_upstream({"type": "response.create"})
    gateway._upstream_task.cancel()
    await asyncio.gather(gateway._upstream_task, return_exceptions=True)

    assert len(created) == 2
    assert created[1].sent == [{"type": "response.create"}]


@pytest.mark.anyio
async def test_exhausted_upstream_connection_sets_server_fault(tmp_path):
    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    faults = []

    async def connector(url, **kwargs):
        raise OSError("sidecar unavailable")

    gateway = RealtimeGateway(
        "ws://sidecar",
        "serena",
        "instructions",
        coordinator,
        lambda action: action,
        None,
        connector=connector,
        server_fault_handler=lambda reason, active, correlation_id: faults.append((reason, active)),
    )

    with pytest.raises(OSError, match="sidecar unavailable"):
        await gateway._ensure_upstream()

    assert faults == [("realtime_upstream", True)]

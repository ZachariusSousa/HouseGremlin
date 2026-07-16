from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .brain_models import ActionIntent, ConversationState, EventSource, WorkPriority
from .coordinator import BrainCoordinator


logger = logging.getLogger("uvicorn.error")
ActionValidator = Callable[[dict[str, Any]], dict[str, Any]]
ActionExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ServerFaultHandler = Callable[[str, bool, str | None], None]
VoiceSessionHandler = Callable[[bool], None]
EYE_EXPRESSIONS = [
    "neutral",
    "angry",
    "cute",
    "concerned",
    "content",
    "happy",
    "startled",
    "sleepy",
    "curious",
    "confused",
    "suspicious",
    "wink",
]
EYE_EXPRESSION_ALIASES = {
    "embarrassed": "cute",
    "scared": "concerned",
    "relaxed": "content",
    "excited": "happy",
    "surprised": "startled",
}
EYE_COMMAND_MARKERS = re.compile(
    r"\b(eye|eyes|expression|show|make|give|try|change|set|want|look)\b",
    re.IGNORECASE,
)


def explicit_eye_expression(text: str, previous: str | None = None) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized or not EYE_COMMAND_MARKERS.search(normalized):
        return None
    for candidate in (*EYE_EXPRESSIONS, *EYE_EXPRESSION_ALIASES):
        if re.search(rf"\b{re.escape(candidate)}\b", normalized):
            return EYE_EXPRESSION_ALIASES.get(candidate, candidate)
    if previous and re.search(r"\b(again|it|that|same)\b", normalized):
        return previous
    return None


def robot_action_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "robot_action",
        "description": "Execute one bounded Robit action through the PC safety layer.",
        "parameters": {
            "type": "object",
            "properties": {
                "movement": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["forward", "reverse", "left", "right", "stop"]},
                        "speed": {"type": "integer", "minimum": 0, "maximum": 255},
                        "duration_ms": {"type": "integer", "minimum": 0},
                    },
                    "required": ["direction"],
                },
                "head": {
                    "type": "object",
                    "properties": {
                        "pan": {"type": "integer", "minimum": 55, "maximum": 135},
                        "tilt": {"type": "integer", "minimum": 35, "maximum": 115},
                        "pan_delta": {"type": "integer", "minimum": -80, "maximum": 80},
                        "tilt_delta": {"type": "integer", "minimum": -80, "maximum": 80},
                    },
                },
                "eyes": {
                    "type": "object",
                    "description": (
                        "Set a temporary emotional eye expression. Use this whenever the user explicitly asks "
                        "Robit to show, make, try, or change an expression. Operational states are automatic."
                    ),
                    "properties": {
                        "expression": {"type": "string", "enum": EYE_EXPRESSIONS},
                        "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
                    },
                    "required": ["expression"],
                },
                "emergency_stop": {"type": "boolean"},
            },
        },
    }


class RealtimeGateway:
    def __init__(
        self,
        upstream_url: str,
        voice: str,
        instructions: str,
        coordinator: BrainCoordinator,
        validate_action: ActionValidator,
        execute_action: ActionExecutor,
        connector=None,
        server_fault_handler: ServerFaultHandler | None = None,
        voice_session_handler: VoiceSessionHandler | None = None,
    ):
        self.upstream_url = upstream_url
        self.voice = voice
        self.instructions = instructions
        self.coordinator = coordinator
        self.validate_action = validate_action
        self.execute_action = execute_action
        self._connector = connector or connect
        self._server_fault_handler = server_fault_handler
        self._voice_session_handler = voice_session_handler
        self._upstream: ClientConnection | None = None
        self._upstream_task: asyncio.Task[None] | None = None
        self._upstream_lock = asyncio.Lock()
        self._client: WebSocket | None = None
        self._client_lock = asyncio.Lock()
        self._current_correlation_id: str | None = None
        self._function_args: dict[str, str] = {}
        self._completed_calls: set[str] = set()
        self._awaiting_tool_followup_response = False
        self._last_explicit_eye_expression: str | None = None
        self._closing = False

    async def handle_browser(self, websocket: WebSocket) -> None:
        await websocket.accept()
        old_client: WebSocket | None
        async with self._client_lock:
            old_client = self._client
            self._client = websocket
        if self._voice_session_handler is not None:
            self._voice_session_handler(True)
        if old_client and old_client is not websocket:
            try:
                await old_client.close(code=4001, reason="Replaced by a newer operator connection")
            except RuntimeError:
                pass

        await websocket.send_json({"type": "robit.session.snapshot", **self.coordinator.snapshot()})
        try:
            await self._ensure_upstream()
            while True:
                event = await websocket.receive_json()
                await self._handle_browser_event(event)
        except WebSocketDisconnect as exc:
            logger.info("realtime.browser_disconnected code=%s", exc.code)
        except (ConnectionClosed, OSError) as exc:
            await self._send_client({"type": "error", "error": {"message": f"Realtime upstream unavailable: {exc}"}})
        finally:
            disconnected_current_client = False
            async with self._client_lock:
                if self._client is websocket:
                    self._client = None
                    disconnected_current_client = True
            if disconnected_current_client and self._voice_session_handler is not None:
                self._voice_session_handler(False)

    async def _ensure_upstream(self) -> ClientConnection:
        if self._upstream is not None:
            return self._upstream
        async with self._upstream_lock:
            if self._upstream is None:
                try:
                    self._upstream = await self._connector(
                        self.upstream_url,
                        max_size=8 * 1024 * 1024,
                        proxy=None,
                    )
                except Exception:
                    self._set_server_fault(True)
                    raise
                self._set_server_fault(False)
                self._upstream_task = asyncio.create_task(self._pump_upstream())
            return self._upstream

    async def _send_upstream(self, event: dict[str, Any]) -> None:
        upstream = await self._ensure_upstream()
        try:
            await upstream.send(json.dumps(event))
        except ConnectionClosed:
            self._upstream = None
            upstream = await self._ensure_upstream()
            await upstream.send(json.dumps(event))

    async def _handle_browser_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == "session.update":
            requested_voice = event.get("session", {}).get("audio", {}).get("output", {}).get("voice")
            if isinstance(requested_voice, str) and requested_voice.strip():
                self.voice = requested_voice.strip()
            await self._send_server_session_update()
            return
        if event_type == "conversation.item.create":
            item = event.get("item") or {}
            if item.get("role") == "user":
                text = self._item_text(item)
                correlation_id = self.coordinator.new_correlation_id()
                self._current_correlation_id = correlation_id
                self.coordinator.record_turn("user", text, EventSource.browser, correlation_id)
                self.coordinator.transition(
                    correlation_id,
                    EventSource.browser,
                    conversation=ConversationState.formulating,
                )
        await self._send_upstream(event)

    async def _pump_upstream(self) -> None:
        upstream = self._upstream
        if upstream is None:
            return
        try:
            async for raw in upstream:
                try:
                    event = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                forward = await self._handle_upstream_event(event)
                if forward:
                    await self._send_client(event)
        except (ConnectionClosed, OSError) as exc:
            if not self._closing:
                self._set_server_fault(True)
                logger.warning("realtime.gateway_upstream_disconnected error=%r", exc)
                await self._send_client({"type": "error", "error": {"message": "Realtime voice server disconnected"}})
        finally:
            if self._upstream is upstream:
                self._upstream = None
                if not self._closing:
                    self._set_server_fault(True)

    async def _handle_upstream_event(self, event: dict[str, Any]) -> bool:
        event_type = str(event.get("type") or "")
        if event_type == "session.created":
            self._function_args.clear()
            self._completed_calls.clear()
            await self._send_server_session_update()
            await self._seed_upstream_history()
            return True
        if event_type == "input_audio_buffer.speech_started":
            self._current_correlation_id = self.coordinator.new_correlation_id()
            self.coordinator.transition(
                self._current_correlation_id,
                EventSource.browser,
                conversation=ConversationState.listening,
            )
        elif event_type == "input_audio_buffer.speech_stopped":
            correlation_id = self._correlation_id()
            self.coordinator.transition(
                correlation_id,
                EventSource.browser,
                conversation=ConversationState.formulating,
            )
        elif "input_audio_transcription" in event_type and not event_type.endswith(".delta"):
            transcript = str(event.get("transcript") or "")
            self.coordinator.record_turn(
                "user",
                transcript,
                EventSource.browser,
                self._correlation_id(),
            )
            await self._execute_explicit_eye_request(transcript)
        elif "transcript" in event_type and "input_audio_transcription" not in event_type and not event_type.endswith(".delta"):
            self.coordinator.record_turn(
                "assistant",
                str(event.get("transcript") or ""),
                EventSource.voice_model,
                self._correlation_id(),
            )
        elif event_type == "response.output_audio.delta":
            self.coordinator.transition(
                self._correlation_id(),
                EventSource.voice_model,
                conversation=ConversationState.speaking,
            )
        elif "function_call_arguments.delta" in event_type:
            call_id = str(event.get("call_id") or event.get("item_id") or "")
            if call_id:
                self._function_args[call_id] = self._function_args.get(call_id, "") + str(event.get("delta") or "")
            return False
        elif "function_call_arguments.done" in event_type:
            await self._execute_tool_event(event)
            return False
        elif event_type == "response.output_item.done" and (event.get("item") or {}).get("type") == "function_call":
            await self._execute_tool_event(event)
            return False
        elif event_type == "response.done":
            if self._awaiting_tool_followup_response:
                self._awaiting_tool_followup_response = False
            else:
                self.coordinator.transition(
                    self._correlation_id(),
                    EventSource.voice_model,
                    conversation=ConversationState.idle,
                )
        return True

    async def _execute_tool_event(self, event: dict[str, Any]) -> None:
        item = event.get("item") or event
        call_id = str(item.get("call_id") or event.get("call_id") or item.get("id") or event.get("item_id") or "")
        name = str(item.get("name") or event.get("name") or "")
        if not call_id or call_id in self._completed_calls:
            return
        self._completed_calls.add(call_id)
        if name != "robot_action":
            await self._return_tool_output(call_id, {"ok": False, "error": f"Unsupported tool: {name}"})
            return
        raw_arguments = item.get("arguments") or event.get("arguments") or self._function_args.pop(call_id, "{}")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            action = self.validate_action(arguments)
            intent = ActionIntent(
                action=action,
                origin=EventSource.voice_model,
                correlation_id=self._correlation_id(),
                priority=WorkPriority.emergency if action.get("emergency_stop") else WorkPriority.model_action,
                reason="Realtime voice tool call",
            )
            result = await self.coordinator.execute_action(intent, self.execute_action)
            output = {"ok": True, "correlation_id": intent.correlation_id, "result": result}
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            correlation_id = self._correlation_id()
            self.coordinator.record(
                "action.rejected",
                EventSource.policy,
                correlation_id,
                {"call_id": call_id, "error": str(exc)},
                WorkPriority.model_action,
            )
            output = {"ok": False, "correlation_id": correlation_id, "error": str(exc)}
        self._awaiting_tool_followup_response = True
        await self._return_tool_output(call_id, output)
        await self._send_client({"type": "robit.action", "call_id": call_id, **output})

    async def _execute_explicit_eye_request(self, transcript: str) -> None:
        expression = explicit_eye_expression(transcript, self._last_explicit_eye_expression)
        if expression is None:
            return
        correlation_id = self._correlation_id()
        try:
            action = self.validate_action({"eyes": {"expression": expression}})
            intent = ActionIntent(
                action=action,
                origin=EventSource.browser,
                correlation_id=correlation_id,
                priority=WorkPriority.manual_action,
                reason="Explicit voice eye request",
            )
            result = await self.coordinator.execute_action(intent, self.execute_action)
            self._last_explicit_eye_expression = expression
            output = {"ok": True, "correlation_id": correlation_id, "result": result}
        except (ValidationError, ValueError, TypeError) as exc:
            self.coordinator.record(
                "action.rejected",
                EventSource.policy,
                correlation_id,
                {"expression": expression, "error": str(exc)},
                WorkPriority.manual_action,
            )
            output = {"ok": False, "correlation_id": correlation_id, "error": str(exc)}
        await self._send_client(
            {
                "type": "robit.action",
                "call_id": f"explicit-eyes-{correlation_id}",
                **output,
            }
        )

    async def _return_tool_output(self, call_id: str, output: dict[str, Any]) -> None:
        await self._send_upstream(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(output, separators=(",", ":"), default=str),
                },
            }
        )
        await self._send_upstream({"type": "response.create"})

    async def _send_server_session_update(self) -> None:
        if self._upstream is None:
            return
        await self._upstream.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": self.instructions,
                        "audio": {"output": {"voice": self.voice}},
                        "tools": [robot_action_tool()],
                        "tool_choice": "auto",
                    },
                }
            )
        )

    async def _seed_upstream_history(self) -> None:
        for turn in self.coordinator.journal.recent_turns(self.coordinator.conversation_id, 20):
            content_type = "input_text" if turn.role == "user" else "output_text"
            await self._send_upstream(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": turn.role,
                        "content": [{"type": content_type, "text": turn.text}],
                    },
                }
            )

    async def _send_client(self, event: dict[str, Any]) -> None:
        async with self._client_lock:
            client = self._client
        if client is None:
            return
        try:
            await client.send_json(event)
        except (RuntimeError, WebSocketDisconnect):
            pass

    def _correlation_id(self) -> str:
        if not self._current_correlation_id:
            self._current_correlation_id = self.coordinator.new_correlation_id()
        return self._current_correlation_id

    def _set_server_fault(self, active: bool) -> None:
        if self._server_fault_handler is not None:
            self._server_fault_handler("realtime_upstream", active, self._current_correlation_id)

    @staticmethod
    def _item_text(item: dict[str, Any]) -> str:
        return " ".join(
            str(part.get("text") or "")
            for part in item.get("content") or []
            if isinstance(part, dict)
        ).strip()

    async def shutdown(self) -> None:
        self._closing = True
        if self._voice_session_handler is not None:
            self._voice_session_handler(False)
        if self._upstream is not None:
            await self._upstream.close()
        if self._upstream_task is not None:
            await asyncio.gather(self._upstream_task, return_exceptions=True)

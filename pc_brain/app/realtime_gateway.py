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
SceneInspector = Callable[[str], Awaitable[dict[str, Any]]]
SceneContextProvider = Callable[[], dict[str, Any] | None]
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
MOVEMENT_COMMAND_MARKERS = re.compile(r"\b(move|drive|go|roll|turn|reverse|back\s+up)\b", re.IGNORECASE)
REPEAT_COMMAND_MARKERS = re.compile(r"\b(again|one more time|do it again|same thing|go ahead)\b", re.IGNORECASE)
EXPLICIT_VISUAL_QUESTION_MARKERS = re.compile(
    r"\b("
    r"what (?:can|do) you see|what(?:'s| is) (?:in|on) (?:the )?(?:image|camera|picture|view)|"
    r"can you see|do you see|you see|see anything|look (?:again|around|at)|"
    r"use (?:your )?vision|(?:take|capture|get) (?:a )?(?:photo|picture|frame)|"
    r"camera|vision|visual|image|picture|photo|current view"
    r")\b",
    re.IGNORECASE,
)
VISUAL_ASSISTANT_CLAIM_MARKERS = re.compile(
    r"\b("
    r"i (?:can )?see|i(?:'m| am) seeing|i(?:'m| am) looking|i (?:just )?(?:looked|checked|inspected|captured)|"
    r"fresh look|scene inspection|current view|the image (?:shows|contains)|my sensors"
    r")\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


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


def explicit_visual_question(text: str) -> bool:
    """Return true only for direct requests about Robit's camera view."""
    normalized = re.sub(r"\s+", " ", text.strip())
    return bool(normalized and EXPLICIT_VISUAL_QUESTION_MARKERS.search(normalized))


def visual_dialogue_turn(role: str, text: str) -> bool:
    """Keep superseded visual claims out of durable conversational context."""
    if role == "user":
        return explicit_visual_question(text)
    return role == "assistant" and bool(VISUAL_ASSISTANT_CLAIM_MARKERS.search(text))


def _number_after(normalized: str, marker: str) -> int | None:
    tail = normalized.split(marker, 1)[1]
    if match := re.search(r"\b\d{1,3}\b", tail):
        return int(match.group())

    tokens = tail.split()
    start = next((index for index, token in enumerate(tokens) if token in NUMBER_WORDS or token in {"a", "hundred"}), None)
    if start is None:
        return None
    value = 0
    current = 0
    found = False
    for token in tokens[start:]:
        if token == "and":
            continue
        if token == "a" and not found:
            current = 1
            found = True
            continue
        if token == "hundred":
            current = max(1, current) * 100
            found = True
            continue
        if token not in NUMBER_WORDS:
            break
        current += NUMBER_WORDS[token]
        found = True
    return value + current if found else None


def explicit_robot_action(text: str, previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not normalized:
        return None

    action: dict[str, Any] = {}
    expression = explicit_eye_expression(normalized, (previous or {}).get("eyes", {}).get("expression"))
    if expression is not None:
        action["eyes"] = {"expression": expression}

    if re.search(r"\b(stop|halt|freeze)\b", normalized) and re.search(
        r"\b(move|moving|drive|driving|robot|robit|please)\b", normalized
    ):
        action["emergency_stop"] = True
    elif MOVEMENT_COMMAND_MARKERS.search(normalized) and "head" not in normalized:
        direction = None
        if re.search(r"\b(forward|forwards|ahead)\b", normalized):
            direction = "forward"
        elif re.search(r"\b(backward|backwards|reverse|back up)\b", normalized):
            direction = "reverse"
        elif re.search(r"\b(left)\b", normalized):
            direction = "left"
        elif re.search(r"\b(right)\b", normalized):
            direction = "right"
        if direction is not None:
            short_move = re.search(r"\b(little|small|short|brief|tiny)\b", normalized) is not None
            action["movement"] = {"direction": direction, "duration_ms": 500 if short_move else 700}

    for axis in ("pan", "tilt"):
        if re.search(rf"\b{axis}\b", normalized):
            value = _number_after(normalized, axis)
            if value is not None:
                action.setdefault("head", {})[axis] = value

    if not action and previous and REPEAT_COMMAND_MARKERS.search(normalized):
        return {key: value.copy() if isinstance(value, dict) else value for key, value in previous.items()}
    return action or None


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


def inspect_scene_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "inspect_scene",
        "description": (
            "Read a fresh Robit camera frame and return a structured scene description. Read-only. "
            "If fresh is false, describe the snapshot as what was last seen, not what is present now."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The visual question to answer from a fresh frame."},
            },
            "required": ["question"],
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
        inspect_scene: SceneInspector | None = None,
        scene_context: SceneContextProvider | None = None,
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
        self._inspect_scene = inspect_scene
        self._scene_context = scene_context
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
        self._last_explicit_robot_action: dict[str, Any] | None = None
        self._explicit_action_correlation_id: str | None = None
        self._vision_used_in_turn = False
        self._visual_response_task: asyncio.Task[None] | None = None
        self._suppress_cancelled_response = False
        self._visual_cancelled = asyncio.Event()
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
                self._vision_used_in_turn = False
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
        if self._suppress_cancelled_response and event_type.startswith("response."):
            if event_type == "response.done":
                self._visual_cancelled.set()
            return False
        if event_type == "input_audio_buffer.speech_started":
            self._current_correlation_id = self.coordinator.new_correlation_id()
            self._vision_used_in_turn = False
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
            if explicit_visual_question(transcript):
                await self._start_explicit_visual_response(transcript)
                return True
            await self.refresh_scene_context()
            await self._execute_explicit_robot_request(transcript)
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

    async def _start_explicit_visual_response(self, question: str) -> None:
        """Replace speculative speech with an answer grounded in one fresh frame."""
        if self._visual_response_task is not None and not self._visual_response_task.done():
            self._visual_response_task.cancel()
        self._vision_used_in_turn = True
        self._visual_cancelled = asyncio.Event()
        self._suppress_cancelled_response = True
        await self._send_upstream({"type": "response.cancel"})
        self._visual_response_task = asyncio.create_task(self._answer_explicit_visual_question(question))

    async def _answer_explicit_visual_question(self, question: str) -> None:
        try:
            if self._inspect_scene is None:
                raise RuntimeError("Vision is not configured")
            result = await self._inspect_scene(question)
            grounding = json.dumps(result, separators=(",", ":"), default=str)
        except (ValueError, TypeError, RuntimeError) as exc:
            grounding = json.dumps({"fresh": False, "error": str(exc)}, separators=(",", ":"))

        try:
            await asyncio.wait_for(self._visual_cancelled.wait(), timeout=1.0)
        except TimeoutError:
            pass

        if self._closing or self._upstream is None:
            return
        self._suppress_cancelled_response = False
        instructions = (
            self._instructions_with_scene()
            + "\n\nMANDATORY CAMERA ANSWER FOR THIS TURN: Answer the user's visual question directly from "
            "the validated inspection JSON below. This data overrides every earlier visual description in the "
            "conversation. Never repeat objects from an older view, never claim an inspection that is not in this "
            "JSON, and do not say you are about to look. If fresh is false, explicitly say it is the last cached view. "
            f"INSPECTION_JSON={grounding}"
        )
        await self._send_upstream(
            {
                "type": "response.create",
                "response": {"instructions": instructions, "tool_choice": "none"},
            }
        )

    async def _execute_tool_event(self, event: dict[str, Any]) -> None:
        item = event.get("item") or event
        call_id = str(item.get("call_id") or event.get("call_id") or item.get("id") or event.get("item_id") or "")
        name = str(item.get("name") or event.get("name") or "")
        if not call_id or call_id in self._completed_calls:
            return
        self._completed_calls.add(call_id)
        raw_arguments = item.get("arguments") or event.get("arguments") or self._function_args.pop(call_id, "{}")
        if name == "inspect_scene":
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                question = str((arguments or {}).get("question") or "What is visible?").strip()
                if self._inspect_scene is None:
                    raise ValueError("Vision is not configured")
                output = {"ok": True, **(await self._inspect_scene(question))}
                self._vision_used_in_turn = True
            except (json.JSONDecodeError, ValueError, TypeError, RuntimeError) as exc:
                output = {"ok": False, "error": str(exc)}
            self._awaiting_tool_followup_response = True
            await self._return_tool_output(call_id, output)
            await self._send_client({"type": "robit.perception", "call_id": call_id, **output})
            return
        if name != "robot_action":
            await self._return_tool_output(call_id, {"ok": False, "error": f"Unsupported tool: {name}"})
            return
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            action = self.validate_action(arguments)
            correlation_id = self._correlation_id()
            if self._vision_used_in_turn and (action.get("movement") or action.get("head")) and not action.get("emergency_stop"):
                raise ValueError("Movement and head actions are blocked in the same turn as a vision result")
            if correlation_id == self._explicit_action_correlation_id and action == self._last_explicit_robot_action:
                output = {"ok": True, "correlation_id": correlation_id, "deduplicated": True}
            else:
                intent = ActionIntent(
                    action=action,
                    origin=EventSource.voice_model,
                    correlation_id=correlation_id,
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

    async def _execute_explicit_robot_request(self, transcript: str) -> None:
        action_payload = explicit_robot_action(transcript, self._last_explicit_robot_action)
        if action_payload is None:
            return
        correlation_id = self._correlation_id()
        try:
            action = self.validate_action(action_payload)
            only_eyes = set(action) == {"eyes"}
            intent = ActionIntent(
                action=action,
                origin=EventSource.browser,
                correlation_id=correlation_id,
                priority=WorkPriority.emergency if action.get("emergency_stop") else WorkPriority.manual_action,
                reason="Explicit voice eye request" if only_eyes else "Explicit voice robot request",
            )
            result = await self.coordinator.execute_action(intent, self.execute_action)
            self._last_explicit_robot_action = action
            self._explicit_action_correlation_id = correlation_id
            if expression := action.get("eyes", {}).get("expression"):
                self._last_explicit_eye_expression = expression
            output = {"ok": True, "correlation_id": correlation_id, "result": result}
        except (ValidationError, ValueError, TypeError) as exc:
            self.coordinator.record(
                "action.rejected",
                EventSource.policy,
                correlation_id,
                {"action": action_payload, "error": str(exc)},
                WorkPriority.manual_action,
            )
            output = {"ok": False, "correlation_id": correlation_id, "error": str(exc)}
        await self._send_client(
            {
                "type": "robit.action",
                "call_id": f"explicit-action-{correlation_id}",
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
                        "instructions": self._instructions_with_scene(),
                        "audio": {"output": {"voice": self.voice}},
                        "tools": [robot_action_tool(), inspect_scene_tool()],
                        "tool_choice": "auto",
                    },
                }
            )
        )

    async def refresh_scene_context(self, snapshot: Any | None = None) -> None:
        if self._upstream is not None:
            await self._send_server_session_update()

    def _instructions_with_scene(self) -> str:
        snapshot = self._scene_context() if self._scene_context is not None else None
        if not snapshot:
            scene = (
                "LIVE VISUAL CONTEXT: unavailable or expired. Do not claim to currently see specific objects. "
                "Use inspect_scene when the user asks a visual question."
            )
        else:
            entities = [
                {"label": entity.get("label"), "confidence": entity.get("confidence")}
                for entity in snapshot.get("entities", [])
                if isinstance(entity, dict)
            ]
            context = {
                "frame_id": snapshot.get("frame_id"),
                "observed_at": snapshot.get("observed_at"),
                "summary": snapshot.get("summary"),
                "entities": entities,
                "uncertainty": snapshot.get("uncertainty"),
            }
            scene = (
                "LIVE VISUAL CONTEXT: "
                + json.dumps(context, separators=(",", ":"), default=str)
                + " Treat this validated snapshot as Robit's current view during normal conversation. "
                "It overrides all user or assistant descriptions of earlier views. Use it naturally when relevant, "
                "but do not invent details outside it or repeat objects remembered from prior dialogue. "
                "For an explicit visual question, use inspect_scene and answer only from its result."
            )
        return f"{self.instructions}\n\n{scene}"

    async def _seed_upstream_history(self) -> None:
        for turn in self.coordinator.journal.recent_turns(self.coordinator.conversation_id, 20):
            if visual_dialogue_turn(turn.role, turn.text):
                continue
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
        if self._visual_response_task is not None and not self._visual_response_task.done():
            self._visual_response_task.cancel()
            await asyncio.gather(self._visual_response_task, return_exceptions=True)
        if self._voice_session_handler is not None:
            self._voice_session_handler(False)
        if self._upstream is not None:
            await self._upstream.close()
        if self._upstream_task is not None:
            await asyncio.gather(self._upstream_task, return_exceptions=True)

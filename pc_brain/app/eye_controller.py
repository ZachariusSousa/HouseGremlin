from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from .brain_models import (
    BodyState,
    BrainEvent,
    CognitiveState,
    ConversationState,
    EmotionalEyeExpression,
    EventSource,
    EyeExpression,
    EyeState,
    utc_now,
)
from .coordinator import BrainCoordinator


RobotPost = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
EMOTIONAL_EYE_EXPRESSIONS: tuple[EmotionalEyeExpression, ...] = (
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
)
OPERATIONAL_EYE_EXPRESSIONS = frozenset({"fault", "listening", "thinking", "speaking"})


class EyeController:
    def __init__(
        self,
        coordinator: BrainCoordinator,
        robot_post: RobotPost,
        *,
        heartbeat_post: RobotPost | None = None,
        default_mood_duration_ms: int = 8000,
        maximum_mood_duration_ms: int = 10000,
        heartbeat_interval_seconds: float | None = 3.0,
    ):
        self.coordinator = coordinator
        self.robot_post = robot_post
        self.heartbeat_post = heartbeat_post or robot_post
        self.default_mood_duration_ms = default_mood_duration_ms
        self.maximum_mood_duration_ms = maximum_mood_duration_ms
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._queue: asyncio.Queue[EyeExpression] = asyncio.Queue(maxsize=1)
        self._worker_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._expiry_task: asyncio.Task[None] | None = None
        self._last_requested: EyeExpression | None = None
        self._last_sent: EyeExpression | None = None
        self._heartbeat_failed = False
        self._heartbeat_ready = heartbeat_interval_seconds is None
        self._server_faults: set[str] = set()
        self._voice_session_active = False
        self._running = False
        coordinator.subscribe_state(self.observe_state)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._command_worker())
        if self.heartbeat_interval_seconds is not None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_worker())
        self.observe_state(self.coordinator.state)
        if self._heartbeat_ready:
            self.force_sync()

    async def shutdown(self) -> None:
        self._running = False
        tasks = [task for task in (self._worker_task, self._heartbeat_task, self._expiry_task) if task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker_task = None
        self._heartbeat_task = None
        self._expiry_task = None

    def select_mood(
        self,
        expression: str,
        duration_ms: int | None,
        source: EventSource,
        correlation_id: str,
    ) -> EyeState:
        if expression not in EMOTIONAL_EYE_EXPRESSIONS:
            self.coordinator.record(
                "eyes.mood.rejected",
                EventSource.policy,
                correlation_id,
                {"expression": expression, "reason": "operational expressions are coordinator-owned"},
            )
            raise ValueError(f"Model cannot select operational eye expression: {expression}")

        lifetime_ms = duration_ms or self.default_mood_duration_ms
        lifetime_ms = min(max(1, lifetime_ms), self.maximum_mood_duration_ms)
        if self._expiry_task is not None:
            self._expiry_task.cancel()
            self._expiry_task = None
        if expression == "neutral":
            lifetime_ms = 0
        eyes = self.coordinator.state.eyes.model_copy(
            update={
                "base_expression": expression,
                "base_duration_ms": lifetime_ms,
                "base_expires_at": None,
            }
        )
        self.coordinator.update_eye_state(
            eyes,
            "eyes.mood.selected",
            source,
            correlation_id,
            {"expression": expression, "duration_ms": lifetime_ms},
        )
        self._apply_effective(source, correlation_id)
        return self.coordinator.state.eyes

    def observe_state(self, state: CognitiveState, event: BrainEvent | None = None) -> None:
        correlation_id = (
            event.correlation_id
            if event is not None
            else state.active_correlation_id or self.coordinator.new_correlation_id()
        )
        source = event.source if event is not None else EventSource.system
        self._expire_mood_if_due(source, correlation_id)
        self._apply_effective(source, correlation_id)

    def set_voice_session_active(self, active: bool) -> None:
        if self._voice_session_active == active:
            return
        self._voice_session_active = active
        correlation_id = self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id()
        self.coordinator.record(
            "eyes.voice_session.changed",
            EventSource.system,
            correlation_id,
            {"active": active},
        )
        self.observe_state(self.coordinator.state)

    def set_server_fault(self, reason: str, active: bool, correlation_id: str | None = None) -> None:
        was_active = reason in self._server_faults
        if active == was_active:
            return
        if active:
            self._server_faults.add(reason)
        else:
            self._server_faults.discard(reason)
        correlation_id = correlation_id or self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id()
        self.coordinator.record(
            "eyes.server_fault.activated" if active else "eyes.server_fault.cleared",
            EventSource.system,
            correlation_id,
            {"reason": reason},
        )
        self.observe_state(self.coordinator.state)

    def force_sync(self) -> None:
        self._last_requested = None
        self._enqueue(self.coordinator.state.eyes.effective_expression)

    def _desired_expression(self, state: CognitiveState) -> EyeExpression:
        if self._server_faults or state.safety == "fault" or state.body == BodyState.fault:
            return "fault"
        if state.conversation == ConversationState.formulating:
            return "thinking"
        if self._voice_session_active and state.conversation == ConversationState.speaking:
            return "speaking"
        if self._voice_session_active and state.conversation == ConversationState.listening:
            return "listening"
        if self._voice_session_active and state.conversation == ConversationState.interrupted:
            return "startled"
        return state.eyes.base_expression

    def _apply_effective(self, source: EventSource, correlation_id: str) -> None:
        desired = self._desired_expression(self.coordinator.state)
        current = self.coordinator.state.eyes
        if (
            desired == current.base_expression
            and current.base_expression != "neutral"
            and current.base_expires_at is None
        ):
            expires_at = utc_now() + timedelta(milliseconds=current.base_duration_ms)
            current = current.model_copy(update={"base_expires_at": expires_at})
            self.coordinator.update_eye_state(
                current,
                "eyes.mood.visible",
                source,
                correlation_id,
                {"expression": current.base_expression, "duration_ms": current.base_duration_ms},
            )
            self._schedule_expiry(expires_at, correlation_id)
        if desired != current.effective_expression:
            updated = current.model_copy(update={"effective_expression": desired})
            self.coordinator.update_eye_state(
                updated,
                "eyes.expression.changed",
                source,
                correlation_id,
                {"previous": current.effective_expression, "expression": desired},
            )
        self._enqueue(desired)

    def _expire_mood_if_due(self, source: EventSource, correlation_id: str) -> bool:
        eyes = self.coordinator.state.eyes
        if eyes.base_expires_at is None or eyes.base_expires_at > utc_now():
            return False
        updated = eyes.model_copy(
            update={"base_expression": "neutral", "base_duration_ms": 0, "base_expires_at": None}
        )
        self.coordinator.update_eye_state(
            updated,
            "eyes.mood.expired",
            source,
            correlation_id,
            {"previous": eyes.base_expression},
        )
        return True

    def _schedule_expiry(self, expires_at, correlation_id: str) -> None:
        if self._expiry_task is not None:
            self._expiry_task.cancel()
        if not self._running:
            return
        self._expiry_task = asyncio.create_task(self._expire_at(expires_at, correlation_id))

    async def _expire_at(self, expires_at, correlation_id: str) -> None:
        delay = max(0.0, (expires_at - utc_now()).total_seconds())
        await asyncio.sleep(delay)
        eyes = self.coordinator.state.eyes
        if eyes.base_expires_at != expires_at:
            return
        if self._expire_mood_if_due(EventSource.system, correlation_id):
            self._apply_effective(EventSource.system, correlation_id)

    def _enqueue(self, expression: EyeExpression) -> None:
        if not self._running or not self._heartbeat_ready or expression == self._last_requested:
            return
        self._last_requested = expression
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(expression)

    async def _command_worker(self) -> None:
        while True:
            expression = await self._queue.get()
            try:
                await self.robot_post("/api/eyes", {"expression": expression, "duration_ms": 0})
                self._last_sent = expression
            except Exception as exc:
                self._last_requested = None
                correlation_id = self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id()
                self.coordinator.record(
                    "eyes.command.failed",
                    EventSource.firmware,
                    correlation_id,
                    {"expression": expression, "error": str(exc)},
                )
            finally:
                self._queue.task_done()

    async def _heartbeat_worker(self) -> None:
        assert self.heartbeat_interval_seconds is not None
        while True:
            try:
                result = await self.heartbeat_post("/api/brain-heartbeat", {})
                recovered = bool(result.get("heartbeat_recovered"))
                first_success = not self._heartbeat_ready
                self._heartbeat_ready = True
                if first_success or self._heartbeat_failed or recovered:
                    correlation_id = self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id()
                    self.coordinator.record(
                        "eyes.heartbeat.recovered",
                        EventSource.firmware,
                        correlation_id,
                        {"firmware_watchdog_recovered": recovered},
                    )
                    self.force_sync()
                self._heartbeat_failed = False
            except Exception as exc:
                if not self._heartbeat_failed:
                    correlation_id = self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id()
                    self.coordinator.record(
                        "eyes.heartbeat.failed",
                        EventSource.firmware,
                        correlation_id,
                        {"error": str(exc)},
                    )
                self._heartbeat_failed = True
            await asyncio.sleep(self.heartbeat_interval_seconds)

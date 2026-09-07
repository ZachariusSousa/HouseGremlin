from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from .brain_models import (
    ActiveFault,
    ActionIntent,
    BodyState,
    BrainEvent,
    CognitiveState,
    ConversationState,
    EyeState,
    EventSource,
    FaultSeverity,
    WorkPriority,
    utc_now,
)
from .correlation import current_correlation_id
from .journal import EventJournal
from .resource_lease import PriorityResourceLease


ActionExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
StateListener = Callable[[CognitiveState, BrainEvent], None]


class BrainCoordinator:
    def __init__(self, journal: EventJournal, conversation_id: str = "default"):
        self.journal = journal
        self.conversation_id = conversation_id
        restored_state = journal.restore_state(conversation_id)
        self._active_faults: dict[str, ActiveFault] = {}
        self.state = restored_state.model_copy(
            update={
                "body": (
                    BodyState.stationary
                    if restored_state.body == BodyState.fault
                    else restored_state.body
                ),
                "safety": "normal",
                "eyes": EyeState(),
            }
        )
        self.resource_lease = PriorityResourceLease()
        self._action_lease = PriorityResourceLease()
        self._state_listeners: list[StateListener] = []

    def subscribe_state(self, listener: StateListener) -> None:
        if listener not in self._state_listeners:
            self._state_listeners.append(listener)

    @property
    def active_faults(self) -> list[ActiveFault]:
        return sorted(
            self._active_faults.values(),
            key=lambda fault: (fault.timestamp, fault.source),
        )

    @property
    def has_critical_fault(self) -> bool:
        return any(fault.severity == "critical" for fault in self._active_faults.values())

    @property
    def critical_fault_reason(self) -> str | None:
        for fault in self.active_faults:
            if fault.severity == "critical":
                return f"{fault.source}: {fault.message}"
        return None

    @staticmethod
    def new_correlation_id() -> str:
        return str(uuid4())

    def record(
        self,
        event_type: str,
        source: EventSource,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
        priority: WorkPriority = WorkPriority.foreground,
        causation_id: str | None = None,
        conversation_id: str | None = None,
    ) -> BrainEvent:
        return self.journal.append(
            BrainEvent(
                event_type=event_type,
                source=source,
                correlation_id=correlation_id,
                causation_id=causation_id,
                conversation_id=conversation_id or self.conversation_id,
                priority=priority,
                payload=payload or {},
            )
        )

    def transition(
        self,
        correlation_id: str,
        source: EventSource = EventSource.system,
        conversation: ConversationState | None = None,
        body: BodyState | None = None,
        safety: str | None = None,
    ) -> BrainEvent:
        update: dict[str, Any] = {
            "active_correlation_id": correlation_id,
            "updated_at": utc_now(),
            "safety": "fault" if self.has_critical_fault else ("stopped" if safety == "stopped" else "normal"),
        }
        if conversation is not None:
            update["conversation"] = conversation
        if body is not None:
            update["body"] = body
        self.state = self.state.model_copy(update=update)
        event = self.record(
            "state.changed",
            source,
            correlation_id,
            {"state": self.state.model_dump(mode="json")},
            WorkPriority.foreground,
        )
        self._notify_state_listeners(event)
        return event

    def register_fault(
        self,
        source: str,
        severity: FaultSeverity,
        message: str,
        correlation_id: str | None = None,
    ) -> ActiveFault:
        existing = self._active_faults.get(source)
        if existing is not None and existing.severity == severity and existing.message == message:
            return existing
        fault = ActiveFault(source=source, severity=severity, message=message)
        self._active_faults[source] = fault
        correlation_id = correlation_id or self.state.active_correlation_id or self.new_correlation_id()
        self.state = self.state.model_copy(
            update={
                "active_correlation_id": correlation_id,
                "updated_at": utc_now(),
                "safety": "fault" if self.has_critical_fault else "normal",
            }
        )
        event = self.record(
            "fault.registered",
            EventSource.system,
            correlation_id,
            {"fault": fault.model_dump(mode="json")},
        )
        self._notify_state_listeners(event)
        return fault

    def clear_fault(self, source: str, correlation_id: str | None = None) -> bool:
        fault = self._active_faults.pop(source, None)
        if fault is None:
            return False
        correlation_id = correlation_id or self.state.active_correlation_id or self.new_correlation_id()
        self.state = self.state.model_copy(
            update={
                "active_correlation_id": correlation_id,
                "updated_at": utc_now(),
                "safety": "fault" if self.has_critical_fault else "normal",
            }
        )
        event = self.record(
            "fault.cleared",
            EventSource.system,
            correlation_id,
            {"fault": fault.model_dump(mode="json")},
        )
        self._notify_state_listeners(event)
        return True

    def _notify_state_listeners(self, event: BrainEvent) -> None:
        for listener in tuple(self._state_listeners):
            try:
                listener(self.state, event)
            except Exception as exc:
                self.record(
                    "state.listener.failed",
                    EventSource.system,
                    event.correlation_id,
                    {"error": str(exc)},
                    WorkPriority.foreground,
                    event.event_id,
                )

    def update_eye_state(
        self,
        eyes: EyeState,
        event_type: str,
        source: EventSource,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
    ) -> BrainEvent:
        self.state = self.state.model_copy(update={"eyes": eyes, "updated_at": utc_now()})
        return self.record(
            event_type,
            source,
            correlation_id,
            {"eyes": eyes.model_dump(mode="json"), **(payload or {})},
            WorkPriority.foreground,
        )

    def record_turn(self, role: str, text: str, source: EventSource, correlation_id: str) -> BrainEvent | None:
        text = text.strip()
        if not text:
            return None
        event_type = "conversation.user.completed" if role == "user" else "conversation.assistant.completed"
        return self.record(event_type, source, correlation_id, {"role": role, "text": text})

    def recent_messages(self, limit: int = 20) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.text} for turn in self.journal.recent_turns(self.conversation_id, limit)]

    async def execute_action(self, intent: ActionIntent, executor: ActionExecutor) -> dict[str, Any]:
        proposed = self.record(
            "action.proposed",
            intent.origin,
            intent.correlation_id,
            {"action": intent.action, "reason": intent.reason},
            intent.priority,
            intent.causation_id,
            intent.conversation_id,
        )
        async def run() -> dict[str, Any]:
            body_state = BodyState.looking if intent.action.get("head") and not intent.action.get("movement") else BodyState.executing_skill
            self.transition(intent.correlation_id, intent.origin, body=body_state)
            self.record(
                "action.approved",
                EventSource.policy,
                intent.correlation_id,
                {"action": intent.action},
                intent.priority,
                proposed.event_id,
            )
            token = current_correlation_id.set(intent.correlation_id)
            try:
                result = await executor(intent.action)
            except Exception as exc:
                self.record(
                    "action.failed",
                    EventSource.firmware,
                    intent.correlation_id,
                    {"action": intent.action, "error": str(exc)},
                    intent.priority,
                    proposed.event_id,
                )
                self.register_fault(
                    "actuation",
                    "critical",
                    str(exc) or type(exc).__name__,
                    intent.correlation_id,
                )
                self.transition(intent.correlation_id, EventSource.system, body=BodyState.fault)
                raise
            finally:
                current_correlation_id.reset(token)
            self.record(
                "action.completed",
                EventSource.firmware,
                intent.correlation_id,
                {"action": intent.action, "result": result},
                intent.priority,
                proposed.event_id,
            )
            self.clear_fault("actuation", intent.correlation_id)
            self.transition(intent.correlation_id, EventSource.system, body=BodyState.stationary)
            return result

        async with self._action_lease.acquire(intent.priority):
            return await run()

    def snapshot(self) -> dict[str, Any]:
        turns = self.journal.recent_turns(self.conversation_id, 20)
        return {
            "state": self.state.model_dump(mode="json"),
            "conversation": [turn.model_dump() for turn in turns],
            "events": [event.model_dump(mode="json") for event in self.journal.recent_events(self.conversation_id, 100)],
            "latest_sequence": self.journal.latest_sequence(self.conversation_id),
        }

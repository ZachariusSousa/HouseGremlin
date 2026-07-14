from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from .brain_models import (
    ActionIntent,
    BodyState,
    BrainEvent,
    CognitiveState,
    ConversationState,
    EventSource,
    WorkPriority,
    utc_now,
)
from .correlation import current_correlation_id
from .journal import EventJournal
from .resource_lease import PriorityResourceLease


ActionExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class BrainCoordinator:
    def __init__(self, journal: EventJournal, conversation_id: str = "default"):
        self.journal = journal
        self.conversation_id = conversation_id
        self.state = journal.restore_state(conversation_id)
        self.resource_lease = PriorityResourceLease()
        self._action_lease = PriorityResourceLease()

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
        update: dict[str, Any] = {"active_correlation_id": correlation_id, "updated_at": utc_now()}
        if conversation is not None:
            update["conversation"] = conversation
        if body is not None:
            update["body"] = body
        if safety is not None:
            update["safety"] = safety
        self.state = self.state.model_copy(update=update)
        return self.record(
            "state.changed",
            source,
            correlation_id,
            {"state": self.state.model_dump(mode="json")},
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
        is_emergency = bool(intent.action.get("emergency_stop"))
        if is_emergency:
            intent.priority = WorkPriority.emergency
            self.transition(intent.correlation_id, intent.origin, body=BodyState.stationary, safety="stopped")

        async def run() -> dict[str, Any]:
            body_state = BodyState.looking if intent.action.get("head") and not intent.action.get("movement") else BodyState.executing_skill
            if not is_emergency:
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
                self.transition(intent.correlation_id, EventSource.system, body=BodyState.fault, safety="fault")
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
            self.transition(intent.correlation_id, EventSource.system, body=BodyState.stationary)
            return result

        if is_emergency:
            return await run()
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

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventSource(str, Enum):
    browser = "browser"
    text_model = "text_model"
    voice_model = "voice_model"
    manual = "manual"
    api = "api"
    policy = "policy"
    firmware = "firmware"
    system = "system"


class WorkPriority(IntEnum):
    emergency = 0
    interruption = 10
    foreground = 20
    manual_action = 30
    model_action = 40
    background = 50


class ConversationState(str, Enum):
    idle = "idle"
    listening = "listening"
    formulating = "formulating"
    speaking = "speaking"
    interrupted = "interrupted"


class BodyState(str, Enum):
    stationary = "stationary"
    looking = "looking"
    wandering = "wandering"
    executing_skill = "executing_skill"
    fault = "fault"


EmotionalEyeExpression = Literal[
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

EyeExpression = Literal[
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
    "fault",
    "listening",
    "thinking",
    "speaking",
]


class EyeState(BaseModel):
    base_expression: EmotionalEyeExpression = "neutral"
    effective_expression: EyeExpression = "neutral"
    base_duration_ms: int = 0
    base_expires_at: datetime | None = None


class CognitiveState(BaseModel):
    conversation: ConversationState = ConversationState.idle
    body: BodyState = BodyState.stationary
    active_goal: str | None = None
    connectivity: Literal["online", "degraded", "offline"] = "online"
    safety: Literal["normal", "stopped", "fault"] = "normal"
    eyes: EyeState = Field(default_factory=EyeState)
    active_correlation_id: str | None = None
    conversation_id: str = "default"
    updated_at: datetime = Field(default_factory=utc_now)


class BrainEvent(BaseModel):
    sequence: int | None = None
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    occurred_at: datetime = Field(default_factory=utc_now)
    source: EventSource
    correlation_id: str
    causation_id: str | None = None
    conversation_id: str = "default"
    priority: WorkPriority = WorkPriority.foreground
    payload: dict[str, Any] = Field(default_factory=dict)


class ActionIntent(BaseModel):
    action: dict[str, Any]
    origin: EventSource
    reason: str = ""
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    causation_id: str | None = None
    conversation_id: str = "default"
    priority: WorkPriority = WorkPriority.model_action


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    correlation_id: str
    sequence: int

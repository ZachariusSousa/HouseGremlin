from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


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


class SceneEntity(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: tuple[float, float, float, float] | None = None

    @field_validator("bounding_box")
    @classmethod
    def _validate_box(cls, box: tuple[float, float, float, float] | None):
        if box is not None and any(value < 0.0 or value > 1.0 for value in box):
            raise ValueError("bounding_box coordinates must be normalized between 0 and 1")
        return box


class SceneSnapshot(BaseModel):
    frame_id: str
    observed_at: datetime
    trigger: Literal["awareness", "explicit"]
    summary: str = Field(min_length=1, max_length=600)
    entities: list[SceneEntity] = Field(default_factory=list, max_length=30)
    novelty: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    model: str
    latency_ms: float = Field(ge=0.0)
    expires_at: datetime


class WorldState(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    snapshot_ids: list[str] = Field(default_factory=list)
    summary: str = "unknown"
    entities: list[SceneEntity] = Field(default_factory=list)


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

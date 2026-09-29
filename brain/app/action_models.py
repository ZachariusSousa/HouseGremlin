from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .brain_models import EyeExpression


class StrictActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MovementAction(StrictActionModel):
    direction: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)
    duration_ms: int | None = Field(default=None, ge=0)


class HeadAction(StrictActionModel):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)
    pan_delta: int | None = Field(default=None, ge=-80, le=80)
    tilt_delta: int | None = Field(default=None, ge=-80, le=80)

    @model_validator(mode="after")
    def require_head_value(self):
        if all(
            value is None
            for value in (self.pan, self.tilt, self.pan_delta, self.tilt_delta)
        ):
            raise ValueError("at least one head value is required")
        return self


class EyeAction(StrictActionModel):
    expression: EyeExpression
    duration_ms: int | None = Field(default=None, ge=0, le=10000)


class RobotActionRequest(StrictActionModel):
    movement: MovementAction | None = None
    head: HeadAction | None = None
    eyes: EyeAction | None = None

    @model_validator(mode="after")
    def require_action(self):
        if not any((self.movement, self.head, self.eyes)):
            raise ValueError("at least one robot action is required")
        return self


class ActionChatOutput(StrictActionModel):
    response: str
    action: RobotActionRequest | None = None
    vision_question: str | None = None

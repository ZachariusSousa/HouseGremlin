from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from math import hypot
from time import monotonic
from typing import Any, Awaitable, Callable, Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from .brain_models import EventSource, WorkPriority
from .coordinator import BrainCoordinator
from .frame_broker import CameraFrame, FrameBroker


TrackingMode = Literal["off", "track"]
TrackingState = Literal["off", "searching", "tracking", "fault"]
HeadCommander = Callable[[int, int, int, bool], Awaitable[Any]]
MoveCommander = Callable[[str, int, int, int], Awaitable[Any]]

TRACKING_CAMERA_FPS = 5.0
TRACKING_CONTRAST_FACTOR = 1.20
FACE_HEIGHT_FRACTION = 0.16
TRACKING_SMOOTHING_MIN_ALPHA = 0.28
TRACKING_SMOOTHING_MAX_ALPHA = 0.78
TRACKING_SMOOTHING_SPEED_GAIN = 2.0
TRACKING_VELOCITY_ALPHA = 0.35
TRACKING_PREDICTION_HORIZON_SECONDS = 0.20
TRACKING_MAX_PREDICTION_LEAD = 0.10
HEAD_DEADBAND = 0.08
HEAD_GAIN = 40.0
HEAD_MAX_STEP_DEGREES = 18
BODY_PIVOT_HEAD_OFFSET_DEGREES = 24
BODY_PIVOT_CONFIRMATION_FRAMES = 2
BODY_PIVOT_SPEED = 170
CAMERA_HORIZONTAL_FOV_DEGREES = 62.0
BODY_TURN_MS_PER_DEGREE = 20.0
BODY_TURN_MIN_DURATION_MS = 300
BODY_TURN_MAX_DURATION_MS = 650
MISSED_DETECTIONS_BEFORE_LOST = 6


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PersonCandidate(BaseModel):
    label: Literal["person"] = "person"
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: tuple[float, float, float, float]

    @field_validator("bounding_box")
    @classmethod
    def validate_box(cls, box: tuple[float, float, float, float]):
        if any(value < 0.0 or value > 1.0 for value in box):
            raise ValueError("bounding_box coordinates must be normalized")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("bounding_box must have positive area")
        return box


class PersonObservation(PersonCandidate):
    track_id: int
    frame_id: str
    observed_at: datetime


class DetectorResult(BaseModel):
    frame_id: str
    captured_at: datetime
    model: str
    backend: str
    latency_ms: float = Field(ge=0.0)
    people: list[PersonCandidate] = Field(default_factory=list)


class TrackingStatus(BaseModel):
    available: bool
    reason: str | None = None
    enabled: bool
    state: TrackingState
    mode: TrackingMode
    target: PersonObservation | None = None
    head: dict[str, int]
    effective_camera_fps: float
    detector_latency_ms: float | None = None
    backend: str | None = None
    model: str | None = None
    started_at: datetime | None = None
    stop_reason: str | None = None


class RFDetrClient:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.available = False
        self.reason: str | None = "RF-DETR sidecar has not been probed"
        self.backend: str | None = None
        self.model: str | None = None

    async def probe(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/health")
                payload = response.json()
            self.available = bool(response.is_success and payload.get("available"))
            self.reason = None if self.available else str(payload.get("reason") or "RF-DETR is unavailable")
            self.backend = payload.get("backend")
            self.model = payload.get("model")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self.available = False
            self.reason = f"RF-DETR sidecar unavailable: {exc}"
        return self.available

    async def detect(self, frame: CameraFrame, content: bytes, threshold: float) -> DetectorResult:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/detect",
                    content=content,
                    headers={
                        "Content-Type": "image/jpeg",
                        "X-Robit-Frame-Id": frame.frame_id,
                        "X-Robit-Captured-At": frame.captured_at.isoformat(),
                        "X-Robit-Threshold": str(threshold),
                    },
                )
                response.raise_for_status()
            payload = DetectorResult.model_validate(response.json())
            if payload.frame_id != frame.frame_id or payload.captured_at != frame.captured_at:
                raise ValueError("RF-DETR returned a result for a different camera frame")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self.available = False
            self.reason = f"RF-DETR inference unavailable: {exc}"
            raise RuntimeError(self.reason) from exc
        self.available = True
        self.reason = None
        self.backend = payload.backend
        self.model = payload.model
        return payload


def rotate_jpeg(content: bytes, degrees: int) -> bytes:
    from PIL import Image, ImageEnhance

    with Image.open(BytesIO(content)) as source:
        image = source.convert("RGB")
        if degrees % 360:
            image = image.rotate(degrees, expand=True)
        image = ImageEnhance.Contrast(image).enhance(TRACKING_CONTRAST_FACTOR)
        output = BytesIO()
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _face_aim_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Estimate face height from a person box without adding another model."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y1 + ((y2 - y1) * FACE_HEIGHT_FRACTION))


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _upright_score(box: tuple[float, float, float, float]) -> float:
    """Favor a head-and-torso box over a wide full-body box for gaze."""
    width = max(0.001, box[2] - box[0])
    height = max(0.001, box[3] - box[1])
    return min(2.0, height / width)


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _overlap_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    smaller_area = min(_box_area(first), _box_area(second))
    return intersection / smaller_area if smaller_area > 0.0 else 0.0


def _group_area(people: list[PersonCandidate]) -> float:
    return _box_area(
        (
            min(person.bounding_box[0] for person in people),
            min(person.bounding_box[1] for person in people),
            max(person.bounding_box[2] for person in people),
            max(person.bounding_box[3] for person in people),
        )
    )


def _body_turn_plan(
    head_pan: int,
    image_error_x: float,
    pan_sign: int,
) -> tuple[str, int]:
    """Estimate one chassis turn that lets the head return toward center."""
    head_offset = head_pan - 90
    residual_camera_angle = image_error_x * CAMERA_HORIZONTAL_FOV_DEGREES * pan_sign
    desired_head_offset = head_offset + residual_camera_angle
    signed_body_turn = -(desired_head_offset * pan_sign)
    direction = "right" if signed_body_turn > 0 else "left"
    duration_ms = round(abs(signed_body_turn) * BODY_TURN_MS_PER_DEGREE)
    return (
        direction,
        max(
            BODY_TURN_MIN_DURATION_MS,
            min(BODY_TURN_MAX_DURATION_MS, duration_ms),
        ),
    )


class PersonTrackingService:
    def __init__(
        self,
        coordinator: BrainCoordinator,
        broker: FrameBroker,
        detector: RFDetrClient,
        head_command: HeadCommander,
        move_command: MoveCommander,
        *,
        enabled: bool = True,
        confidence: float = 0.40,
        rotate_degrees: int = 180,
        pan_sign: int = 1,
        tilt_sign: int = 1,
    ):
        self.coordinator = coordinator
        self.broker = broker
        self.detector = detector
        self.head_command = head_command
        self.move_command = move_command
        self.enabled = enabled
        self.confidence = confidence
        self.rotate_degrees = rotate_degrees
        self.pan_sign = 1 if pan_sign >= 0 else -1
        self.tilt_sign = 1 if tilt_sign >= 0 else -1

        self.mode: TrackingMode = "track" if enabled else "off"
        self.state: TrackingState = "searching" if enabled else "off"
        self.target: PersonObservation | None = None
        self.head_pan = 90
        self.head_tilt = 90
        self.started_at: datetime | None = utc_now() if enabled else None
        self.stop_reason: str | None = None
        self.detector_latency_ms: float | None = None
        self.backend: str | None = None
        self.model: str | None = None

        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._last_frame_id: str | None = None
        self._next_track_id = 1
        self._smoothed_center: tuple[float, float] | None = None
        self._last_raw_center: tuple[float, float] | None = None
        self._last_raw_observed_at: datetime | None = None
        self._aim_velocity = (0.0, 0.0)
        self._last_head_command_at = 0.0
        self._pivot_error_frames = 0
        self._last_pivot_at = 0.0
        self._missed_frames = 0
        self._search_recentered = False
        self._suspended_until = 0.0
        self._motion_generation = 0
        self._update_camera_rate()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="robit-person-tracking")

    async def shutdown(self) -> None:
        self._stopping = True
        self.broker.release_rate_lease("tracking")
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def enable(self) -> TrackingStatus:
        if not self.enabled:
            raise RuntimeError("Person tracking is disabled")
        self._motion_generation += 1
        self._suspended_until = 0.0
        self.mode = "track"
        self.state = "tracking" if self.target else "searching"
        self.started_at = utc_now()
        self.stop_reason = None
        self._pivot_error_frames = 0
        self._missed_frames = 0
        self._search_recentered = False
        self._update_camera_rate()
        self._record("tracking.started", {"mode": "track"})
        return self.status()

    async def stop(self, reason: str = "requested") -> TrackingStatus:
        self._motion_generation += 1
        self.mode = "off"
        self.state = "off"
        self.target = None
        self._reset_aim_filter()
        self._pivot_error_frames = 0
        self._missed_frames = 0
        self._search_recentered = False
        self.stop_reason = reason
        self._update_camera_rate()
        self._record("tracking.stopped", {"reason": reason})
        return self.status()

    def suspend_for_manual_control(self, seconds: float = 2.0) -> None:
        self._motion_generation += 1
        self._suspended_until = max(self._suspended_until, monotonic() + seconds)
        self._reset_aim_filter()
        self._pivot_error_frames = 0
        self._missed_frames = 0
        self._search_recentered = False

    def sync_head_position(self, pan: int | None, tilt: int | None) -> None:
        if isinstance(pan, int) and not isinstance(pan, bool):
            self.head_pan = max(55, min(135, pan))
        if isinstance(tilt, int) and not isinstance(tilt, bool):
            self.head_tilt = max(35, min(115, tilt))
        self._reset_aim_filter()
        self._pivot_error_frames = 0
        self._missed_frames = 0

    def emergency_disable(self) -> None:
        self._motion_generation += 1
        self.mode = "off"
        self.state = "off"
        self.target = None
        self._reset_aim_filter()
        self._pivot_error_frames = 0
        self._missed_frames = 0
        self.stop_reason = "emergency stop"
        self._suspended_until = float("inf")
        self._update_camera_rate()

    def motion_authorized(self, generation: int, *, body: bool, allow_search: bool = False) -> bool:
        target = self.target
        if (
            generation != self._motion_generation
            or not self.enabled
            or self.mode == "off"
            or monotonic() < self._suspended_until
        ):
            return False
        if not body and allow_search:
            return self.state == "searching" and target is None
        if (
            self.state != "tracking"
            or target is None
            or (utc_now() - target.observed_at).total_seconds() >= 0.5
        ):
            return False
        return not body or self.mode == "track"

    def _command_completed(self, generation: int, result: Any) -> bool:
        if generation != self._motion_generation:
            return False
        if isinstance(result, dict) and (result.get("skipped") or result.get("ok") is False):
            return False
        return True

    def status(self) -> TrackingStatus:
        return TrackingStatus(
            available=self.detector.available,
            reason=self.detector.reason,
            enabled=self.enabled and self.mode != "off",
            state=self.state,
            mode=self.mode,
            target=self.target,
            head={"pan": self.head_pan, "tilt": self.head_tilt},
            effective_camera_fps=self.broker.effective_fps,
            detector_latency_ms=self.detector_latency_ms,
            backend=self.backend or self.detector.backend,
            model=self.model or self.detector.model,
            started_at=self.started_at,
            stop_reason=self.stop_reason,
        )

    async def process_result(self, result: DetectorResult) -> None:
        self.detector.available = True
        self.detector.reason = None
        self.detector_latency_ms = result.latency_ms
        self.backend = result.backend
        self.model = result.model
        if not self.enabled or self.mode == "off":
            self.target = None
            self.state = "off"
            self._reset_aim_filter()
            self._pivot_error_frames = 0
            self._missed_frames = 0
            self._update_camera_rate()
            return
        if (utc_now() - result.captured_at).total_seconds() >= 0.5:
            await self._handle_missing_target()
            return
        candidate = self._select_candidate(result.people)
        if candidate is None:
            await self._handle_missing_target()
            return

        self._missed_frames = 0
        self._search_recentered = False
        self._motion_generation += 1
        previous = self.target
        track_id = previous.track_id if previous is not None else self._next_track_id
        if previous is None:
            self._next_track_id += 1
        self.target = PersonObservation(
            **candidate.model_dump(),
            track_id=track_id,
            frame_id=result.frame_id,
            observed_at=result.captured_at,
        )
        self.state = "tracking"
        if previous is None:
            self._record("tracking.target_acquired", {"track_id": track_id, "confidence": candidate.confidence})
        self._update_camera_rate()
        if monotonic() >= self._suspended_until:
            await self._update_head_and_body()

    def _select_candidate(self, people: list[PersonCandidate]) -> PersonCandidate | None:
        people = [person for person in people if person.confidence >= self.confidence]
        if not people:
            return None

        if self.target is not None:
            previous_box = self.target.bounding_box
            previous_center = _box_center(previous_box)

            def latch_score(person: PersonCandidate) -> float:
                center = _box_center(person.bounding_box)
                center_distance = hypot(center[0] - previous_center[0], center[1] - previous_center[1])
                return (
                    (_box_iou(previous_box, person.bounding_box) * 2.0)
                    - center_distance
                    + (person.confidence * 0.1)
                    + (_upright_score(person.bounding_box) * 0.08)
                )

            candidate = max(people, key=latch_score)
            candidate_center = _box_center(candidate.bounding_box)
            candidate_distance = hypot(
                candidate_center[0] - previous_center[0],
                candidate_center[1] - previous_center[1],
            )
            if _box_iou(previous_box, candidate.bounding_box) < 0.10 and candidate_distance > 0.20:
                return None
            return candidate

        groups: list[list[PersonCandidate]] = []
        for person in people:
            matching_group = next(
                (
                    group
                    for group in groups
                    if any(
                        _overlap_over_smaller(person.bounding_box, member.bounding_box) >= 0.55
                        for member in group
                    )
                ),
                None,
            )
            if matching_group is None:
                groups.append([person])
            else:
                matching_group.append(person)

        def group_score(group: list[PersonCandidate]) -> float:
            group_center_x = sum(_box_center(person.bounding_box)[0] for person in group) / len(group)
            return (
                max(person.confidence for person in group)
                + ((_group_area(group) ** 0.5) * 0.80)
                + ((len(group) - 1) * 0.08)
                - (abs(group_center_x - 0.5) * 0.10)
            )

        selected_group = max(groups, key=group_score)

        def initial_score(person: PersonCandidate) -> float:
            center = _box_center(person.bounding_box)
            center_distance = hypot(center[0] - 0.5, center[1] - 0.5)
            return (
                person.confidence
                + (_upright_score(person.bounding_box) * 0.18)
                + (_box_area(person.bounding_box) ** 0.5 * 0.03)
                - (center_distance * 0.15)
            )

        return max(selected_group, key=initial_score)

    async def _handle_missing_target(self) -> None:
        if self.target is None:
            self.state = "searching" if self.mode == "track" else "off"
            self._update_camera_rate()
            await self._recenter_while_searching()
            return

        self._motion_generation += 1
        self._missed_frames += 1
        self._pivot_error_frames = 0
        if self._missed_frames < MISSED_DETECTIONS_BEFORE_LOST:
            self.state = "tracking"
            return

        self.state = "searching"
        self._record("tracking.target_lost", {"track_id": self.target.track_id})
        self.target = None
        self._reset_aim_filter()
        self._missed_frames = 0
        self.state = "searching" if self.mode == "track" else "off"
        self._update_camera_rate()
        await self._recenter_while_searching()

    async def _recenter_while_searching(self) -> None:
        """Recover from losing a person while the camera is parked at an end stop."""
        if (
            self.mode != "track"
            or self.target is not None
            or self._search_recentered
            or monotonic() < self._suspended_until
        ):
            return
        if abs(self.head_pan - 90) <= 5:
            self._search_recentered = True
            return

        generation = self._motion_generation
        result = await self.head_command(90, self.head_tilt, generation, True)
        if self._command_completed(generation, result):
            reported_pan = result.get("pan") if isinstance(result, dict) else None
            self.head_pan = (
                max(55, min(135, reported_pan))
                if isinstance(reported_pan, int) and not isinstance(reported_pan, bool)
                else 90
            )
            self._last_head_command_at = monotonic()
            self._search_recentered = True
            self._record("tracking.search_recentered", {"pan": self.head_pan, "tilt": self.head_tilt})

    async def _update_head_and_body(self) -> None:
        if self.target is None:
            return
        raw_center = _face_aim_point(self.target.bounding_box)
        observed_at = self.target.observed_at
        if self._last_raw_center is not None and self._last_raw_observed_at is not None:
            elapsed = (observed_at - self._last_raw_observed_at).total_seconds()
            if 0.05 <= elapsed <= 1.0:
                measured_velocity = (
                    (raw_center[0] - self._last_raw_center[0]) / elapsed,
                    (raw_center[1] - self._last_raw_center[1]) / elapsed,
                )
                velocity_alpha = TRACKING_VELOCITY_ALPHA
                self._aim_velocity = (
                    (velocity_alpha * measured_velocity[0])
                    + ((1.0 - velocity_alpha) * self._aim_velocity[0]),
                    (velocity_alpha * measured_velocity[1])
                    + ((1.0 - velocity_alpha) * self._aim_velocity[1]),
                )
            else:
                self._aim_velocity = (0.0, 0.0)
        self._last_raw_center = raw_center
        self._last_raw_observed_at = observed_at

        predicted_center = (
            raw_center[0]
            + max(
                -TRACKING_MAX_PREDICTION_LEAD,
                min(
                    TRACKING_MAX_PREDICTION_LEAD,
                    self._aim_velocity[0] * TRACKING_PREDICTION_HORIZON_SECONDS,
                ),
            ),
            raw_center[1]
            + max(
                -TRACKING_MAX_PREDICTION_LEAD,
                min(
                    TRACKING_MAX_PREDICTION_LEAD,
                    self._aim_velocity[1] * TRACKING_PREDICTION_HORIZON_SECONDS,
                ),
            ),
        )
        predicted_center = (
            max(0.0, min(1.0, predicted_center[0])),
            max(0.0, min(1.0, predicted_center[1])),
        )
        if self._smoothed_center is None:
            self._smoothed_center = predicted_center
        else:
            movement_speed = hypot(*self._aim_velocity)
            alpha = max(
                TRACKING_SMOOTHING_MIN_ALPHA,
                min(
                    TRACKING_SMOOTHING_MAX_ALPHA,
                    TRACKING_SMOOTHING_MIN_ALPHA
                    + (movement_speed * TRACKING_SMOOTHING_SPEED_GAIN),
                ),
            )
            self._smoothed_center = (
                (alpha * predicted_center[0]) + ((1.0 - alpha) * self._smoothed_center[0]),
                (alpha * predicted_center[1]) + ((1.0 - alpha) * self._smoothed_center[1]),
            )
        error_x = self._smoothed_center[0] - 0.5
        error_y = self._smoothed_center[1] - 0.5
        now = monotonic()

        if now - self._last_head_command_at >= 0.15 and (
            abs(error_x) > HEAD_DEADBAND or abs(error_y) > HEAD_DEADBAND
        ):
            pan_step = (
                max(-HEAD_MAX_STEP_DEGREES, min(HEAD_MAX_STEP_DEGREES, round(error_x * HEAD_GAIN)))
                if abs(error_x) > HEAD_DEADBAND
                else 0
            )
            tilt_step = (
                max(-HEAD_MAX_STEP_DEGREES, min(HEAD_MAX_STEP_DEGREES, round(error_y * HEAD_GAIN)))
                if abs(error_y) > HEAD_DEADBAND
                else 0
            )
            next_pan = max(55, min(135, self.head_pan + (pan_step * self.pan_sign)))
            next_tilt = max(35, min(115, self.head_tilt + (tilt_step * self.tilt_sign)))
            generation = self._motion_generation
            result = await self.head_command(next_pan, next_tilt, generation, False)
            if self._command_completed(generation, result):
                reported_pan = result.get("pan") if isinstance(result, dict) else None
                reported_tilt = result.get("tilt") if isinstance(result, dict) else None
                self.head_pan = (
                    max(55, min(135, reported_pan))
                    if isinstance(reported_pan, int) and not isinstance(reported_pan, bool)
                    else next_pan
                )
                self.head_tilt = (
                    max(35, min(115, reported_tilt))
                    if isinstance(reported_tilt, int) and not isinstance(reported_tilt, bool)
                    else next_tilt
                )
                self._last_head_command_at = monotonic()

        if self.mode != "track":
            self._pivot_error_frames = 0
            return
        if (self.backend or "").startswith("cpu") and (self.detector_latency_ms or 0.0) > 250.0:
            self._pivot_error_frames = 0
            return
        head_offset = self.head_pan - 90
        if abs(head_offset) >= BODY_PIVOT_HEAD_OFFSET_DEGREES or abs(error_x) > 0.30:
            self._pivot_error_frames += 1
        else:
            self._pivot_error_frames = 0
        if (
            self._pivot_error_frames >= BODY_PIVOT_CONFIRMATION_FRAMES
            and now - self._last_pivot_at >= 1.0
        ):
            direction, duration_ms = _body_turn_plan(
                self.head_pan,
                error_x,
                self.pan_sign,
            )
            generation = self._motion_generation
            result = await self.move_command(
                direction,
                BODY_PIVOT_SPEED,
                duration_ms,
                generation,
            )
            if self._command_completed(generation, result):
                self._last_pivot_at = monotonic()
                self._pivot_error_frames = 0
                head_result = await self.head_command(
                    90,
                    self.head_tilt,
                    generation,
                    False,
                )
                if self._command_completed(generation, head_result):
                    reported_pan = head_result.get("pan") if isinstance(head_result, dict) else None
                    self.head_pan = (
                        max(55, min(135, reported_pan))
                        if isinstance(reported_pan, int) and not isinstance(reported_pan, bool)
                        else 90
                    )
                    self._last_head_command_at = monotonic()

    def _reset_aim_filter(self) -> None:
        self._smoothed_center = None
        self._last_raw_center = None
        self._last_raw_observed_at = None
        self._aim_velocity = (0.0, 0.0)

    async def _run(self) -> None:
        while not self._stopping:
            if not self.enabled or self.mode == "off":
                await asyncio.sleep(0.25)
                continue
            if not self.detector.available and not await self.detector.probe():
                self.state = "fault"
                await asyncio.sleep(5.0)
                continue
            try:
                frame = await self.broker.get_frame()
                if frame.frame_id == self._last_frame_id:
                    await asyncio.sleep(0.05)
                    continue
                self._last_frame_id = frame.frame_id
                content = await asyncio.to_thread(rotate_jpeg, frame.content, self.rotate_degrees)
                result = await self.detector.detect(frame, content, self.confidence)
                await self.process_result(result)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state = "fault"
                self.detector.reason = str(exc)
                self._record("tracking.faulted", {"error": str(exc)})
                await asyncio.sleep(1.0)

    def _update_camera_rate(self) -> None:
        if not self.enabled or self.mode == "off":
            self.broker.release_rate_lease("tracking")
            return
        if (self.backend or "").startswith("cpu") and (self.detector_latency_ms or 0.0) > 250.0:
            self.broker.set_rate_lease("tracking", 1.0)
            return
        self.broker.set_rate_lease("tracking", TRACKING_CAMERA_FPS)

    def _record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.coordinator.record(
            event_type,
            EventSource.system,
            self.coordinator.state.active_correlation_id or self.coordinator.new_correlation_id(),
            payload,
            WorkPriority.background,
        )

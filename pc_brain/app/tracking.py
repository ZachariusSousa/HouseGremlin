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
TrackingState = Literal[
    "off",
    "searching",
    "acquiring",
    "tracking",
    "repositioning",
    "settling",
    "lost",
    "suspended",
    "fault",
]
HeadCommander = Callable[[int, int, int, bool], Awaitable[Any]]
MoveCommander = Callable[[str, int, int, int], Awaitable[Any]]
HeadStateProvider = Callable[[], dict[str, Any]]
TrackingStateListener = Callable[["TrackingStatus"], Awaitable[None] | None]

SEARCH_DETECTOR_FPS = 2.0
STABLE_DETECTOR_FPS = 1.0
VOICE_DETECTOR_FPS = 0.5
TRACKING_CAMERA_FPS = 2.0
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
BODY_PIVOT_CONFIRMATION_SECONDS = 1.5
BODY_PIVOT_SPEED = 140
CAMERA_HORIZONTAL_FOV_DEGREES = 62.0
BODY_TURN_MS_PER_DEGREE = 20.0
BODY_TURN_MIN_DURATION_MS = 300
BODY_TURN_MAX_DURATION_MS = 500
TARGET_MISSING_GRACE_SECONDS = 2.5
LOST_NEUTRAL_SECONDS = 5.0
PIVOT_SETTLING_SECONDS = 1.0
OPPOSITE_PIVOT_BLOCK_SECONDS = 5.0
ACQUIRE_REQUIRED_DETECTIONS = 2
ACQUIRE_WINDOW_SECONDS = 2.0
SWITCH_REQUIRED_DETECTIONS = 3
ESTIMATOR_ALPHA = 0.45
ESTIMATOR_BETA = 0.12
MAX_PREDICTION_SECONDS = 0.5


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
    queue_ms: float = Field(default=0.0, ge=0.0)
    request_bytes: int = Field(default=0, ge=0)
    people: list[PersonCandidate] = Field(default_factory=list)


class TrackingStatus(BaseModel):
    available: bool
    reason: str | None = None
    enabled: bool
    state: TrackingState
    mode: TrackingMode
    target: PersonObservation | None = None
    head: dict[str, int]
    desired_head: dict[str, int] = Field(default_factory=dict)
    actual_head: dict[str, int] = Field(default_factory=dict)
    effective_camera_fps: float
    detector_latency_ms: float | None = None
    detector_queue_ms: float | None = None
    backend: str | None = None
    model: str | None = None
    started_at: datetime | None = None
    stop_reason: str | None = None
    target_age_seconds: float | None = None
    target_confidence: float | None = None
    detector_cadence_fps: float = 0.0
    estimator_confidence: float = 0.0
    settling_remaining_seconds: float = 0.0
    last_pivot: dict[str, Any] | None = None
    command_counts: dict[str, int] = Field(default_factory=dict)
    association_result: str | None = None


class RFDetrClient:
    def __init__(self, base_url: str, timeout_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.available = False
        self.reason: str | None = "RF-DETR sidecar has not been probed"
        self.backend: str | None = None
        self.model: str | None = None
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def probe(self) -> bool:
        try:
            response = await self._http().get(f"{self.base_url}/health")
            payload = response.json()
            self.available = bool(response.is_success and payload.get("available"))
            self.reason = None if self.available else str(payload.get("reason") or "RF-DETR is unavailable")
            self.backend = payload.get("backend")
            self.model = payload.get("model")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            self.available = False
            self.reason = (
                f"RF-DETR sidecar unavailable during probe "
                f"({type(exc).__name__}): {exc}"
            )
        return self.available

    async def detect(self, frame: CameraFrame, content: bytes, threshold: float) -> DetectorResult:
        try:
            response = await self._http().post(
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
            self.reason = (
                f"RF-DETR inference unavailable during detect "
                f"({type(exc).__name__}): {exc}"
            )
            raise RuntimeError(self.reason) from exc
        self.available = True
        self.reason = None
        self.backend = payload.backend
        self.model = payload.model
        return payload


class TrackingWorkloadPreempted(RuntimeError):
    pass


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
        search_fps: float = SEARCH_DETECTOR_FPS,
        stable_fps: float = STABLE_DETECTOR_FPS,
        voice_fps: float = VOICE_DETECTOR_FPS,
        acquire_required_detections: int = ACQUIRE_REQUIRED_DETECTIONS,
        acquire_window_seconds: float = ACQUIRE_WINDOW_SECONDS,
        missing_grace_seconds: float = TARGET_MISSING_GRACE_SECONDS,
        pivot_confirmation_seconds: float = BODY_PIVOT_CONFIRMATION_SECONDS,
        settling_seconds: float = PIVOT_SETTLING_SECONDS,
        opposite_pivot_block_seconds: float = OPPOSITE_PIVOT_BLOCK_SECONDS,
        lost_neutral_seconds: float = LOST_NEUTRAL_SECONDS,
        head_state_provider: HeadStateProvider | None = None,
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
        self.search_fps = max(0.1, search_fps)
        self.stable_fps = max(0.1, stable_fps)
        self.voice_fps = max(0.1, voice_fps)
        self.acquire_required_detections = max(1, acquire_required_detections)
        self.acquire_window_seconds = max(0.1, acquire_window_seconds)
        self.missing_grace_seconds = max(0.1, missing_grace_seconds)
        self.pivot_confirmation_seconds = max(0.1, pivot_confirmation_seconds)
        self.settling_seconds = max(0.1, settling_seconds)
        self.opposite_pivot_block_seconds = max(
            0.1, opposite_pivot_block_seconds
        )
        self.lost_neutral_seconds = max(0.1, lost_neutral_seconds)
        self.head_state_provider = head_state_provider

        self.mode: TrackingMode = "track" if enabled else "off"
        self.state: TrackingState = "searching" if enabled else "off"
        self.target: PersonObservation | None = None
        self.head_pan = 90
        self.head_tilt = 90
        self.started_at: datetime | None = utc_now() if enabled else None
        self.stop_reason: str | None = None
        self.detector_latency_ms: float | None = None
        self.detector_queue_ms: float | None = None
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
        self._pivot_excessive_since: float | None = None
        self._last_pivot_at = 0.0
        self._last_pivot_direction: str | None = None
        self._settling_until = 0.0
        self._missed_frames = 0
        self._last_fresh_detection_at = 0.0
        self._lost_since: float | None = None
        self._neutral_issued = False
        self._acquire_candidate: PersonCandidate | None = None
        self._acquire_count = 0
        self._acquire_started_at = 0.0
        self._switch_candidate: PersonCandidate | None = None
        self._switch_count = 0
        self._estimator_center: tuple[float, float] | None = None
        self._estimator_velocity = (0.0, 0.0)
        self._estimator_box: tuple[float, float, float, float] | None = None
        self._estimator_at = 0.0
        self._estimator_confidence = 0.0
        self._association_result: str | None = None
        self._detector_cadence_fps = self.search_fps
        self._last_detection_started_at = 0.0
        self._last_sample_at = 0.0
        self._command_counts = {"head": 0, "pivot": 0, "neutral": 0}
        self._state_listeners: list[TrackingStateListener] = []
        self._search_recentered = False
        self._suspended_until = 0.0
        self._motion_generation = 0
        self._update_camera_rate()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._run(), name="robit-person-tracking")

    def subscribe_state(self, listener: TrackingStateListener) -> None:
        if listener not in self._state_listeners:
            self._state_listeners.append(listener)

    async def shutdown(self) -> None:
        self._stopping = True
        self.broker.release_rate_lease("tracking")
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        close = getattr(self.detector, "close", None)
        if close is not None:
            await close()

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
        self._lost_since = None
        self._neutral_issued = False
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

    def suspend_for_manual_control(self, seconds: float = 3.0) -> None:
        self._motion_generation += 1
        self._suspended_until = max(self._suspended_until, monotonic() + seconds)
        self.target = None
        self.state = "suspended" if self.mode != "off" else "off"
        self._reset_aim_filter()
        self._reset_association()
        self._pivot_error_frames = 0
        self._missed_frames = 0
        self._search_recentered = False
        try:
            asyncio.get_running_loop().create_task(self._notify_state_listeners())
        except RuntimeError:
            pass

    def sync_head_position(self, pan: int | None, tilt: int | None) -> None:
        if isinstance(pan, int) and not isinstance(pan, bool):
            self.head_pan = max(55, min(135, pan))
        if isinstance(tilt, int) and not isinstance(tilt, bool):
            self.head_tilt = max(35, min(115, tilt))
        self._reset_aim_filter()
        self._pivot_error_frames = 0
        self._missed_frames = 0

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
            return self.state in {"searching", "acquiring", "lost"} and target is None
        if (
            self.state not in {"tracking", "repositioning", "settling"}
            or target is None
            or (utc_now() - target.observed_at).total_seconds() >= 0.5
        ):
            return False
        return not body or (
            self.mode == "track"
            and self.state in {"tracking", "repositioning"}
            and monotonic() >= self._settling_until
        )

    def _command_completed(self, generation: int, result: Any) -> bool:
        if generation != self._motion_generation:
            return False
        if isinstance(result, dict) and (result.get("skipped") or result.get("ok") is False):
            return False
        return True

    def status(self) -> TrackingStatus:
        telemetry = self.head_state_provider() if self.head_state_provider else {}
        desired_head = {
            "pan": int(telemetry.get("pan_target", self.head_pan)),
            "tilt": int(telemetry.get("tilt_target", self.head_tilt)),
        }
        actual_head = {
            "pan": int(telemetry.get("pan_actual", self.head_pan)),
            "tilt": int(telemetry.get("tilt_actual", self.head_tilt)),
        }
        target_age = (
            max(0.0, (utc_now() - self.target.observed_at).total_seconds())
            if self.target is not None
            else None
        )
        return TrackingStatus(
            available=self.detector.available,
            reason=self.detector.reason,
            enabled=self.enabled and self.mode != "off",
            state=self.state,
            mode=self.mode,
            target=self.target,
            head={"pan": self.head_pan, "tilt": self.head_tilt},
            desired_head=desired_head,
            actual_head=actual_head,
            effective_camera_fps=self.broker.effective_fps,
            detector_latency_ms=self.detector_latency_ms,
            detector_queue_ms=self.detector_queue_ms,
            backend=self.backend or self.detector.backend,
            model=self.model or self.detector.model,
            started_at=self.started_at,
            stop_reason=self.stop_reason,
            target_age_seconds=target_age,
            target_confidence=self.target.confidence if self.target else None,
            detector_cadence_fps=self._detector_cadence_fps,
            estimator_confidence=self._estimator_confidence,
            settling_remaining_seconds=max(0.0, self._settling_until - monotonic()),
            last_pivot=(
                {
                    "direction": self._last_pivot_direction,
                    "age_seconds": max(0.0, monotonic() - self._last_pivot_at),
                }
                if self._last_pivot_direction
                else None
            ),
            command_counts=dict(self._command_counts),
            association_result=self._association_result,
        )

    async def process_result(self, result: DetectorResult) -> None:
        self.detector.available = True
        self.detector.reason = None
        self.detector_latency_ms = result.latency_ms
        self.detector_queue_ms = result.queue_ms
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
        now = monotonic()
        if now < self._suspended_until:
            self.state = "suspended"
            return
        if self.state == "suspended":
            self.state = "searching"
            self._reset_association()
        if (utc_now() - result.captured_at).total_seconds() >= 0.5:
            self._association_result = "stale_detection"
            self._record_sample(result, None)
            await self._handle_missing_target()
            return
        candidate = self._select_candidate(result.people)
        if candidate is None:
            if self._association_result != "switch_pending":
                self._association_result = "missing"
            self._record_sample(result, None)
            await self._handle_missing_target()
            return

        previous = self.target
        switched = self._association_result == "switched"
        if previous is None:
            if not self._consistent_candidate(self._acquire_candidate, candidate):
                self._acquire_candidate = candidate
                self._acquire_count = 1
                self._acquire_started_at = now
            elif now - self._acquire_started_at <= self.acquire_window_seconds:
                self._acquire_candidate = candidate
                self._acquire_count += 1
            else:
                self._acquire_candidate = candidate
                self._acquire_count = 1
                self._acquire_started_at = now
            self._association_result = "acquiring"
            self.state = "acquiring"
            if self._acquire_count < self.acquire_required_detections:
                self._update_camera_rate()
                self._record_sample(result, candidate)
                return
            self._reset_association()

        self._missed_frames = 0
        self._last_fresh_detection_at = now
        self._lost_since = None
        self._neutral_issued = False
        self._search_recentered = False
        track_id = (
            previous.track_id
            if previous is not None and not switched
            else self._next_track_id
        )
        if previous is None or switched:
            self._next_track_id += 1
        self.target = PersonObservation(
            **candidate.model_dump(),
            track_id=track_id,
            frame_id=result.frame_id,
            observed_at=result.captured_at,
        )
        self._update_estimator(candidate, now)
        self._association_result = (
            "switched" if switched else ("maintained" if previous is not None else "acquired")
        )
        self.state = "tracking"
        if previous is None:
            self._record("tracking.target_acquired", {"track_id": track_id, "confidence": candidate.confidence})
            await self._notify_state_listeners()
        self._update_camera_rate()
        await self._update_head_and_body()
        self._record_sample(result, candidate)

    @staticmethod
    def _consistent_candidate(
        previous: PersonCandidate | None,
        candidate: PersonCandidate,
    ) -> bool:
        if previous is None:
            return False
        previous_center = _box_center(previous.bounding_box)
        candidate_center = _box_center(candidate.bounding_box)
        return (
            _box_iou(previous.bounding_box, candidate.bounding_box) >= 0.15
            or hypot(
                previous_center[0] - candidate_center[0],
                previous_center[1] - candidate_center[1],
            )
            <= 0.15
        )

    def _reset_association(self) -> None:
        self._acquire_candidate = None
        self._acquire_count = 0
        self._acquire_started_at = 0.0
        self._switch_candidate = None
        self._switch_count = 0

    def _update_estimator(self, candidate: PersonCandidate, now: float) -> None:
        measured = _face_aim_point(candidate.bounding_box)
        if self._estimator_center is None or self._estimator_at <= 0.0:
            self._estimator_center = measured
            self._estimator_velocity = (0.0, 0.0)
        else:
            elapsed = max(0.05, min(MAX_PREDICTION_SECONDS, now - self._estimator_at))
            predicted = (
                self._estimator_center[0] + self._estimator_velocity[0] * elapsed,
                self._estimator_center[1] + self._estimator_velocity[1] * elapsed,
            )
            residual = (measured[0] - predicted[0], measured[1] - predicted[1])
            self._estimator_center = (
                predicted[0] + ESTIMATOR_ALPHA * residual[0],
                predicted[1] + ESTIMATOR_ALPHA * residual[1],
            )
            self._estimator_velocity = (
                self._estimator_velocity[0]
                + (ESTIMATOR_BETA / elapsed) * residual[0],
                self._estimator_velocity[1]
                + (ESTIMATOR_BETA / elapsed) * residual[1],
            )
        self._estimator_box = candidate.bounding_box
        self._estimator_at = now
        self._estimator_confidence = candidate.confidence

    def _record_sample(
        self,
        result: DetectorResult,
        candidate: PersonCandidate | None,
    ) -> None:
        now = monotonic()
        if now - self._last_sample_at < 1.0:
            return
        self._last_sample_at = now
        status = self.status()
        self._record(
            "tracking.sample",
            {
                "frame_id": result.frame_id,
                "box": candidate.bounding_box if candidate else None,
                "confidence": candidate.confidence if candidate else None,
                "association": self._association_result,
                "desired_head": status.desired_head,
                "actual_head": status.actual_head,
                "detector_latency_ms": result.latency_ms,
            },
        )

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
                superior = max(
                    people,
                    key=lambda person: (
                        person.confidence + (_box_area(person.bounding_box) ** 0.5 * 0.2)
                    ),
                )
                if (
                    superior.confidence >= self.target.confidence + 0.10
                    and self._consistent_candidate(self._switch_candidate, superior)
                ):
                    self._switch_count += 1
                else:
                    self._switch_candidate = superior
                    self._switch_count = 1
                if self._switch_count >= SWITCH_REQUIRED_DETECTIONS:
                    self._switch_candidate = None
                    self._switch_count = 0
                    self._association_result = "switched"
                    return superior
                self._association_result = "switch_pending"
                return None
            self._switch_candidate = None
            self._switch_count = 0
            self._association_result = "maintained"
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
        now = monotonic()
        if self.target is None:
            if self._lost_since is None:
                self._lost_since = now
            self.state = "lost" if self.mode == "track" else "off"
            self._update_camera_rate()
            if (
                self.mode == "track"
                and not self._neutral_issued
                and now - self._lost_since >= self.lost_neutral_seconds
                and now >= self._suspended_until
            ):
                generation = self._motion_generation
                result = await self.head_command(90, 90, generation, True)
                if self._command_completed(generation, result):
                    self.head_pan = 90
                    self.head_tilt = 90
                    self._neutral_issued = True
                    self._command_counts["neutral"] += 1
                    self.state = "searching"
            return

        self._missed_frames += 1
        self._pivot_error_frames = 0
        self._pivot_excessive_since = None
        missing_age = now - self._last_fresh_detection_at
        if missing_age <= self.missing_grace_seconds:
            self.state = "tracking"
            self._estimator_confidence *= 0.85
            return

        self._record("tracking.target_lost", {"track_id": self.target.track_id})
        self.target = None
        self._reset_aim_filter()
        self._reset_association()
        self._missed_frames = 0
        self._lost_since = now
        self._neutral_issued = False
        self.state = "lost" if self.mode == "track" else "off"
        self._update_camera_rate()
        await self._notify_state_listeners()

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
        estimator_age = max(0.0, min(MAX_PREDICTION_SECONDS, monotonic() - self._estimator_at))
        raw_center = (
            (
                self._estimator_center[0] + self._estimator_velocity[0] * estimator_age,
                self._estimator_center[1] + self._estimator_velocity[1] * estimator_age,
            )
            if self._estimator_center is not None
            else _face_aim_point(self.target.bounding_box)
        )
        raw_center = (
            max(0.0, min(1.0, raw_center[0])),
            max(0.0, min(1.0, raw_center[1])),
        )
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

        if now - self._last_head_command_at >= 0.25 and (
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
                self._command_counts["head"] += 1

        if self.mode != "track":
            self._pivot_error_frames = 0
            return
        if now < self._settling_until:
            self.state = "settling"
            self._pivot_excessive_since = None
            return
        if self.state == "settling":
            self.state = "tracking"
        if (self.backend or "").startswith("cpu") and (self.detector_latency_ms or 0.0) > 250.0:
            self._pivot_error_frames = 0
            return
        head_offset = self.head_pan - 90
        if abs(head_offset) >= BODY_PIVOT_HEAD_OFFSET_DEGREES or abs(error_x) > 0.30:
            if self._pivot_excessive_since is None:
                self._pivot_excessive_since = now
        else:
            self._pivot_excessive_since = None
        if (
            self._pivot_excessive_since is not None
            and now - self._pivot_excessive_since >= self.pivot_confirmation_seconds
        ):
            direction, duration_ms = _body_turn_plan(
                self.head_pan,
                error_x,
                self.pan_sign,
            )
            if (
                self._last_pivot_direction is not None
                and direction != self._last_pivot_direction
                and now - self._last_pivot_at < self.opposite_pivot_block_seconds
            ):
                return
            generation = self._motion_generation
            self.state = "repositioning"
            result = await self.move_command(
                direction,
                BODY_PIVOT_SPEED,
                duration_ms,
                generation,
            )
            if self._command_completed(generation, result):
                self._last_pivot_at = monotonic()
                self._last_pivot_direction = direction
                self._pivot_excessive_since = None
                self._settling_until = self._last_pivot_at + self.settling_seconds
                self._command_counts["pivot"] += 1
                self.state = "settling"
                self._reset_aim_filter()
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
        self._estimator_center = None
        self._estimator_velocity = (0.0, 0.0)
        self._estimator_box = None
        self._estimator_at = 0.0
        self._estimator_confidence = 0.0

    def _current_detector_fps(self) -> float:
        conversation = self.coordinator.state.conversation.value
        if conversation in {"listening", "speaking"}:
            return self.voice_fps
        if self.state in {"tracking", "repositioning", "settling"} and self.target:
            return self.stable_fps
        return self.search_fps

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
                if monotonic() < self._suspended_until:
                    self.state = "suspended"
                    await asyncio.sleep(0.1)
                    continue
                self._detector_cadence_fps = self._current_detector_fps()
                interval = 1.0 / self._detector_cadence_fps
                delay = self._last_detection_started_at + interval - monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._last_detection_started_at = monotonic()
                frame, content = await self.broker.get_rotated_jpeg(
                    self.rotate_degrees
                )
                if frame.frame_id == self._last_frame_id:
                    await asyncio.sleep(0.05)
                    continue
                self._last_frame_id = frame.frame_id
                result = await self._detect_with_workload_lease(frame, content)
                await self.process_result(result)
            except asyncio.CancelledError:
                raise
            except TrackingWorkloadPreempted:
                await asyncio.sleep(0)
                continue
            except Exception as exc:
                self.state = "fault"
                self.detector.reason = f"{type(exc).__name__}: {exc}"
                self._record(
                    "tracking.faulted",
                    {
                        "stage": "detection_loop",
                        "exception_class": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(1.0)

    async def _detect_with_workload_lease(
        self,
        frame: CameraFrame,
        content: bytes,
    ) -> DetectorResult:
        cancelled = asyncio.Event()

        async def run() -> DetectorResult:
            async with self.coordinator.resource_lease.acquire(
                WorkPriority.target_reacquisition,
                cancelled.set,
            ):
                if cancelled.is_set():
                    raise asyncio.CancelledError
                return await self.detector.detect(frame, content, self.confidence)

        job = asyncio.create_task(run(), name="rfdetr-request")
        try:
            return await job
        except asyncio.CancelledError:
            if asyncio.current_task() and asyncio.current_task().cancelling():
                raise
            self._association_result = "detector_preempted"
            raise TrackingWorkloadPreempted(
                "RF-DETR request preempted by foreground workload"
            )

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

    async def _notify_state_listeners(self) -> None:
        status = self.status()
        for listener in tuple(self._state_listeners):
            try:
                result = listener(status)
                if result is not None:
                    await result
            except (OSError, RuntimeError, ValueError):
                continue

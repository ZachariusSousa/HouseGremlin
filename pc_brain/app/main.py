import asyncio
import inspect
import importlib.util
import json
import logging
import re
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .audio_utils import ensure_data_dirs
from .brain_models import ActionIntent, ConversationState, EventSource, EyeExpression, WorkPriority
from .config import settings
from .control_channel import ActuatorBroker, ControlChannelClient
from .coordinator import BrainCoordinator
from .correlation import current_correlation_id
from .eye_controller import EMOTIONAL_EYE_EXPRESSIONS, EyeController
from .frame_broker import FrameBroker
from .journal import EventJournal
from .llm import OpenAICompatibleChatClient
from .realtime_gateway import RealtimeGateway, explicit_robot_action
from .robot_status import normalize_robot_status
from .telemetry import HostSampler, LlmHealthState, SystemTelemetryService
from .timing import timed
from .tracking import PersonTrackingService, RFDetrClient
from .vision import VisionService, VisionUnavailable


logger = logging.getLogger("uvicorn.error")
llm_client = OpenAICompatibleChatClient(settings)
llm_health = LlmHealthState(settings.llm_provider, settings.llm_model)
telemetry_service = SystemTelemetryService(
    host_sampler=HostSampler(),
    llm_health=llm_health,
    llm_probe=llm_client.probe_models,
    event_loop_lag=lambda: event_loop_lag_ms,
)
robot_request_lock = asyncio.Lock()
robot_http_client: httpx.AsyncClient | None = None
robot_status_cache: dict | None = None
robot_status_cache_at = 0.0
ROBOT_STATUS_CACHE_SECONDS = 15.0
CAMERA_READY_MAX_AGE_SECONDS = 5.0
robot_camera_request_lock = asyncio.Lock()
brain_journal: EventJournal | None = None
brain_coordinator: BrainCoordinator | None = None
realtime_gateway: RealtimeGateway | None = None
eye_controller: EyeController | None = None
frame_broker: FrameBroker | None = None
vision_service: VisionService | None = None
tracking_service: PersonTrackingService | None = None
control_channel: ControlChannelClient | None = None
actuator_broker: ActuatorBroker | None = None
event_loop_monitor_task: asyncio.Task[None] | None = None
event_loop_lag_ms: float = 0.0
event_loop_lag_peak_ms: float = 0.0


async def monitor_event_loop_lag() -> None:
    global event_loop_lag_ms, event_loop_lag_peak_ms
    interval = 0.1
    expected = time.monotonic() + interval
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        event_loop_lag_ms = max(0.0, (now - expected) * 1000.0)
        event_loop_lag_peak_ms = max(event_loop_lag_peak_ms, event_loop_lag_ms)
        if event_loop_lag_ms >= 25.0:
            logger.warning("event_loop.lag lag_ms=%.2f", event_loop_lag_ms)
        expected = now + interval


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot_http_client, event_loop_monitor_task
    actuators = None
    controller = None
    broker = None
    vision = None
    tracking = None
    telemetry_started = False
    ensure_data_dirs(settings.data_dir)
    get_brain_coordinator()
    try:
        robot_http_client = httpx.AsyncClient(timeout=settings.request_timeout)
        event_loop_monitor_task = asyncio.create_task(
            monitor_event_loop_lag(), name="event-loop-monitor"
        )
        telemetry_started = True
        await telemetry_service.start()
        actuators = get_actuator_broker()
        await actuators.start()
        controller = get_eye_controller()
        broker = get_frame_broker()
        vision = get_vision_service()
        tracking = get_tracking_service()
        await controller.start()
        await broker.start()
        try:
            robot_status = await robot_get("/api/status")
            tracking.sync_head_position(robot_status.get("pan"), robot_status.get("tilt"))
        except Exception as exc:
            logger.warning("tracking.head_sync_failed error=%s", exc)
        await tracking.start()
        await vision.start()
        if settings.warm_models:
            await llm_client.warmup()
        yield
    finally:
        async def shutdown_safely(name: str, component) -> None:
            if component is None:
                return
            try:
                await component.shutdown()
            except Exception as exc:
                logger.warning("lifespan.shutdown_failed component=%s error=%s", name, exc)

        await shutdown_safely("realtime_gateway", realtime_gateway)
        await shutdown_safely("vision", vision)
        await shutdown_safely("tracking", tracking)
        await shutdown_safely("frame_broker", broker)
        await shutdown_safely("eye_controller", controller)
        await shutdown_safely("actuator_broker", actuators)
        if telemetry_started:
            await shutdown_safely("telemetry", telemetry_service)
        if event_loop_monitor_task is not None:
            event_loop_monitor_task.cancel()
            await asyncio.gather(event_loop_monitor_task, return_exceptions=True)
            event_loop_monitor_task = None
        if brain_journal is not None:
            try:
                brain_journal.close()
            except Exception as exc:
                logger.warning("lifespan.shutdown_failed component=journal error=%s", exc)
        if robot_http_client is not None:
            try:
                await robot_http_client.aclose()
            except Exception as exc:
                logger.warning("lifespan.shutdown_failed component=http_client error=%s", exc)
            finally:
                robot_http_client = None


app = FastAPI(title="Robit PC Brain", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Robit-Frame-Id", "X-Robit-Captured-At", "X-Robit-Frame-Interval"],
)
WEB_CONTROL_INDEX = Path(__file__).resolve().parents[2] / "web_control" / "index.html"
REALTIME_EYE_POLICY = (
    "When the user explicitly asks Robit to move, stop, pan, or tilt its head, you MUST call robot_action. "
    "Never say that a physical action happened unless you made that tool call and received a successful result. "
    "You may optionally select a temporary emotional eye expression when a message genuinely warrants it. "
    "When the user explicitly asks you to show, make, try, or change an eye expression, you MUST call "
    "robot_action with the eyes field. Never claim an eye expression changed unless you made that tool call. "
    "Allowed emotional expressions are neutral, angry, cute, concerned, content, happy, startled, sleepy, "
    "curious, confused, suspicious, and wink. Operational listening, thinking, speaking, and fault eye states "
    "are automatic and must never be requested. When asked about the current view, call inspect_scene and use "
    "only its result. Never call movement or head actions in the same user turn after inspect_scene."
)


def effective_realtime_instructions() -> str:
    instructions = settings.realtime_instructions.strip()
    if "Operational listening, thinking, speaking, and fault eye states are automatic" in instructions:
        return instructions
    return f"{instructions} {REALTIME_EYE_POLICY}".strip()


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DriveCommand(StrictRequest):
    move: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)


class HeadCommand(StrictRequest):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)


class MovementAction(StrictRequest):
    direction: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)
    duration_ms: int | None = Field(default=None, ge=0)


class HeadAction(StrictRequest):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)
    pan_delta: int | None = Field(default=None, ge=-80, le=80)
    tilt_delta: int | None = Field(default=None, ge=-80, le=80)

    @model_validator(mode="after")
    def require_head_value(self):
        if all(value is None for value in (self.pan, self.tilt, self.pan_delta, self.tilt_delta)):
            raise ValueError("at least one head value is required")
        return self


class EyeAction(StrictRequest):
    expression: EyeExpression
    duration_ms: int | None = Field(default=None, ge=0, le=10000)


class RobotActionRequest(StrictRequest):
    movement: MovementAction | None = None
    head: HeadAction | None = None
    eyes: EyeAction | None = None

    @model_validator(mode="after")
    def require_action(self):
        if not any((self.movement, self.head, self.eyes)):
            raise ValueError("at least one robot action is required")
        return self


class ChatRequest(StrictRequest):
    text: str
    conversation_id: Literal["default"] = "default"


class ChatActionRequest(ChatRequest):
    pass


class PerceptionQueryRequest(StrictRequest):
    question: str = Field(min_length=1, max_length=500)
    fresh: bool = True


def get_brain_coordinator() -> BrainCoordinator:
    global brain_journal, brain_coordinator
    if brain_coordinator is None:
        brain_journal = EventJournal(
            settings.data_dir / "brain.db",
            queue_limit=getattr(settings, "journal_queue_limit", 1000),
        )
        brain_coordinator = BrainCoordinator(brain_journal)
    return brain_coordinator


async def handle_control_connection_change(connected: bool) -> None:
    coordinator = get_brain_coordinator()
    if connected:
        coordinator.clear_fault("control_transport")
        return
    channel = control_channel
    message = (
        channel.stats.last_error
        if channel is not None and channel.stats.last_error
        else "robot control channel disconnected"
    )
    coordinator.register_fault(
        "control_transport",
        "critical",
        message,
    )


def get_actuator_broker() -> ActuatorBroker:
    global control_channel, actuator_broker
    if actuator_broker is None:
        parsed = urlparse(settings.robot_base_url)
        control_channel = ControlChannelClient(
            parsed.hostname or "robit.local",
            getattr(settings, "control_tcp_port", 82),
            heartbeat_interval_seconds=getattr(
                settings, "control_heartbeat_interval_seconds", 1.0
            ),
            command_timeout_seconds=getattr(
                settings, "control_command_timeout_seconds", 1.5
            ),
        )
        actuator_broker = ActuatorBroker(
            control_channel,
            tracking_rate_hz=getattr(settings, "actuator_tracking_rate_hz", 4.0),
            manual_rate_hz=getattr(settings, "actuator_manual_rate_hz", 10.0),
            minimum_head_change_degrees=getattr(
                settings, "actuator_head_deadband_degrees", 2.0
            ),
            manual_lease_seconds=getattr(
                settings, "actuator_manual_lease_seconds", 3.0
            ),
        )
        control_channel.add_connection_listener(handle_control_connection_change)
    return actuator_broker


def get_eye_controller() -> EyeController:
    global eye_controller
    if eye_controller is None or eye_controller.coordinator is not get_brain_coordinator():
        eye_controller = EyeController(
            get_brain_coordinator(),
            robot_post,
            heartbeat_post=robot_heartbeat_post,
        )
    return eye_controller


def get_frame_broker() -> FrameBroker:
    global frame_broker
    if frame_broker is None:
        frame_broker = FrameBroker(
            fetch_robot_camera_frame,
            getattr(settings, "camera_frame_interval_seconds", 5.0),
            max_fps=2.0,
        )
    return frame_broker


def tracking_motion_authorized(
    generation: int,
    *,
    body: bool,
    allow_search: bool = False,
) -> bool:
    service = tracking_service
    return bool(
        service is not None
        and callable(getattr(service, "motion_authorized", None))
        and service.motion_authorized(generation, body=body, allow_search=allow_search)
    )


async def tracking_head_command(
    pan: int,
    tilt: int,
    generation: int,
    allow_search: bool = False,
) -> dict:
    if not tracking_motion_authorized(generation, body=False, allow_search=allow_search):
        return {"ok": False, "skipped": "tracking authorization expired"}
    return await get_actuator_broker().head_target(
        pan,
        tilt,
        source="tracking",
        wait_for_ack=False,
        authorization=lambda: tracking_motion_authorized(
            generation,
            body=False,
            allow_search=allow_search,
        ),
    )


async def tracking_move_command(direction: str, speed: int, duration_ms: int, generation: int) -> dict:
    if not tracking_motion_authorized(generation, body=True):
        return {"ok": False, "skipped": "tracking authorization expired"}
    correlation_id = get_brain_coordinator().new_correlation_id()
    intent = ActionIntent(
        action={"movement": {"direction": direction, "speed": speed, "duration_ms": duration_ms}},
        origin=EventSource.policy,
        correlation_id=correlation_id,
        reason="RF-DETR person tracking pivot",
        priority=WorkPriority.background,
    )

    async def execute(payload: dict) -> dict:
        if not tracking_motion_authorized(generation, body=True):
            return {"ok": False, "skipped": "tracking authorization expired"}
        movement = payload["movement"]
        return await get_actuator_broker().tracking_drive(
            movement["direction"],
            movement["speed"],
            movement["duration_ms"],
            authorization=lambda: tracking_motion_authorized(
                generation, body=True
            ),
        )

    return await get_brain_coordinator().execute_action(intent, execute)


def get_tracking_service() -> PersonTrackingService:
    global tracking_service
    coordinator = get_brain_coordinator()
    if tracking_service is None or getattr(tracking_service, "coordinator", coordinator) is not coordinator:
        tracking_service = PersonTrackingService(
            coordinator,
            get_frame_broker(),
            RFDetrClient(
                getattr(settings, "tracking_base_url", "http://127.0.0.1:8091"),
                getattr(settings, "tracking_request_timeout_seconds", 2.0),
            ),
            tracking_head_command,
            tracking_move_command,
            enabled=getattr(settings, "tracking_enabled", True),
            confidence=getattr(settings, "tracking_confidence", 0.40),
            rotate_degrees=getattr(settings, "camera_rotate_degrees", 180),
            pan_sign=getattr(settings, "tracking_pan_sign", 1),
            tilt_sign=getattr(settings, "tracking_tilt_sign", 1),
            search_fps=getattr(settings, "tracking_search_fps", 2.0),
            stable_fps=getattr(settings, "tracking_stable_fps", 1.0),
            voice_fps=getattr(settings, "tracking_voice_fps", 0.5),
            acquire_window_seconds=getattr(
                settings, "tracking_acquire_window_seconds", 2.0
            ),
            missing_grace_seconds=getattr(
                settings, "tracking_missing_grace_seconds", 2.5
            ),
            pivot_confirmation_seconds=getattr(
                settings, "tracking_pivot_confirmation_seconds", 1.5
            ),
            settling_seconds=getattr(settings, "tracking_settling_seconds", 1.0),
            opposite_pivot_block_seconds=getattr(
                settings, "tracking_opposite_pivot_block_seconds", 5.0
            ),
            lost_neutral_seconds=getattr(
                settings, "tracking_lost_neutral_seconds", 5.0
            ),
            head_state_provider=lambda: (
                control_channel.telemetry if control_channel is not None else {}
            ),
        )
    return tracking_service


def get_vision_service() -> VisionService:
    global vision_service
    coordinator = get_brain_coordinator()
    if vision_service is None or getattr(vision_service, "coordinator", coordinator) is not coordinator:
        vision_service = VisionService(settings, coordinator, get_frame_broker())
    return vision_service


async def inspect_scene(question: str) -> dict:
    result = await get_vision_service().query(question, fresh=True)
    return {
        "fresh": result.fresh,
        "warning": result.warning,
        "snapshot": result.snapshot.model_dump(mode="json"),
    }


def current_person_context() -> dict:
    status = get_tracking_service().status()
    target = status.target
    target_age = (datetime.now(timezone.utc) - target.observed_at).total_seconds() if target is not None else None
    if target is None or status.state != "tracking" or target_age is None or target_age > 1.0:
        return {
            "person_detected": None,
            "state": status.state,
            "mode": status.mode,
            "instruction": "Person presence is unknown; do not claim that nobody is present.",
        }
    center_x = (target.bounding_box[0] + target.bounding_box[2]) / 2.0
    position = "center"
    if center_x < 0.35:
        position = "left"
    elif center_x > 0.65:
        position = "right"
    return {
        "person_detected": True,
        "position": position,
        "confidence": target.confidence,
        "observed_at": target.observed_at.isoformat(),
        "frame_id": target.frame_id,
        "instruction": "A person is detected, but their identity is unknown.",
    }


TRACKING_START_MARKERS = re.compile(
    r"\b("
    r"(?:track|follow)\s+(?:me|the person|that person)|"
    r"(?:start|resume)\s+(?:tracking|following)(?:\s+(?:me|the person|that person))?|"
    r"enable (?:person )?tracking|"
    r"turn (?:person )?tracking on|"
    r"look at me again"
    r")\b",
    re.IGNORECASE,
)
TRACKING_STOP_MARKERS = re.compile(r"\b(stop|quit|end)\s+(tracking|following)\b", re.IGNORECASE)
TRACKING_OFF_MARKERS = re.compile(
    r"\b("
    r"stop (?:looking at|watching|tracking|following) me|"
    r"(?:do not|don't) (?:look at|watch|track|follow) me|"
    r"(?:do not|don't) (?:start|resume) (?:tracking|following)(?: me)?|"
    r"(?:do not|don't) enable (?:person )?tracking|"
    r"(?:do not|don't) turn (?:person )?tracking on|"
    r"turn (?:person )?tracking off|"
    r"disable (?:person )?tracking"
    r")\b",
    re.IGNORECASE,
)


async def execute_explicit_tracking_request(text: str) -> dict | None:
    if TRACKING_OFF_MARKERS.search(text):
        status = await get_tracking_service().stop(reason="explicit request")
        return {"ok": True, "command": "off", "status": status.model_dump(mode="json")}
    if TRACKING_STOP_MARKERS.search(text):
        status = await get_tracking_service().stop(reason="explicit request")
        return {"ok": True, "command": "off", "status": status.model_dump(mode="json")}
    if TRACKING_START_MARKERS.search(text):
        service = get_tracking_service()
        if not service.detector.available and not await service.detector.probe():
            return {"ok": False, "command": "track", "error": service.detector.reason}
        status = await service.enable()
        return {"ok": True, "command": "track", "status": status.model_dump(mode="json")}
    return None


def require_model_eye_expression(action: RobotActionRequest) -> None:
    if action.eyes and action.eyes.expression not in EMOTIONAL_EYE_EXPRESSIONS:
        correlation_id = (
            current_correlation_id.get()
            or get_brain_coordinator().state.active_correlation_id
            or get_brain_coordinator().new_correlation_id()
        )
        get_brain_coordinator().record(
            "eyes.mood.rejected",
            EventSource.policy,
            correlation_id,
            {"expression": action.eyes.expression, "reason": "operational expressions are coordinator-owned"},
        )
        raise ValueError(f"Model cannot select operational eye expression: {action.eyes.expression}")


def validate_action_payload(payload: dict) -> dict:
    action = RobotActionRequest.model_validate(normalize_llm_action_body(payload))
    require_model_eye_expression(action)
    return action.model_dump(exclude_none=True)


def get_realtime_gateway() -> RealtimeGateway:
    global realtime_gateway
    if realtime_gateway is None:
        vision = get_vision_service()

        def current_scene_context() -> dict | None:
            snapshot = vision.current_snapshot()
            context = snapshot.model_dump(mode="json") if snapshot is not None else {}
            context["person_tracking"] = current_person_context()
            return context

        realtime_gateway = RealtimeGateway(
            settings.realtime_ws_url,
            settings.realtime_voice,
            effective_realtime_instructions(),
            get_brain_coordinator(),
            validate_action_payload,
            execute_voice_model_action_payload,
            server_fault_handler=get_eye_controller().set_server_fault,
            voice_session_handler=get_eye_controller().set_voice_session_active,
            inspect_scene=inspect_scene,
            scene_context=current_scene_context,
            tracking_command=execute_explicit_tracking_request,
        )
        vision.subscribe_snapshot(realtime_gateway.refresh_scene_context)
        get_tracking_service().subscribe_state(
            realtime_gateway.refresh_scene_context
        )
    return realtime_gateway


def robot_client() -> httpx.AsyncClient:
    global robot_http_client
    if robot_http_client is None or getattr(robot_http_client, "is_closed", False):
        robot_http_client = httpx.AsyncClient(timeout=settings.request_timeout)
    return robot_http_client


def parse_robot_response(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {"ok": True, "body": response.text}


def cache_robot_status(path: str, payload: dict) -> None:
    global robot_status_cache, robot_status_cache_at
    if path in {"/status", "/api/status"} and payload.get("ok") is True:
        robot_status_cache = payload
        robot_status_cache_at = time.monotonic()


async def robot_request(
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
    authorization: Callable[[], bool] | None = None,
):
    url = f"{settings.robot_base_url}{path}"
    retries = max(0, getattr(settings, "robot_request_retries", 2))
    backoff = max(0.0, getattr(settings, "robot_retry_backoff_seconds", 0.15))
    last_error: httpx.HTTPError | None = None

    async with robot_request_lock:
        for attempt in range(retries + 1):
            if authorization is not None and not authorization():
                return {"ok": False, "skipped": "tracking authorization expired"}
            try:
                correlation_id = current_correlation_id.get()
                headers = {"x-robit-correlation-id": correlation_id} if correlation_id else None
                request_kwargs = {"params": params, "json": body}
                if headers:
                    request_kwargs["headers"] = headers
                response = await robot_client().request(method, url, **request_kwargs)
                response.raise_for_status()
                payload = parse_robot_response(response)
                cache_robot_status(path, payload)
                return payload
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "robot.proxy_retry method=%s path=%s attempt=%s/%s error=%r",
                    method,
                    path,
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if attempt < retries:
                    await asyncio.sleep(backoff * (attempt + 1))

    raise HTTPException(status_code=502, detail=f"Robot request failed: {last_error}") from last_error


async def robot_get(path: str, params: dict | None = None):
    return await robot_request("GET", path, params=params)


async def robot_post(
    path: str,
    body: dict | None = None,
    authorization: Callable[[], bool] | None = None,
):
    if authorization is not None and not authorization():
        return {"ok": False, "skipped": "tracking authorization expired"}
    payload = body or {}
    broker = get_actuator_broker()
    if path == "/api/move":
        direction = str(payload.get("direction", "stop"))
        if direction == "stop":
            return await broker.stop()
        return await broker.drive(
            direction,
            int(payload.get("speed", settings.robot_llm_default_speed)),
            int(payload.get("duration_ms", 0)),
        )
    if path == "/api/head":
        current = broker.status()["desired_head"]
        pan = int(payload.get("pan", current["pan"])) + int(payload.get("pan_delta", 0))
        tilt = int(payload.get("tilt", current["tilt"])) + int(payload.get("tilt_delta", 0))
        return await broker.head_target(pan, tilt, source="manual")
    if path == "/api/eyes":
        return await broker.eyes(
            str(payload.get("expression", "neutral")),
            int(payload.get("duration_ms", 0)),
        )
    raise HTTPException(
        status_code=410,
        detail=f"ESP HTTP control endpoint removed: {path}",
    )


async def robot_heartbeat_post(path: str, body: dict | None = None):
    """Compatibility adapter for the eye controller; TCP owns heartbeats."""
    channel = get_actuator_broker().channel
    channel_status = channel.status()
    if not channel_status["ready"]:
        raise RuntimeError(channel_status["last_error"] or "control channel not ready")
    consume_recovery = getattr(channel, "consume_watchdog_recovery", None)
    recovered = bool(consume_recovery()) if callable(consume_recovery) else bool(
        channel_status.get("watchdog_recovered")
        or channel_status.get("telemetry", {}).get("heartbeat_recovered")
    )
    return {"ok": True, "heartbeat_recovered": recovered}


def cached_robot_status() -> dict | None:
    if robot_status_cache and time.monotonic() - robot_status_cache_at <= ROBOT_STATUS_CACHE_SECONDS:
        return robot_status_cache
    return None


async def robot_fetch_bytes(path: str, base_url: str | None = None):
    url = f"{base_url or settings.robot_base_url}{path}"
    try:
        # Camera acquisition can take about half a second on the ESP. Keep it
        # off the control queue so fresh head commands are not expired while
        # waiting behind the continuous frame broker.
        async with robot_camera_request_lock:
            response = await robot_client().get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        message = str(exc) or type(exc).__name__
        logger.warning(
            "camera.failed stage=acquisition error_class=%s error=%s",
            type(exc).__name__,
            message,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Robot camera request failed ({type(exc).__name__}): {message}",
        ) from exc
    return response.content, response.headers.get("content-type", "image/jpeg")


async def fetch_robot_camera_frame():
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "robit.local"
    scheme = parsed.scheme or "http"
    camera_base_url = urlunparse((scheme, f"{hostname}:81", "", "", "", ""))
    started = time.perf_counter()
    content, media_type = await robot_fetch_bytes("/capture", camera_base_url)
    logger.info(
        "camera.acquired bytes=%s acquisition_ms=%.2f",
        len(content),
        (time.perf_counter() - started) * 1000.0,
    )
    return content, media_type


def camera_urls() -> dict:
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "robit.local"
    scheme = parsed.scheme or "http"
    capture_url = urlunparse((scheme, f"{hostname}:81", "/capture", "", "", ""))
    stream_url = urlunparse((scheme, f"{hostname}:81", "/stream", "", "", ""))
    broker = get_frame_broker()
    configured_interval = getattr(settings, "camera_frame_interval_seconds", 5.0)
    interval = getattr(broker, "effective_interval_seconds", configured_interval)
    fps = getattr(broker, "effective_fps", 1.0 / interval)
    return {
        "ok": True,
        "robot_base_url": settings.robot_base_url,
        "page_url": f"{settings.robot_base_url}/camera",
        "capture_url": capture_url,
        "stream_url": stream_url,
        "frame_interval_seconds": interval,
        "effective_fps": fps,
    }


def sanitized_action_payload(action: RobotActionRequest) -> dict:
    payload: dict = {}
    if action.movement:
        movement = action.movement.model_dump(exclude_none=True)
        maximum_speed = settings.robot_llm_max_speed
        if "speed" in movement:
            movement["speed"] = min(movement["speed"], maximum_speed)
        elif movement["direction"] != "stop":
            movement["speed"] = min(
                max(0, getattr(settings, "robot_llm_default_speed", 170)),
                maximum_speed,
            )
        if "duration_ms" in movement:
            movement["duration_ms"] = min(movement["duration_ms"], settings.robot_llm_max_duration_ms)
        elif movement["direction"] != "stop":
            movement["duration_ms"] = min(300, settings.robot_llm_max_duration_ms)
        payload["movement"] = movement
    if action.head:
        payload["head"] = action.head.model_dump(exclude_none=True)
    if action.eyes:
        payload["eyes"] = action.eyes.model_dump(exclude_none=True)
    return payload


def normalize_llm_action_body(action_body: dict) -> dict:
    normalized = dict(action_body)
    movement = normalized.get("movement")
    if isinstance(movement, dict):
        movement = dict(movement)
        if "speed" in movement and isinstance(movement["speed"], float):
            if 0 <= movement["speed"] <= 1:
                movement["speed"] = round(movement["speed"] * settings.robot_llm_max_speed)
            else:
                movement["speed"] = round(movement["speed"])
        if "duration_ms" in movement and isinstance(movement["duration_ms"], float):
            movement["duration_ms"] = round(movement["duration_ms"])
        normalized["movement"] = movement

    head = normalized.get("head")
    if isinstance(head, dict):
        head = dict(head)
        for key in ("pan", "tilt", "pan_delta", "tilt_delta"):
            if key in head and isinstance(head[key], float):
                head[key] = round(head[key])
        normalized["head"] = head

    eyes = normalized.get("eyes")
    if isinstance(eyes, dict):
        eyes = dict(eyes)
        if "duration_ms" in eyes and isinstance(eyes["duration_ms"], float):
            eyes["duration_ms"] = round(eyes["duration_ms"])
        normalized["eyes"] = eyes

    return normalized


async def execute_robot_action(action: RobotActionRequest, mood_source: EventSource | None = None) -> dict:
    payload = sanitized_action_payload(action)
    executed: list[dict] = []
    skipped: list[dict] = []

    if payload.get("movement") or payload.get("head"):
        get_tracking_service().suspend_for_manual_control()

    model_eyes = payload.get("eyes") if mood_source is not None else None
    if model_eyes:
        correlation_id = current_correlation_id.get() or get_brain_coordinator().new_correlation_id()
        state = get_eye_controller().select_mood(
            model_eyes["expression"],
            model_eyes.get("duration_ms"),
            mood_source,
            correlation_id,
        )
        executed.append({"type": "eyes.mood", "request": model_eyes, "queued": True, "state": state.model_dump(mode="json")})

    if movement := payload.get("movement"):
        result = await robot_post("/api/move", movement)
        executed.append({"type": "movement", "request": movement, "result": result})

    if head := payload.get("head"):
        result = await robot_post("/api/head", head)
        get_tracking_service().sync_head_position(
            result.get("pan_target", result.get("pan")),
            result.get("tilt_target", result.get("tilt")),
        )
        executed.append({"type": "head", "request": head, "result": result})

    if mood_source is None and (eyes := payload.get("eyes")):
        try:
            result = await robot_post("/api/eyes", eyes)
            executed.append({"type": "eyes", "request": eyes, "result": result})
        except HTTPException as exc:
            skipped.append({"type": "eyes", "request": eyes, "reason": exc.detail})

    logger.info("robot.llm_action %s", json.dumps(payload))
    successful = all(
        not isinstance(item.get("result"), dict)
        or item["result"].get("ok") is not False
        for item in executed
    )
    return {
        "ok": successful,
        "action": payload,
        "executed": executed,
        "skipped": skipped,
    }


async def execute_action_payload(payload: dict) -> dict:
    action = RobotActionRequest.model_validate(payload)
    return await execute_robot_action(action)


async def execute_voice_model_action_payload(payload: dict) -> dict:
    action = RobotActionRequest.model_validate(payload)
    require_model_eye_expression(action)
    return await execute_robot_action(action, EventSource.voice_model)


async def execute_text_model_action_payload(payload: dict) -> dict:
    action = RobotActionRequest.model_validate(payload)
    require_model_eye_expression(action)
    return await execute_robot_action(action, EventSource.text_model)


def correlation_from_request(request: Request) -> str:
    return request.headers.get("x-correlation-id") or get_brain_coordinator().new_correlation_id()


async def coordinated_action(
    action: RobotActionRequest,
    origin: EventSource,
    correlation_id: str,
    reason: str,
    priority: WorkPriority,
    executor=execute_action_payload,
) -> dict:
    payload = action.model_dump(exclude_none=True)
    intent = ActionIntent(
        action=payload,
        origin=origin,
        correlation_id=correlation_id,
        reason=reason,
        priority=priority,
    )
    result = await get_brain_coordinator().execute_action(intent, executor)
    return {**result, "correlation_id": correlation_id}


def manual_action_response(result: dict) -> dict:
    executed = result.get("executed") or []
    if executed and isinstance(executed[0].get("result"), dict):
        return {**executed[0]["result"], "correlation_id": result["correlation_id"]}
    return result


async def execute_manual_drive(payload: dict) -> dict:
    return await robot_post("/api/move", payload["movement"])


async def execute_manual_head(payload: dict) -> dict:
    result = await robot_post("/api/head", payload["head"])
    get_tracking_service().sync_head_position(
        result.get("pan_target", result.get("pan")),
        result.get("tilt_target", result.get("tilt")),
    )
    return result


async def execute_manual_stop(payload: dict) -> dict:
    return await robot_post("/api/move", {"direction": "stop"})


def prompt_with_live_scene(text: str) -> str:
    service = vision_service
    snapshot = service.current_snapshot() if service is not None and hasattr(service, "current_snapshot") else None
    if snapshot is None:
        context = "LIVE VISUAL CONTEXT: unavailable or expired. Do not claim to currently see specific objects."
    else:
        context = "LIVE VISUAL CONTEXT: " + json.dumps(
            {
                "frame_id": snapshot.frame_id,
                "observed_at": snapshot.observed_at.isoformat(),
                "summary": snapshot.summary,
                "entities": [
                    {"label": entity.label, "confidence": entity.confidence}
                    for entity in snapshot.entities
                ],
                "uncertainty": snapshot.uncertainty,
            },
            separators=(",", ":"),
        )
    person_context = json.dumps(current_person_context(), separators=(",", ":"))
    return (
        f"{context}\nLIVE PERSON-TRACKING CONTEXT: {person_context}\n"
        "Use person tracking only for presence and rough left/center/right position; never infer identity. "
        "Use this validated scene as Robit's current visual awareness when relevant, "
        "without inventing additional details.\nUser request: " + text
    )


async def call_llm(
    method_name: str,
    text: str,
    history: list[dict[str, str]],
    include_live_scene: bool = True,
):
    method = getattr(llm_client, method_name)
    prompt = prompt_with_live_scene(text) if include_live_scene else text
    started_at = time.monotonic()
    try:
        async with get_brain_coordinator().resource_lease.acquire(WorkPriority.foreground):
            if "history" in inspect.signature(method).parameters:
                result = await method(prompt, history=history)
            else:
                result = await method(prompt)
    except Exception as exc:
        llm_health.record_inference(
            success=False,
            checked_at=datetime.now(timezone.utc).isoformat(),
            latency_ms=(time.monotonic() - started_at) * 1000.0,
            error=f"inference failed ({type(exc).__name__})",
        )
        raise
    llm_health.record_inference(
        success=True,
        checked_at=datetime.now(timezone.utc).isoformat(),
        latency_ms=(time.monotonic() - started_at) * 1000.0,
    )
    return result


def parse_action_response(content: str) -> dict | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


VISUAL_QUESTION_MARKERS = re.compile(
    r"\b(what do you see|what can you see|can you see|look at|camera|in front of you|around you|visible)\b",
    re.IGNORECASE,
)


def explicit_visual_question(text: str) -> str | None:
    stripped = text.strip()
    return stripped if stripped and VISUAL_QUESTION_MARKERS.search(stripped) else None


async def answer_visual_question(
    question: str,
    history: list[dict[str, str]],
    correlation_id: str,
) -> tuple[str, str, dict]:
    result = await get_vision_service().query(question, fresh=True)
    snapshot = result.snapshot.model_dump(mode="json")
    freshness_instruction = (
        "If it is cached, use past-tense wording such as 'my last image showed' and say you cannot confirm what is there now. "
        if not result.fresh
        else ""
    )
    grounded_prompt = (
        "Answer the user's visual question using only this validated SceneSnapshot. "
        "Be concise, say when uncertainty is high, and do not propose or claim any physical action. "
        f"The observation is {'fresh' if result.fresh else 'cached and not current'}. "
        f"{freshness_instruction}"
        f"User question: {question}\nSceneSnapshot: {json.dumps(snapshot, separators=(',', ':'))}"
    )
    response = await call_llm("chat", grounded_prompt, history, include_live_scene=False)
    get_brain_coordinator().record(
        "perception.query.answered",
        EventSource.text_model,
        correlation_id,
        {"frame_id": result.snapshot.frame_id, "fresh": result.fresh, "warning": result.warning},
    )
    return response.response, response.model, {
        "fresh": result.fresh,
        "warning": result.warning,
        "snapshot": snapshot,
    }


def voice_health_snapshot() -> dict:
    sox_available = shutil.which("sox") is not None
    microphone_enhancement_available = importlib.util.find_spec("noisereduce") is not None
    degraded_reasons = [
        reason
        for available, reason in (
            (sox_available, "SoX is unavailable"),
            (
                microphone_enhancement_available,
                "microphone enhancement is unavailable",
            ),
        )
        if not available
    ]
    return {
        "status": "ok" if not degraded_reasons else "degraded",
        "sox_available": sox_available,
        "microphone_enhancement_available": microphone_enhancement_available,
        "degraded_reasons": degraded_reasons,
    }


def public_base_url(value: str) -> str:
    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def tracking_telemetry_payload(tracking, broker) -> dict:
    latest_frame = broker.latest
    latest_frame_age_seconds = None
    if latest_frame is not None:
        latest_frame_age_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - latest_frame.captured_at).total_seconds(),
        )
    return {
        "available": bool(tracking.available),
        "active": bool(tracking.enabled),
        "latest_frame_age_seconds": latest_frame_age_seconds,
        "latest_result_age_seconds": getattr(tracking, "target_age_seconds", None),
        "error": tracking.reason,
    }


def readiness_components(robot: dict, tracking, broker, voice_health: dict) -> dict:
    checked_at = datetime.now(timezone.utc).isoformat()
    robot_status_value = str(robot.get("status") or "offline")
    robot_control = robot.get("control") or {}
    if robot_status_value == "online":
        robot_reason = None
    elif robot_status_value == "stale":
        robot_reason = "robot telemetry is stale"
    elif robot_status_value == "fault":
        robot_reason = get_brain_coordinator().critical_fault_reason or "robot fault is active"
    else:
        robot_reason = robot_control.get("error") or "robot telemetry is unavailable"

    llm = llm_health.snapshot()
    llm_status = llm["status"]
    if llm_status == "ready":
        llm_reason = None
    elif llm_status == "degraded":
        llm_reason = llm["last_inference_error"] or "latest inference failed"
    elif llm_status == "unavailable":
        llm_reason = llm["last_probe_error"] or "latest LLM probe failed"
    else:
        llm_reason = "LLM has not been probed"

    camera_status = broker.status()
    camera_age = camera_status.get("frame_age_seconds")
    camera_error = camera_status.get("last_error")
    if camera_error:
        camera_component_status = "degraded"
        camera_reason = camera_error
        camera_checked_at = camera_status.get("last_error_at")
    elif camera_age is None:
        camera_component_status = "degraded"
        camera_reason = "no camera frame is available"
        camera_checked_at = camera_status.get("last_success_at") or checked_at
    elif camera_age > CAMERA_READY_MAX_AGE_SECONDS:
        camera_component_status = "stale"
        camera_reason = "camera frame is older than 5 seconds"
        camera_checked_at = camera_status.get("last_success_at")
    else:
        camera_component_status = "ready"
        camera_reason = None
        camera_checked_at = camera_status.get("last_success_at")
    tracking_ready = bool(tracking.available)
    voice_ready = voice_health["status"] == "ok"
    return {
        "robot": {
            "status": robot_status_value,
            "reason": robot_reason,
            "checked_at": checked_at,
            "latency_ms": robot_control.get("round_trip_ms"),
        },
        "llm": {
            "status": llm_status,
            "reason": llm_reason,
            "checked_at": (
                llm["last_inference_at"]
                if llm_status == "degraded"
                else llm["last_probe_at"]
            ),
            "latency_ms": (
                llm["last_inference_latency_ms"]
                if llm_status == "degraded"
                else llm["last_probe_latency_ms"]
            ),
        },
        "tracking": {
            "status": "ready" if tracking_ready else "degraded",
            "reason": None if tracking_ready else (tracking.reason or "tracking is unavailable"),
            "checked_at": checked_at,
            "latency_ms": tracking.detector_latency_ms,
        },
        "camera": {
            "status": camera_component_status,
            "reason": camera_reason,
            "checked_at": camera_checked_at,
            "latency_ms": camera_status.get("last_acquisition_ms"),
        },
        "voice": {
            "status": "ready" if voice_ready else "degraded",
            "reason": (
                None
                if voice_ready
                else "; ".join(voice_health["degraded_reasons"])
            ),
            "checked_at": checked_at,
            "latency_ms": None,
        },
    }


@app.get("/health")
async def health(request: Request):
    websocket_scheme = "wss" if request.url.scheme == "https" else "ws"
    gateway_url = f"{websocket_scheme}://{request.url.netloc}/v1/realtime"
    tracking = get_tracking_service().status()
    broker = get_frame_broker()
    robot = await robot_status()
    voice_health = voice_health_snapshot()
    components = readiness_components(robot, tracking, broker, voice_health)
    return {
        "ok": True,
        "ready": (
            components["robot"]["status"] == "online"
            and components["llm"]["status"] == "ready"
        ),
        "components": components,
        "robot_base_url": public_base_url(settings.robot_base_url),
        "llm_provider": settings.llm_provider,
        "llm_base_url": public_base_url(settings.llm_base_url),
        "llm_model": settings.llm_model,
        "realtime": {
            "ws_url": gateway_url,
            "gateway_path": "/v1/realtime",
            "voice": settings.realtime_voice,
            "instructions": effective_realtime_instructions(),
        },
        "tracking": {
            "available": tracking.available,
            "mode": tracking.mode,
            "backend": tracking.backend,
            "reason": tracking.reason,
            "detector_latency_ms": tracking.detector_latency_ms,
            "detector_queue_ms": tracking.detector_queue_ms,
            "detector_cadence_fps": tracking.detector_cadence_fps,
        },
        "control_channel": get_actuator_broker().channel.status(),
        "actuators": get_actuator_broker().status(),
        "journal": brain_journal.status() if brain_journal is not None else None,
        "camera": broker.status(),
        "workload": get_brain_coordinator().resource_lease.status(),
        "event_loop": {
            "lag_ms": event_loop_lag_ms,
            "peak_lag_ms": event_loop_lag_peak_ms,
        },
        "voice_health": voice_health,
    }


@app.get("/system/telemetry")
async def system_telemetry():
    sampled = await telemetry_service.latest_host()
    tracking = get_tracking_service().status()
    broker = get_frame_broker()
    return {
        **sampled,
        "llm": llm_health.snapshot(),
        "robot": await robot_status(),
        "tracking": tracking_telemetry_payload(tracking, broker),
        "faults": [
            fault.model_dump(mode="json")
            for fault in get_brain_coordinator().active_faults
        ],
    }


@app.get("/", include_in_schema=False)
async def web_control():
    if not WEB_CONTROL_INDEX.exists():
        raise HTTPException(status_code=404, detail="web_control/index.html was not found")
    return FileResponse(WEB_CONTROL_INDEX, headers={"Cache-Control": "no-store"})


@app.get("/robot/status")
async def robot_status(refresh: bool = False):
    broker = get_actuator_broker()
    channel = broker.channel
    channel_status = channel.status()
    broker_status = broker.status()
    now = time.monotonic()

    if refresh:
        sample = await robot_get("/status")
        source = "http"
        sample_age_ms = 0.0
        connected = sample.get("ok") is True
        ready = connected
    elif channel.telemetry:
        sample = channel.telemetry
        source = "tcp"
        received_at = getattr(channel, "telemetry_received_at", None)
        sample_age_ms = (
            max(0.0, (now - received_at) * 1000.0)
            if received_at is not None
            else (0.0 if channel.stats.ready else None)
        )
        connected = bool(channel.stats.connected)
        ready = bool(channel.stats.ready)
    elif (cached := cached_robot_status()) is not None:
        sample = cached
        source = "cache"
        sample_age_ms = max(0.0, (now - robot_status_cache_at) * 1000.0)
        connected = False
        ready = False
    else:
        sample = None
        source = "none"
        sample_age_ms = None
        connected = bool(channel.stats.connected)
        ready = bool(channel.stats.ready)

    coordinator = get_brain_coordinator()
    return normalize_robot_status(
        sample,
        source=source,
        sample_age_ms=sample_age_ms,
        connected=connected,
        ready=ready,
        desired_head=broker_status.get("desired_head"),
        desired_eyes=broker_status.get("desired_eyes"),
        override_reason=coordinator.critical_fault_reason,
        control=channel_status,
        critical_fault=coordinator.has_critical_fault,
    )


@app.get("/robot/camera")
async def robot_camera():
    return camera_urls()


@app.get("/robot/camera/capture")
async def robot_camera_capture(fresh: bool = False):
    frame = await get_frame_broker().get_frame(force_fresh=fresh)
    return Response(
        content=frame.content,
        media_type=frame.media_type,
        headers={
            "X-Robit-Frame-Id": frame.frame_id,
            "X-Robit-Captured-At": frame.captured_at.isoformat(),
            "X-Robit-Frame-Interval": str(
                getattr(
                    get_frame_broker(),
                    "effective_interval_seconds",
                    getattr(settings, "camera_frame_interval_seconds", 5.0),
                )
            ),
            "Cache-Control": "no-store",
        },
    )


@app.get("/perception/latest")
async def perception_latest():
    return get_vision_service().latest_payload()


@app.post("/perception/query")
async def perception_query(query: PerceptionQueryRequest):
    try:
        result = await get_vision_service().query(query.question, query.fresh)
    except VisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "fresh": result.fresh,
        "warning": result.warning,
        "snapshot": result.snapshot.model_dump(mode="json"),
        "world_state": get_vision_service().world_state().model_dump(mode="json"),
    }


@app.get("/tracking/status")
async def tracking_status():
    return get_tracking_service().status().model_dump(mode="json")


@app.post("/tracking/start")
async def tracking_start():
    service = get_tracking_service()
    if not service.detector.available and not await service.detector.probe():
        raise HTTPException(status_code=503, detail=service.detector.reason or "RF-DETR is unavailable")
    try:
        status = await service.enable()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return status.model_dump(mode="json")


@app.post("/tracking/stop")
async def tracking_stop():
    status = await get_tracking_service().stop(reason="API request")
    return status.model_dump(mode="json")


@app.post("/robot/drive")
async def robot_drive(command: DriveCommand, request: Request):
    get_tracking_service().suspend_for_manual_control()
    movement = MovementAction(direction=command.move, speed=command.speed)
    result = await coordinated_action(
        RobotActionRequest(movement=movement),
        EventSource.manual,
        correlation_from_request(request),
        "Manual drive control",
        WorkPriority.manual_action,
        execute_manual_drive,
    )
    return manual_action_response(result)


@app.post("/robot/head")
async def robot_head(command: HeadCommand, request: Request):
    if command.pan is None and command.tilt is None:
        raise HTTPException(status_code=400, detail="pan or tilt is required")
    get_tracking_service().suspend_for_manual_control()
    result = await coordinated_action(
        RobotActionRequest(head=HeadAction(pan=command.pan, tilt=command.tilt)),
        EventSource.manual,
        correlation_from_request(request),
        "Manual head control",
        WorkPriority.manual_action,
        execute_manual_head,
    )
    return manual_action_response(result)


@app.post("/robot/stop")
async def robot_stop(request: Request):
    get_tracking_service().suspend_for_manual_control()
    result = await coordinated_action(
        RobotActionRequest(movement=MovementAction(direction="stop")),
        EventSource.manual,
        correlation_from_request(request),
        "Manual stop",
        WorkPriority.manual_action,
        execute_manual_stop,
    )
    return manual_action_response(result)


@app.post("/robot/action")
async def robot_action(action: RobotActionRequest, request: Request):
    return await coordinated_action(
        action,
        EventSource.api,
        correlation_from_request(request),
        "Validated API action",
        WorkPriority.model_action,
    )


@app.post("/chat")
async def chat(chat_request: ChatRequest, request: Request):
    coordinator = get_brain_coordinator()
    correlation_id = correlation_from_request(request)
    history = coordinator.recent_messages(20)
    coordinator.record_turn("user", chat_request.text, EventSource.browser, correlation_id)
    coordinator.transition(correlation_id, EventSource.browser, conversation=ConversationState.formulating)
    with timed("endpoint.chat", prompt_chars=len(chat_request.text)):
        result = await call_llm("chat", chat_request.text, history)
    coordinator.record_turn("assistant", result.response, EventSource.text_model, correlation_id)
    coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
    return {
        "response": result.response,
        "model": result.model,
        "conversation_id": chat_request.conversation_id,
        "correlation_id": correlation_id,
    }


@app.post("/chat/action")
async def chat_action(chat_request: ChatActionRequest, request: Request):
    coordinator = get_brain_coordinator()
    correlation_id = correlation_from_request(request)
    history = coordinator.recent_messages(20)
    coordinator.record_turn("user", chat_request.text, EventSource.browser, correlation_id)
    coordinator.transition(correlation_id, EventSource.browser, conversation=ConversationState.formulating)
    explicit_action = explicit_robot_action(chat_request.text)
    if explicit_action and explicit_action.get("movement", {}).get("direction") == "stop":
        action_result = await coordinated_action(
            RobotActionRequest(movement=MovementAction(direction="stop")),
            EventSource.browser,
            correlation_id,
            "Explicit text stop",
            WorkPriority.manual_action,
            execute_text_model_action_payload,
        )
        response_text = "I stopped Robit's movement."
        coordinator.record_turn("assistant", response_text, EventSource.system, correlation_id)
        coordinator.transition(correlation_id, EventSource.system, conversation=ConversationState.idle)
        return {
            "response": response_text,
            "model": "deterministic-control",
            "conversation_id": chat_request.conversation_id,
            "correlation_id": correlation_id,
            "action": {"movement": {"direction": "stop"}},
            "action_result": action_result,
            "vision": None,
            "parse_error": None,
        }
    tracking_result = await execute_explicit_tracking_request(chat_request.text)
    if tracking_result is not None:
        if tracking_result.get("ok"):
            responses = {
                "track": "I am tracking the visible person continuously.",
                "off": "Person tracking is off.",
            }
            response_text = responses.get(str(tracking_result.get("command")), "Tracking updated.")
        else:
            response_text = f"I could not do that safely: {tracking_result.get('error', 'tracking is unavailable')}."
        coordinator.record_turn("assistant", response_text, EventSource.system, correlation_id)
        coordinator.transition(correlation_id, EventSource.system, conversation=ConversationState.idle)
        return {
            "response": response_text,
            "model": "deterministic-tracking",
            "conversation_id": chat_request.conversation_id,
            "correlation_id": correlation_id,
            "action": None,
            "action_result": tracking_result,
            "vision": None,
            "parse_error": None,
        }
    with timed("endpoint.chat_action", prompt_chars=len(chat_request.text)):
        chat_result = await call_llm("action_chat", chat_request.text, history)

    parsed = parse_action_response(chat_result.response)
    if parsed is None:
        llm_health.mark_last_inference_malformed("malformed action response")
    response_text = str(parsed.get("response") or "").strip() if parsed else ""
    action_body = parsed.get("action") if parsed else None
    action = None
    if action_body is not None:
        try:
            if not isinstance(action_body, dict):
                raise ValueError("action must be an object or null")
            action = RobotActionRequest.model_validate(normalize_llm_action_body(action_body))
            require_model_eye_expression(action)
        except (ValidationError, ValueError) as exc:
            llm_health.mark_last_inference_malformed("malformed action response")
            coordinator.record_turn("assistant", response_text, EventSource.text_model, correlation_id)
            coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
            return {
                "response": response_text,
                "model": chat_result.model,
                "conversation_id": chat_request.conversation_id,
                "correlation_id": correlation_id,
                "action": action_body if isinstance(action_body, dict) else None,
                "action_result": None,
                "vision": None,
                "parse_error": f"LLM returned an invalid robot action; no robot action was executed: {exc}",
            }
    vision_question = explicit_visual_question(chat_request.text)
    if parsed is not None and isinstance(parsed.get("vision_question"), str):
        vision_question = parsed["vision_question"].strip() or vision_question
    if vision_question:
        try:
            response_text, model, vision = await answer_visual_question(vision_question, history, correlation_id)
        except VisionUnavailable as exc:
            response_text = f"I cannot inspect the camera right now: {exc}"
            model = chat_result.model
            vision = {"fresh": False, "warning": str(exc), "snapshot": None}
        coordinator.record_turn("assistant", response_text, EventSource.text_model, correlation_id)
        coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
        return {
            "response": response_text,
            "model": model,
            "conversation_id": chat_request.conversation_id,
            "correlation_id": correlation_id,
            "action": None,
            "action_result": None,
            "vision": vision,
            "parse_error": None,
        }
    if parsed is None:
        coordinator.record_turn("assistant", chat_result.response, EventSource.text_model, correlation_id)
        coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
        return {
            "response": chat_result.response,
            "model": chat_result.model,
            "conversation_id": chat_request.conversation_id,
            "correlation_id": correlation_id,
            "action": None,
            "action_result": None,
            "vision": None,
            "parse_error": "LLM did not return strict JSON; no robot action was executed.",
        }

    action_result = None
    if action is not None:
        action_result = await coordinated_action(
            action,
            EventSource.text_model,
            correlation_id,
            "Text model action",
            WorkPriority.model_action,
            execute_text_model_action_payload,
        )

    coordinator.record_turn("assistant", response_text, EventSource.text_model, correlation_id)
    coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)

    return {
        "response": response_text,
        "model": chat_result.model,
        "conversation_id": chat_request.conversation_id,
        "correlation_id": correlation_id,
        "action": action_body if isinstance(action_body, dict) else None,
        "action_result": action_result,
        "vision": None,
        "parse_error": None,
    }


@app.get("/brain/state")
async def brain_state():
    return get_brain_coordinator().snapshot()


@app.get("/brain/events")
async def brain_events(
    conversation_id: str = "default",
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    correlation_id: str | None = None,
):
    events = get_brain_coordinator().journal.list_events(
        conversation_id,
        after_sequence,
        limit,
        correlation_id,
    )
    return {"events": [event.model_dump(mode="json") for event in events]}


@app.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    await get_realtime_gateway().handle_browser(websocket)


@app.get("/tools")
async def tools():
    return {
        "tools": [
            {
                "name": "drive",
                "description": "Move Robit briefly in one direction. Prefer short durations and stop after movement.",
                "endpoint": "POST /robot/drive",
            },
            {
                "name": "set_head",
                "description": "Set Robit's pan/tilt head within safe angle limits.",
                "endpoint": "POST /robot/head",
            },
            {
                "name": "stop",
                "description": "Immediately stop Robit's tracks.",
                "endpoint": "POST /robot/stop",
            },
            {
                "name": "action",
                "description": "Execute a combined bounded robot action from the PC safety layer.",
                "endpoint": "POST /robot/action",
            },
            {
                "name": "person_tracking",
                "description": "Enable or disable continuous RF-DETR person tracking.",
                "endpoint": "POST /tracking/start",
            },
        ]
    }

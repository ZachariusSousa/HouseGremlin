import asyncio
import inspect
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .audio_utils import ensure_data_dirs
from .brain_models import ActionIntent, ConversationState, EventSource, EyeExpression, WorkPriority
from .config import settings
from .coordinator import BrainCoordinator
from .correlation import current_correlation_id
from .eye_controller import EMOTIONAL_EYE_EXPRESSIONS, EyeController
from .frame_broker import FrameBroker
from .journal import EventJournal
from .llm import OpenAICompatibleChatClient
from .realtime_gateway import RealtimeGateway
from .timing import timed
from .vision import VisionService, VisionUnavailable


logger = logging.getLogger("uvicorn.error")
llm_client = OpenAICompatibleChatClient(settings)
robot_request_lock = asyncio.Lock()
robot_http_client: httpx.AsyncClient | None = None
robot_status_cache: dict | None = None
robot_status_cache_at = 0.0
ROBOT_STATUS_CACHE_SECONDS = 3.0
brain_journal: EventJournal | None = None
brain_coordinator: BrainCoordinator | None = None
realtime_gateway: RealtimeGateway | None = None
eye_controller: EyeController | None = None
frame_broker: FrameBroker | None = None
vision_service: VisionService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot_http_client
    ensure_data_dirs(settings.data_dir)
    get_brain_coordinator()
    robot_http_client = httpx.AsyncClient(timeout=settings.request_timeout)
    controller = get_eye_controller()
    broker = get_frame_broker()
    vision = get_vision_service()
    await controller.start()
    await broker.start()
    await vision.start()
    if settings.warm_models:
        await llm_client.warmup()
    try:
        yield
    finally:
        if realtime_gateway is not None:
            await realtime_gateway.shutdown()
        await vision.shutdown()
        await broker.shutdown()
        await controller.shutdown()
        await robot_http_client.aclose()
        robot_http_client = None


app = FastAPI(title="Robit PC Brain", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
    emergency_stop: bool = False

    @model_validator(mode="after")
    def require_action(self):
        if not any((self.movement, self.head, self.eyes, self.emergency_stop)):
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
        brain_journal = EventJournal(settings.data_dir / "brain.db")
        brain_coordinator = BrainCoordinator(brain_journal)
    return brain_coordinator


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
        frame_broker = FrameBroker(fetch_robot_camera_frame, settings.camera_frame_interval_seconds)
    return frame_broker


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
            return snapshot.model_dump(mode="json") if snapshot is not None else None

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
        )
        vision.subscribe_snapshot(realtime_gateway.refresh_scene_context)
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
    if path in {"/status", "/api/move", "/api/head", "/api/emergency-stop"} and payload.get("ok") is True:
        robot_status_cache = payload
        robot_status_cache_at = time.monotonic()


async def robot_request(method: str, path: str, params: dict | None = None, body: dict | None = None):
    url = f"{settings.robot_base_url}{path}"
    retries = max(0, getattr(settings, "robot_request_retries", 2))
    backoff = max(0.0, getattr(settings, "robot_retry_backoff_seconds", 0.15))
    last_error: httpx.HTTPError | None = None

    async with robot_request_lock:
        for attempt in range(retries + 1):
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


async def robot_post(path: str, body: dict | None = None):
    return await robot_request("POST", path, body=body or {})


async def robot_heartbeat_post(path: str, body: dict | None = None):
    """Heartbeat traffic shares the ESP control queue and never retries."""
    url = f"{settings.robot_base_url}{path}"
    async with robot_request_lock:
        response = await robot_client().post(url, json=body or {}, timeout=settings.request_timeout)
        response.raise_for_status()
        return parse_robot_response(response)


async def robot_emergency_stop_request() -> dict:
    """Emergency stop bypasses the normal serialized robot request queue."""
    url = f"{settings.robot_base_url}/api/emergency-stop"
    correlation_id = current_correlation_id.get()
    headers = {"x-robit-correlation-id": correlation_id} if correlation_id else None
    try:
        response = await robot_client().post(url, json={}, headers=headers)
        response.raise_for_status()
        payload = parse_robot_response(response)
        cache_robot_status("/api/emergency-stop", payload)
        return payload
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Robot emergency stop failed: {exc}") from exc


def cached_robot_status() -> dict | None:
    if robot_status_cache and time.monotonic() - robot_status_cache_at <= ROBOT_STATUS_CACHE_SECONDS:
        return robot_status_cache
    return None


async def robot_fetch_bytes(path: str, base_url: str | None = None):
    url = f"{base_url or settings.robot_base_url}{path}"
    try:
        async with robot_request_lock:
            response = await robot_client().get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        message = str(exc) or type(exc).__name__
        raise HTTPException(status_code=502, detail=f"Robot camera request failed: {message}") from exc
    return response.content, response.headers.get("content-type", "image/jpeg")


async def fetch_robot_camera_frame():
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "robit.local"
    scheme = parsed.scheme or "http"
    camera_base_url = urlunparse((scheme, f"{hostname}:81", "", "", "", ""))
    return await robot_fetch_bytes("/capture", camera_base_url)


def camera_urls() -> dict:
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "robit.local"
    scheme = parsed.scheme or "http"
    capture_url = urlunparse((scheme, f"{hostname}:81", "/capture", "", "", ""))
    stream_url = urlunparse((scheme, f"{hostname}:81", "/stream", "", "", ""))
    return {
        "ok": True,
        "robot_base_url": settings.robot_base_url,
        "page_url": f"{settings.robot_base_url}/camera",
        "capture_url": capture_url,
        "stream_url": stream_url,
        "frame_interval_seconds": settings.camera_frame_interval_seconds,
    }


def sanitized_action_payload(action: RobotActionRequest) -> dict:
    payload: dict = {}
    if action.movement:
        movement = action.movement.model_dump(exclude_none=True)
        if "speed" in movement:
            movement["speed"] = min(movement["speed"], settings.robot_llm_max_speed)
        if "duration_ms" in movement:
            movement["duration_ms"] = min(movement["duration_ms"], settings.robot_llm_max_duration_ms)
        elif movement["direction"] != "stop":
            movement["duration_ms"] = min(300, settings.robot_llm_max_duration_ms)
        payload["movement"] = movement
    if action.head:
        payload["head"] = action.head.model_dump(exclude_none=True)
    if action.eyes:
        payload["eyes"] = action.eyes.model_dump(exclude_none=True)
    if action.emergency_stop:
        payload["emergency_stop"] = True
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

    if payload.get("emergency_stop"):
        result = await robot_emergency_stop_request()
        executed.append({"type": "emergency_stop", "result": result})
        logger.info("robot.llm_action emergency_stop")
        return {"ok": True, "action": payload, "executed": executed, "skipped": skipped}

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
        executed.append({"type": "head", "request": head, "result": result})

    if mood_source is None and (eyes := payload.get("eyes")):
        try:
            result = await robot_post("/api/eyes", eyes)
            executed.append({"type": "eyes", "request": eyes, "result": result})
        except HTTPException as exc:
            skipped.append({"type": "eyes", "request": eyes, "reason": exc.detail})

    logger.info("robot.llm_action %s", json.dumps(payload))
    return {"ok": True, "action": payload, "executed": executed, "skipped": skipped}


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
        priority=WorkPriority.emergency if action.emergency_stop else priority,
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
    return await robot_post("/api/head", payload["head"])


async def execute_manual_stop(payload: dict) -> dict:
    try:
        return await robot_emergency_stop_request()
    except HTTPException:
        return await robot_get("/cmd", {"move": "stop"})


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
    return (
        f"{context}\nUse this validated scene as Robit's current visual awareness when relevant, "
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
    async with get_brain_coordinator().resource_lease.acquire(WorkPriority.foreground):
        if "history" in inspect.signature(method).parameters:
            return await method(prompt, history=history)
        return await method(prompt)


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


@app.get("/health")
async def health(request: Request):
    websocket_scheme = "wss" if request.url.scheme == "https" else "ws"
    gateway_url = f"{websocket_scheme}://{request.url.netloc}/v1/realtime"
    return {
        "ok": True,
        "robot_base_url": settings.robot_base_url,
        "llm_provider": settings.llm_provider,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "realtime": {
            "ws_url": gateway_url,
            "gateway_path": "/v1/realtime",
            "voice": settings.realtime_voice,
            "instructions": effective_realtime_instructions(),
        },
    }


@app.get("/", include_in_schema=False)
async def web_control():
    if not WEB_CONTROL_INDEX.exists():
        raise HTTPException(status_code=404, detail="web_control/index.html was not found")
    return FileResponse(WEB_CONTROL_INDEX, headers={"Cache-Control": "no-store"})


@app.get("/robot/status")
async def robot_status():
    if cached := cached_robot_status():
        return cached
    if robot_request_lock.locked() and robot_status_cache:
        return robot_status_cache
    return await robot_get("/status")


@app.get("/robot/camera")
async def robot_camera():
    return camera_urls()


@app.get("/robot/camera/capture")
async def robot_camera_capture():
    frame = await get_frame_broker().get_frame()
    return Response(
        content=frame.content,
        media_type=frame.media_type,
        headers={
            "X-Robit-Frame-Id": frame.frame_id,
            "X-Robit-Captured-At": frame.captured_at.isoformat(),
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


@app.post("/robot/drive")
async def robot_drive(command: DriveCommand, request: Request):
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
    result = await coordinated_action(
        RobotActionRequest(emergency_stop=True),
        EventSource.manual,
        correlation_from_request(request),
        "Manual emergency stop",
        WorkPriority.emergency,
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
    with timed("endpoint.chat_action", prompt_chars=len(chat_request.text)):
        chat_result = await call_llm("action_chat", chat_request.text, history)

    parsed = parse_action_response(chat_result.response)
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

    response_text = str(parsed.get("response") or "").strip()
    action_body = parsed.get("action")
    action_result = None
    if isinstance(action_body, dict) and action_body:
        try:
            action = RobotActionRequest.model_validate(normalize_llm_action_body(action_body))
            require_model_eye_expression(action)
        except (ValidationError, ValueError) as exc:
            coordinator.record_turn("assistant", response_text, EventSource.text_model, correlation_id)
            coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
            return {
                "response": response_text or parsed.get("response") or "",
                "model": chat_result.model,
                "conversation_id": chat_request.conversation_id,
                "correlation_id": correlation_id,
                "action": action_body,
                "action_result": None,
                "vision": None,
                "parse_error": f"LLM returned an invalid robot action; no robot action was executed: {exc}",
            }
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
        ]
    }

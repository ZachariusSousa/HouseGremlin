import asyncio
import inspect
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .audio_utils import ensure_data_dirs
from .brain_models import ActionIntent, ConversationState, EventSource, WorkPriority
from .config import settings
from .coordinator import BrainCoordinator
from .correlation import current_correlation_id
from .journal import EventJournal
from .llm import OpenAICompatibleChatClient
from .realtime_gateway import RealtimeGateway
from .timing import timed


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot_http_client
    ensure_data_dirs(settings.data_dir)
    get_brain_coordinator()
    robot_http_client = httpx.AsyncClient(timeout=settings.request_timeout)
    if settings.warm_models:
        await llm_client.warmup()
    try:
        yield
    finally:
        if realtime_gateway is not None:
            await realtime_gateway.shutdown()
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
    expression: str
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


def get_brain_coordinator() -> BrainCoordinator:
    global brain_journal, brain_coordinator
    if brain_coordinator is None:
        brain_journal = EventJournal(settings.data_dir / "brain.db")
        brain_coordinator = BrainCoordinator(brain_journal)
    return brain_coordinator


def validate_action_payload(payload: dict) -> dict:
    action = RobotActionRequest.model_validate(normalize_llm_action_body(payload))
    return action.model_dump(exclude_none=True)


def get_realtime_gateway() -> RealtimeGateway:
    global realtime_gateway
    if realtime_gateway is None:
        realtime_gateway = RealtimeGateway(
            settings.realtime_ws_url,
            settings.realtime_voice,
            settings.realtime_instructions,
            get_brain_coordinator(),
            validate_action_payload,
            execute_action_payload,
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


async def robot_fetch_bytes(path: str):
    url = f"{settings.robot_base_url}{path}"
    try:
        async with robot_request_lock:
            response = await robot_client().get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Robot camera request failed: {exc}") from exc
    return response.content, response.headers.get("content-type", "image/jpeg")


def camera_urls() -> dict:
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "robit.local"
    scheme = parsed.scheme or "http"
    stream_url = urlunparse((scheme, f"{hostname}:81", "/stream", "", "", ""))
    return {
        "ok": True,
        "robot_base_url": settings.robot_base_url,
        "page_url": f"{settings.robot_base_url}/camera",
        "capture_url": f"{settings.robot_base_url}/camera/capture",
        "stream_url": stream_url,
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


async def execute_robot_action(action: RobotActionRequest) -> dict:
    payload = sanitized_action_payload(action)
    executed: list[dict] = []
    skipped: list[dict] = []

    if payload.get("emergency_stop"):
        result = await robot_emergency_stop_request()
        executed.append({"type": "emergency_stop", "result": result})
        logger.info("robot.llm_action emergency_stop")
        return {"ok": True, "action": payload, "executed": executed, "skipped": skipped}

    if movement := payload.get("movement"):
        result = await robot_post("/api/move", movement)
        executed.append({"type": "movement", "request": movement, "result": result})

    if head := payload.get("head"):
        result = await robot_post("/api/head", head)
        executed.append({"type": "head", "request": head, "result": result})

    if eyes := payload.get("eyes"):
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


async def call_llm(method_name: str, text: str, history: list[dict[str, str]]):
    method = getattr(llm_client, method_name)
    if "history" in inspect.signature(method).parameters:
        return await method(text, history=history)
    return await method(text)


def parse_action_response(content: str) -> dict | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
            "instructions": settings.realtime_instructions,
        },
    }


@app.get("/", include_in_schema=False)
async def web_control():
    if not WEB_CONTROL_INDEX.exists():
        raise HTTPException(status_code=404, detail="web_control/index.html was not found")
    return FileResponse(WEB_CONTROL_INDEX)


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
    content, media_type = await robot_fetch_bytes("/camera/capture")
    return StreamingResponse(iter([content]), media_type=media_type)


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
            "parse_error": "LLM did not return strict JSON; no robot action was executed.",
        }

    response_text = str(parsed.get("response") or "").strip()
    action_body = parsed.get("action")
    action_result = None
    if isinstance(action_body, dict) and action_body:
        try:
            action = RobotActionRequest.model_validate(normalize_llm_action_body(action_body))
        except ValidationError as exc:
            coordinator.record_turn("assistant", response_text, EventSource.text_model, correlation_id)
            coordinator.transition(correlation_id, EventSource.text_model, conversation=ConversationState.idle)
            return {
                "response": response_text or parsed.get("response") or "",
                "model": chat_result.model,
                "conversation_id": chat_request.conversation_id,
                "correlation_id": correlation_id,
                "action": action_body,
                "action_result": None,
                "parse_error": f"LLM returned an invalid robot action; no robot action was executed: {exc.errors()[0]['msg']}",
            }
        action_result = await coordinated_action(
            action,
            EventSource.text_model,
            correlation_id,
            "Text model action",
            WorkPriority.model_action,
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

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .cuda_paths import configure_windows_cuda_dll_paths

configure_windows_cuda_dll_paths()

from .audio_utils import ensure_data_dirs, save_upload
from .config import settings
from .llm import OllamaChatClient
from .stt import FasterWhisperTranscriber
from .timing import timed
from .tts import ChatterboxSynthesizer, SynthesisStreamEvent
from .voices import VoiceStore


logger = logging.getLogger("uvicorn.error")
llm_client = OllamaChatClient(settings)
transcriber = FasterWhisperTranscriber(settings)
tts = ChatterboxSynthesizer(settings)
voice_store = VoiceStore(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs(settings.data_dir, settings.voices_dir, settings.audio_dir, settings.uploads_dir)
    if settings.warm_models:
        await asyncio.gather(
            llm_client.warmup(),
            asyncio.to_thread(tts.warmup, False),
        )
    yield


app = FastAPI(title="Robit PC Brain", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
ensure_data_dirs(settings.audio_dir)
app.mount("/audio", StaticFiles(directory=settings.audio_dir), name="audio")
WEB_CONTROL_INDEX = Path(__file__).resolve().parents[2] / "web_control" / "index.html"


class DriveCommand(BaseModel):
    move: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)


class HeadCommand(BaseModel):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)


class MovementAction(BaseModel):
    direction: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)
    duration_ms: int | None = Field(default=None, ge=0)


class HeadAction(BaseModel):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)
    pan_delta: int | None = Field(default=None, ge=-80, le=80)
    tilt_delta: int | None = Field(default=None, ge=-80, le=80)


class EyeAction(BaseModel):
    expression: str
    duration_ms: int | None = Field(default=None, ge=0, le=10000)


class RobotActionRequest(BaseModel):
    movement: MovementAction | None = None
    head: HeadAction | None = None
    eyes: EyeAction | None = None
    emergency_stop: bool = False


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"


class ChatSpeakRequest(ChatRequest):
    voice_id: str | None = None


class ChatActionRequest(ChatRequest):
    voice_id: str | None = None


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str | None = None


def ndjson_stream(
    events: Iterable[SynthesisStreamEvent],
    final_extra: dict | None = None,
):
    for item in events:
        event = dict(item.event)
        if item.final_result and final_extra:
            event.update(final_extra)
        yield json.dumps(event) + "\n"


def chat_speak_stream(chat_result, voice_id: str, conversation_id: str):
    yield json.dumps(
        {
            "type": "response",
            "response": chat_result.response,
            "model": chat_result.model,
            "conversation_id": conversation_id,
            "voice_id": voice_id,
        }
    ) + "\n"

    result = tts.synthesize(chat_result.response, voice_id)
    yield json.dumps(
        {
            "type": "final",
            "audio_url": result.audio_url,
            "audio_urls": result.audio_urls,
            "voice_id": result.voice_id,
            "spoken_text": result.spoken_text,
            "tts_input_chars": result.tts_input_chars,
            "active_reference_count": result.active_reference_count,
            "response": chat_result.response,
            "model": chat_result.model,
            "conversation_id": conversation_id,
        }
    ) + "\n"


async def robot_get(path: str, params: dict | None = None):
    url = f"{settings.robot_base_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return {"ok": True, "body": response.text}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Robot request failed: {exc}") from exc


async def robot_post(path: str, body: dict | None = None):
    url = f"{settings.robot_base_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.post(url, json=body or {})
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return {"ok": True, "body": response.text}
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Robot request failed: {exc}") from exc


def camera_urls() -> dict:
    parsed = urlparse(settings.robot_base_url)
    hostname = parsed.hostname or "192.168.4.1"
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
        result = await robot_post("/api/emergency-stop")
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


def parse_action_response(content: str) -> dict | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


@app.get("/health")
async def health():
    return {
        "ok": True,
        "robot_base_url": settings.robot_base_url,
        "llm_model": settings.llm_model,
        "stt_model": settings.stt_model,
        "tts_model": settings.tts_model,
        "tts_runtime": tts.runtime_info(),
    }


@app.get("/", include_in_schema=False)
async def web_control():
    if not WEB_CONTROL_INDEX.exists():
        raise HTTPException(status_code=404, detail="web_control/index.html was not found")
    return FileResponse(WEB_CONTROL_INDEX)


@app.get("/robot/status")
async def robot_status():
    return await robot_get("/status")


@app.get("/robot/camera")
async def robot_camera():
    return camera_urls()


@app.get("/robot/camera/capture")
async def robot_camera_capture():
    url = f"{settings.robot_base_url}/camera/capture"
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Robot camera request failed: {exc}") from exc
    return StreamingResponse(
        iter([response.content]),
        media_type=response.headers.get("content-type", "image/jpeg"),
    )


@app.post("/robot/drive")
async def robot_drive(command: DriveCommand):
    if command.speed is not None:
        await robot_get("/speed", {"value": command.speed})
    return await robot_get("/cmd", {"move": command.move})


@app.post("/robot/head")
async def robot_head(command: HeadCommand):
    params = {}
    if command.pan is not None:
        params["pan"] = command.pan
    if command.tilt is not None:
        params["tilt"] = command.tilt
    if not params:
        raise HTTPException(status_code=400, detail="pan or tilt is required")
    return await robot_get("/servo", params)


@app.post("/robot/stop")
async def robot_stop():
    try:
        return await robot_post("/api/emergency-stop")
    except HTTPException:
        return await robot_get("/cmd", {"move": "stop"})


@app.post("/robot/action")
async def robot_action(action: RobotActionRequest):
    return await execute_robot_action(action)


@app.get("/voices")
async def list_voices():
    voice_ids = voice_store.list_voice_ids()
    if settings.voice_id not in voice_ids:
        voice_ids.insert(0, settings.voice_id)
    return {
        "voices": voice_ids,
        "voice_details": [
            {"voice_id": voice_id, "sample_count": voice_store.sample_count(voice_id)}
            for voice_id in voice_ids
        ],
        "default_voice_id": settings.voice_id,
    }


@app.post("/voices")
async def upload_voice(
    voice_id: str = Form(default=settings.voice_id),
    sample: UploadFile = File(...),
):
    with timed("endpoint.voices.upload", voice_id=voice_id, filename=sample.filename or "unknown"):
        with timed("upload.save", kind="voice", filename=sample.filename or "unknown"):
            uploaded = await save_upload(sample, settings.uploads_dir, "voice")
        cleaned_voice_id, reference, references = voice_store.save_reference(voice_id, uploaded)
        with timed("tts.register_voice", voice_id=cleaned_voice_id, reference_count=len(references)):
            tts.register_voice(cleaned_voice_id, references)
    return {
        "voice_id": cleaned_voice_id,
        "reference_path": str(reference),
        "sample_count": len(references),
        "ok": True,
    }


@app.post("/voice/transcribe")
async def voice_transcribe(audio: UploadFile = File(...)):
    with timed("endpoint.voice.transcribe", filename=audio.filename or "unknown"):
        with timed("upload.save", kind="stt", filename=audio.filename or "unknown"):
            uploaded = await save_upload(audio, settings.uploads_dir, "stt")
        with timed("stt.transcribe"):
            result = await asyncio.to_thread(transcriber.transcribe, uploaded)
    return {
        "text": result.text,
        "language": result.language,
        "duration_seconds": result.duration_seconds,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    with timed("endpoint.chat", prompt_chars=len(request.text)):
        result = await llm_client.chat(request.text)
    return {
        "response": result.response,
        "model": result.model,
        "conversation_id": request.conversation_id,
    }


@app.post("/chat/speak")
async def chat_speak(request: ChatSpeakRequest):
    voice_id = request.voice_id or settings.voice_id
    with timed("endpoint.chat_speak", voice_id=voice_id, prompt_chars=len(request.text)):
        with timed("stage.chat_speak.llm", prompt_chars=len(request.text)):
            chat_result = await llm_client.chat(request.text)
    return StreamingResponse(
        chat_speak_stream(chat_result, voice_id, request.conversation_id),
        media_type="application/x-ndjson",
    )


@app.post("/chat/action")
async def chat_action(request: ChatActionRequest):
    with timed("endpoint.chat_action", prompt_chars=len(request.text)):
        chat_result = await llm_client.action_chat(request.text)

    parsed = parse_action_response(chat_result.response)
    if parsed is None:
        return {
            "response": chat_result.response,
            "model": chat_result.model,
            "conversation_id": request.conversation_id,
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
            return {
                "response": response_text or parsed.get("response") or "",
                "model": chat_result.model,
                "conversation_id": request.conversation_id,
                "action": action_body,
                "action_result": None,
                "parse_error": f"LLM returned an invalid robot action; no robot action was executed: {exc.errors()[0]['msg']}",
            }
        action_result = await execute_robot_action(action)

    return {
        "response": response_text,
        "model": chat_result.model,
        "conversation_id": request.conversation_id,
        "action": action_body if isinstance(action_body, dict) else None,
        "action_result": action_result,
        "parse_error": None,
    }


@app.post("/voice/synthesize")
async def voice_synthesize(request: SynthesizeRequest):
    voice_id = request.voice_id or settings.voice_id
    with timed("endpoint.voice.synthesize", voice_id=voice_id, text_chars=len(request.text)):
        events = tts.synthesize_stream(request.text, voice_id)
    return StreamingResponse(ndjson_stream(events), media_type="application/x-ndjson")


@app.post("/voice/roundtrip")
async def voice_roundtrip(
    audio: UploadFile = File(...),
    voice_id: str = Form(default=settings.voice_id),
    conversation_id: str = Form(default="default"),
):
    with timed("endpoint.voice.roundtrip", voice_id=voice_id, filename=audio.filename or "unknown"):
        with timed("upload.save", kind="roundtrip", filename=audio.filename or "unknown"):
            uploaded = await save_upload(audio, settings.uploads_dir, "roundtrip")
        with timed("stage.roundtrip.stt"):
            transcript = await asyncio.to_thread(transcriber.transcribe, uploaded)
        with timed("stage.roundtrip.llm", transcript_chars=len(transcript.text)):
            chat_result = await llm_client.chat(transcript.text)
    events = tts.synthesize_stream(chat_result.response, voice_id)
    return StreamingResponse(
        ndjson_stream(
            events,
            {
                "conversation_id": conversation_id,
                "transcript": transcript.text,
                "response": chat_result.response,
                "model": chat_result.model,
            },
        ),
        media_type="application/x-ndjson",
    )


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

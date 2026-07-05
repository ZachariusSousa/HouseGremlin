import asyncio
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio_utils import ensure_data_dirs, save_upload
from .config import settings
from .llm import OllamaChatClient
from .stt import FasterWhisperTranscriber
from .timing import timed
from .tts import ChatterboxTurboSynthesizer
from .voices import VoiceStore


llm_client = OllamaChatClient(settings)
transcriber = FasterWhisperTranscriber(settings)
tts = ChatterboxTurboSynthesizer(settings)
voice_store = VoiceStore(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs(settings.data_dir, settings.voices_dir, settings.audio_dir, settings.uploads_dir)
    if settings.warm_models:
        await asyncio.gather(
            llm_client.warmup(),
            asyncio.to_thread(transcriber.warmup),
            asyncio.to_thread(tts.warmup),
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


class DriveCommand(BaseModel):
    move: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)


class HeadCommand(BaseModel):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"


class ChatSpeakRequest(ChatRequest):
    voice_id: str | None = None


class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str | None = None


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


@app.get("/robot/status")
async def robot_status():
    return await robot_get("/status")


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
    return await robot_get("/cmd", {"move": "stop"})


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
        with timed("stage.chat_speak.tts", voice_id=voice_id, response_chars=len(chat_result.response)):
            speech = await asyncio.to_thread(tts.synthesize, chat_result.response, voice_id)
    return {
        "response": chat_result.response,
        "model": chat_result.model,
        "conversation_id": request.conversation_id,
        "audio_url": speech.audio_url,
        "audio_urls": speech.audio_urls,
        "voice_id": speech.voice_id,
        "spoken_text": speech.spoken_text,
        "tts_input_chars": speech.tts_input_chars,
        "active_reference_count": speech.active_reference_count,
    }


@app.post("/voice/synthesize")
async def voice_synthesize(request: SynthesizeRequest):
    voice_id = request.voice_id or settings.voice_id
    with timed("endpoint.voice.synthesize", voice_id=voice_id, text_chars=len(request.text)):
        result = await asyncio.to_thread(tts.synthesize, request.text, voice_id)
    return {
        "audio_url": result.audio_url,
        "audio_urls": result.audio_urls,
        "voice_id": result.voice_id,
        "spoken_text": result.spoken_text,
        "tts_input_chars": result.tts_input_chars,
        "active_reference_count": result.active_reference_count,
    }


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
        with timed("stage.roundtrip.tts", voice_id=voice_id, response_chars=len(chat_result.response)):
            speech = await asyncio.to_thread(tts.synthesize, chat_result.response, voice_id)
    return {
        "conversation_id": conversation_id,
        "transcript": transcript.text,
        "response": chat_result.response,
        "model": chat_result.model,
        "audio_url": speech.audio_url,
        "audio_urls": speech.audio_urls,
        "voice_id": speech.voice_id,
        "spoken_text": speech.spoken_text,
        "tts_input_chars": speech.tts_input_chars,
        "active_reference_count": speech.active_reference_count,
    }


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
        ]
    }

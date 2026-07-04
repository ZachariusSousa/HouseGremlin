import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROBIT_BASE_URL = os.getenv("ROBIT_BASE_URL", "http://192.168.4.1").rstrip("/")
REQUEST_TIMEOUT = 2.0

app = FastAPI(title="Robit PC Brain", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DriveCommand(BaseModel):
    move: Literal["forward", "reverse", "left", "right", "stop"]
    speed: int | None = Field(default=None, ge=0, le=255)


class HeadCommand(BaseModel):
    pan: int | None = Field(default=None, ge=55, le=135)
    tilt: int | None = Field(default=None, ge=35, le=115)


async def robot_get(path: str, params: dict | None = None):
    url = f"{ROBIT_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
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
    return {"ok": True, "robot_base_url": ROBIT_BASE_URL}


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

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field


MODEL_NAME = "Roboflow/rf-detr-nano"
PERSON_LABEL = "person"


class PersonDetection(BaseModel):
    label: str = PERSON_LABEL
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: tuple[float, float, float, float]


class DetectionResponse(BaseModel):
    frame_id: str
    captured_at: str
    model: str = MODEL_NAME
    backend: str
    latency_ms: float
    people: list[PersonDetection]


@dataclass
class BackendStatus:
    available: bool = False
    backend: str = "unavailable"
    reason: str | None = "RF-DETR has not loaded"


class RFDetrBackend:
    """Small synchronous wrapper; the API runs its work in a worker thread."""

    def __init__(self, requested_device: str | None = None):
        self.requested_device = (requested_device or os.getenv("ROBIT_TRACKING_DEVICE", "auto")).strip().lower()
        self.model: Any | None = None
        self.status = BackendStatus()

    def load(self) -> None:
        try:
            import torch
            from rfdetr import RFDETRNano

            device = self._resolve_device(torch)
            model = RFDETRNano(device=device)
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            model.optimize_for_inference(compile=False, dtype=dtype, inplace=True)
            self.model = model
            precision = "bfloat16" if device.startswith("cuda") else "float32"
            self.status = BackendStatus(available=True, backend=f"{device}/{precision}", reason=None)
        except Exception as exc:
            self.model = None
            self.status = BackendStatus(reason=f"{type(exc).__name__}: {exc}")

    def _resolve_device(self, torch: Any) -> str:
        requested = self.requested_device
        if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError("ROBIT_TRACKING_DEVICE must be auto, cpu, cuda, or cuda:N")
        if requested == "cpu":
            return "cpu"
        if requested == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return "cuda"
            return "cpu"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA was requested but this GPU does not support bfloat16")
        return requested

    def detect(self, jpeg: bytes, threshold: float) -> tuple[list[PersonDetection], float]:
        if self.model is None:
            raise RuntimeError(self.status.reason or "RF-DETR is unavailable")

        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(jpeg)) as source:
            image = source.convert("RGB")
            width, height = image.size
            started = perf_counter()
            detections = self.model.predict(image, threshold=threshold, include_source_image=False)
            latency_ms = (perf_counter() - started) * 1000.0

        detection_names = list(getattr(detections, "data", {}).get("class_name", []))
        people: list[PersonDetection] = []
        for index, (box, confidence) in enumerate(zip(detections.xyxy, detections.confidence)):
            label = str(detection_names[index]) if index < len(detection_names) else ""
            if label.strip().lower() != PERSON_LABEL:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            people.append(
                PersonDetection(
                    confidence=float(confidence),
                    bounding_box=(
                        min(1.0, max(0.0, x1 / width)),
                        min(1.0, max(0.0, y1 / height)),
                        min(1.0, max(0.0, x2 / width)),
                        min(1.0, max(0.0, y2 / height)),
                    ),
                )
            )
        people.sort(key=lambda detection: detection.confidence, reverse=True)
        return people, latency_ms


backend = RFDetrBackend()
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(backend.load)
    yield


app = FastAPI(title="Robit RF-DETR Sidecar", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": backend.status.available,
        "available": backend.status.available,
        "reason": backend.status.reason,
        "backend": backend.status.backend,
        "model": MODEL_NAME,
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    x_robit_frame_id: str = Header(min_length=1),
    x_robit_captured_at: str = Header(min_length=1),
    x_robit_threshold: float = Header(default=0.55, ge=0.0, le=1.0),
) -> DetectionResponse:
    if not backend.status.available:
        raise HTTPException(status_code=503, detail=backend.status.reason or "RF-DETR is unavailable")
    jpeg = await request.body()
    if not jpeg:
        raise HTTPException(status_code=400, detail="JPEG body is required")
    try:
        async with inference_lock:
            people, latency_ms = await asyncio.to_thread(backend.detect, jpeg, x_robit_threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RF-DETR inference failed: {exc}") from exc
    return DetectionResponse(
        frame_id=x_robit_frame_id,
        captured_at=x_robit_captured_at,
        backend=backend.status.backend,
        latency_ms=latency_ms,
        people=people,
    )

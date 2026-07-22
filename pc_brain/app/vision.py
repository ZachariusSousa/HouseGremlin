from __future__ import annotations

import asyncio
import base64
import importlib.util
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from time import monotonic, perf_counter
from typing import Any, Awaitable, Callable, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .brain_models import (
    ConversationState,
    EventSource,
    SceneEntity,
    SceneSnapshot,
    WorkPriority,
    WorldState,
)
from .config import Settings
from .coordinator import BrainCoordinator
from .frame_broker import CameraFrame, FrameBroker


class VisionUnavailable(RuntimeError):
    pass


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=600)
    entities: list[SceneEntity] = Field(default_factory=list, max_length=8)
    uncertainty: float = Field(ge=0.0, le=1.0)


class VisionAdapter(Protocol):
    model_name: str

    async def infer(self, image: Any, question: str) -> VisionOutput: ...


@dataclass(frozen=True)
class FrameQuality:
    image: Any
    preview: Any
    blur_score: float
    novelty: float
    blurred: bool
    changed: bool


@dataclass(frozen=True)
class VisionQueryResult:
    snapshot: SceneSnapshot
    fresh: bool
    warning: str | None = None


def vision_dependencies_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("PIL", "numpy"))


def inspect_frame(
    content: bytes,
    previous_preview: Any | None,
    rotate_degrees: int,
    change_threshold: float,
    blur_threshold: float,
) -> FrameQuality:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised on machines without vision extras.
        raise VisionUnavailable("Install the PC brain dependencies with Scripts\\setup.bat") from exc

    image = Image.open(BytesIO(content)).convert("RGB")
    if rotate_degrees:
        image = image.rotate(rotate_degrees, expand=True)
    preview = np.asarray(image.resize((160, 120)).convert("L"), dtype=np.float32)
    center = preview[1:-1, 1:-1]
    laplacian = (
        preview[:-2, 1:-1]
        + preview[2:, 1:-1]
        + preview[1:-1, :-2]
        + preview[1:-1, 2:]
        - (4.0 * center)
    )
    blur_score = float(laplacian.var())
    novelty = 1.0
    if previous_preview is not None:
        novelty = float(np.mean(np.abs(preview - previous_preview)) / 255.0)
    return FrameQuality(
        image=image,
        preview=preview,
        blur_score=blur_score,
        novelty=min(1.0, max(0.0, novelty)),
        blurred=blur_score < blur_threshold,
        changed=previous_preview is None or novelty >= change_threshold,
    )


class LlamaServerVisionAdapter:
    def __init__(self, base_url: str, model_name: str, timeout_seconds: float, max_output_tokens: int):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens

    async def probe(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/models")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise VisionUnavailable(f"E4B llama-server is unavailable: {exc}") from exc
        if not isinstance(payload, dict):
            raise VisionUnavailable("E4B llama-server returned invalid model metadata")
        models = [
            item
            for collection in (payload.get("data", []), payload.get("models", []))
            if isinstance(collection, list)
            for item in collection
            if isinstance(item, dict)
        ]
        matching = [
            item
            for item in models
            if self.model_name in {item.get("id"), item.get("name"), item.get("model")}
        ]
        candidates = matching or models
        if not any(self._is_multimodal(model) for model in candidates):
            raise VisionUnavailable("E4B llama-server is running without multimodal image capability")

    @staticmethod
    def _is_multimodal(model: dict[str, Any]) -> bool:
        if model.get("multimodal") is True:
            return True
        capabilities = model.get("capabilities")
        if isinstance(capabilities, dict):
            if any(capabilities.get(key) is True for key in ("multimodal", "vision", "image")):
                return True
        if isinstance(capabilities, list) and any(value in capabilities for value in ("multimodal", "vision", "image")):
            return True
        modalities = model.get("modalities") or model.get("input_modalities")
        return isinstance(modalities, list) and "image" in modalities

    async def infer(self, image: Any, question: str) -> VisionOutput:
        system_prompt = (
            "You are a stateless visual parser. Analyze only pixels in the attached image. "
            "Do not use application identity, prior conversation, or assumptions about who owns the image. "
            "Use generic visible-object labels and say unknown when evidence is insufficient. "
            "Return only the requested JSON schema."
        )
        prompt = (
            "Answer this question from visible evidence only: " + question + " "
            "Keep the summary concise. Include at most eight clearly visible entities. "
            "Omit bounding_box unless its position is useful; coordinates are normalized [x1,y1,x2,y2]."
        )
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        request = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": 0,
            "cache_prompt": False,
            "max_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "robit_scene",
                    "strict": True,
                    "schema": VisionOutput.model_json_schema(),
                },
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
            text = payload["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise VisionUnavailable(f"E4B vision inference failed: {exc}") from exc
        return VisionOutput.model_validate_json(text)


class VisionService:
    def __init__(
        self,
        settings: Settings,
        coordinator: BrainCoordinator,
        frame_broker: FrameBroker,
        adapter: VisionAdapter | None = None,
    ):
        self.settings = settings
        self.coordinator = coordinator
        self.frame_broker = frame_broker
        adapter_was_provided = adapter is not None
        self.adapter = adapter or LlamaServerVisionAdapter(
            settings.vision_base_url,
            settings.vision_model,
            settings.vision_request_timeout_seconds,
            settings.vision_max_output_tokens,
        )
        self.enabled = settings.vision_enabled and (adapter_was_provided or vision_dependencies_available())
        self.unavailable_reason = None if self.enabled else "Vision is disabled or optional dependencies are missing"
        self._snapshots: deque[SceneSnapshot] = deque(maxlen=100)
        self._preview = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._last_awareness_at = 0.0
        self._awareness_cancel: asyncio.Event | None = None
        self._snapshot_listeners: list[Callable[[SceneSnapshot], Awaitable[None] | None]] = []
        self.coordinator.subscribe_state(self._observe_state)

    @property
    def latest(self) -> SceneSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def subscribe_snapshot(self, listener: Callable[[SceneSnapshot], Awaitable[None] | None]) -> None:
        if listener not in self._snapshot_listeners:
            self._snapshot_listeners.append(listener)

    def current_snapshot(self) -> SceneSnapshot | None:
        return self._current_snapshot()

    async def start(self) -> None:
        if self.enabled and hasattr(self.adapter, "probe"):
            try:
                await self.adapter.probe()
            except VisionUnavailable as exc:
                self.enabled = False
                self.unavailable_reason = str(exc)
        if self.enabled and (self._task is None or self._task.done()):
            self._stopping = False
            self._task = asyncio.create_task(self._awareness_loop(), name="robit-vision-awareness")

    async def shutdown(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def world_state(self) -> WorldState:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.settings.vision_world_window_seconds)
        recent = [snapshot for snapshot in self._snapshots if snapshot.observed_at >= cutoff]
        current = recent[-1] if recent and recent[-1].expires_at > now else None
        return WorldState(
            snapshot_ids=[snapshot.frame_id for snapshot in recent],
            summary=current.summary if current else "unknown",
            entities=list(current.entities) if current else [],
        )

    def latest_payload(self) -> dict[str, Any]:
        snapshot = self.latest
        now = datetime.now(timezone.utc)
        return {
            "available": snapshot is not None,
            "enabled": self.enabled,
            "reason": self.unavailable_reason,
            "stale": snapshot is None or snapshot.expires_at <= now,
            "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
            "world_state": self.world_state().model_dump(mode="json"),
        }

    def awareness_ready(self, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        return (
            self.enabled
            and self.coordinator.state.conversation == ConversationState.idle
            and current - self._last_awareness_at >= self.settings.vision_awareness_interval_seconds
        )

    def _observe_state(self, state, event) -> None:
        if state.conversation != ConversationState.idle and self._awareness_cancel is not None:
            self._awareness_cancel.set()

    async def query(self, question: str, fresh: bool = True) -> VisionQueryResult:
        if not self.enabled:
            if self._current_snapshot() is not None:
                return VisionQueryResult(self.latest, False, self.unavailable_reason)
            raise VisionUnavailable(self.unavailable_reason or "Vision is unavailable")
        try:
            snapshot = await self._process_explicit(question, fresh)
            return VisionQueryResult(snapshot, True)
        except (VisionUnavailable, ValidationError, ValueError, OSError, RuntimeError) as exc:
            if self._current_snapshot() is not None:
                return VisionQueryResult(self.latest, False, str(exc))
            raise VisionUnavailable(str(exc)) from exc

    def _current_snapshot(self) -> SceneSnapshot | None:
        snapshot = self.latest
        if snapshot is None or snapshot.expires_at <= datetime.now(timezone.utc):
            return None
        return snapshot

    async def _process_explicit(self, question: str, fresh: bool) -> SceneSnapshot:
        last_error: Exception | None = None
        for _ in range(2):
            frame = await self.frame_broker.get_frame(force_fresh=fresh)
            quality = inspect_frame(
                frame.content,
                self._preview,
                self.settings.camera_rotate_degrees,
                self.settings.vision_change_threshold,
                self.settings.vision_blur_threshold,
            )
            self._preview = quality.preview
            if quality.blurred:
                last_error = VisionUnavailable(f"Camera frame was blurred (score {quality.blur_score:.1f})")
                fresh = True
                continue
            try:
                async with self.coordinator.resource_lease.acquire(WorkPriority.foreground):
                    return await self._infer(frame, quality, question, "explicit")
            except (VisionUnavailable, ValidationError, ValueError) as exc:
                last_error = exc
                fresh = True
                continue
        raise last_error or VisionUnavailable("No usable camera frame was available")

    async def _awareness_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(1.0)
            if not self.awareness_ready():
                continue
            try:
                frame = await self.frame_broker.get_frame()
                quality = inspect_frame(
                    frame.content,
                    self._preview,
                    self.settings.camera_rotate_degrees,
                    self.settings.vision_change_threshold,
                    self.settings.vision_blur_threshold,
                )
                self._preview = quality.preview
                if quality.blurred:
                    continue
                if not quality.changed and await self._carry_forward_snapshot(frame, quality):
                    self._last_awareness_at = monotonic()
                    continue
                cancelled = asyncio.Event()
                self._awareness_cancel = cancelled
                try:
                    async with self.coordinator.resource_lease.acquire(WorkPriority.background, cancelled.set):
                        if cancelled.is_set() or self.coordinator.state.conversation != ConversationState.idle:
                            continue
                        await self._infer(
                            frame,
                            quality,
                            "Describe the current scene briefly.",
                            "awareness",
                            cancelled,
                        )
                        self._last_awareness_at = monotonic()
                finally:
                    self._awareness_cancel = None
            except (VisionUnavailable, ValidationError, ValueError, OSError, RuntimeError) as exc:
                self.coordinator.record(
                    "perception.awareness.failed",
                    EventSource.system,
                    self.coordinator.new_correlation_id(),
                    {"error": str(exc)},
                    WorkPriority.background,
                )

    async def _infer(
        self,
        frame: CameraFrame,
        quality: FrameQuality,
        question: str,
        trigger: str,
        cancellation_requested: asyncio.Event | None = None,
    ) -> SceneSnapshot:
        started = perf_counter()
        output = await self.adapter.infer(quality.image, question)
        if cancellation_requested is not None and cancellation_requested.is_set():
            raise VisionUnavailable("Background awareness was preempted by foreground work")
        observed_at = frame.captured_at
        snapshot = SceneSnapshot(
            frame_id=frame.frame_id,
            observed_at=observed_at,
            trigger=trigger,
            summary=output.summary,
            entities=output.entities,
            novelty=quality.novelty,
            uncertainty=output.uncertainty,
            model=self.adapter.model_name,
            latency_ms=(perf_counter() - started) * 1000.0,
            expires_at=observed_at + timedelta(seconds=self.settings.vision_snapshot_ttl_seconds),
        )
        await self._store_snapshot(snapshot, trigger)
        return snapshot

    async def _carry_forward_snapshot(self, frame: CameraFrame, quality: FrameQuality) -> bool:
        previous = self.latest
        if previous is None:
            return False
        snapshot = SceneSnapshot(
            frame_id=frame.frame_id,
            observed_at=frame.captured_at,
            trigger="awareness",
            summary=previous.summary,
            entities=previous.entities,
            novelty=quality.novelty,
            uncertainty=previous.uncertainty,
            model=previous.model,
            latency_ms=0.0,
            expires_at=frame.captured_at + timedelta(seconds=self.settings.vision_snapshot_ttl_seconds),
        )
        await self._store_snapshot(snapshot, "awareness")
        return True

    async def _store_snapshot(self, snapshot: SceneSnapshot, trigger: str) -> None:
        self._snapshots.append(snapshot)
        self.coordinator.record(
            "perception.snapshot.created",
            EventSource.system,
            self.coordinator.new_correlation_id(),
            {"snapshot": snapshot.model_dump(mode="json")},
            WorkPriority.background if trigger == "awareness" else WorkPriority.foreground,
        )
        for listener in tuple(self._snapshot_listeners):
            try:
                result = listener(snapshot)
                if result is not None:
                    await result
            except (OSError, RuntimeError, ValueError) as exc:
                self.coordinator.record(
                    "perception.snapshot.listener_failed",
                    EventSource.system,
                    self.coordinator.new_correlation_id(),
                    {"error": str(exc)},
                    WorkPriority.background,
                )

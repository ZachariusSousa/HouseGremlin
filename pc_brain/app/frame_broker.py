from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from uuid import uuid4


FrameFetcher = Callable[[], Awaitable[tuple[bytes, str]]]


@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    captured_at: datetime
    content: bytes
    media_type: str = "image/jpeg"


class FrameBroker:
    """Fetch at most one robot frame per interval and share it with all consumers."""

    def __init__(self, fetcher: FrameFetcher, interval_seconds: float = 1.0, max_fps: float = 3.0):
        self.fetcher = fetcher
        self.interval_seconds = max(0.05, interval_seconds)
        self.max_fps = max(0.1, max_fps)
        self._rate_leases: dict[str, float] = {}
        self._frame: CameraFrame | None = None
        self._last_fetch_at = 0.0
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._rate_changed = asyncio.Event()
        self._stopping = False

    @property
    def latest(self) -> CameraFrame | None:
        return self._frame

    @property
    def effective_fps(self) -> float:
        base_fps = min(self.max_fps, 1.0 / self.interval_seconds)
        return min(self.max_fps, max([base_fps, *self._rate_leases.values()]))

    @property
    def effective_interval_seconds(self) -> float:
        return 1.0 / self.effective_fps

    def set_rate_lease(self, owner: str, fps: float | None) -> None:
        """Raise the shared acquisition rate for one consumer without adding another poller."""
        if fps is None or fps <= 0:
            self._rate_leases.pop(owner, None)
        else:
            self._rate_leases[owner] = min(self.max_fps, max(0.1, float(fps)))
        self._rate_changed.set()

    def release_rate_lease(self, owner: str) -> None:
        self.set_rate_lease(owner, None)

    async def get_frame(self, force_fresh: bool = False) -> CameraFrame:
        async with self._lock:
            age = monotonic() - self._last_fetch_at
            interval = self.effective_interval_seconds
            if self._frame is not None and not force_fresh and age < interval:
                return self._frame
            if force_fresh and self._frame is not None and age < interval:
                await asyncio.sleep(interval - age)
            content, media_type = await self.fetcher()
            self._frame = CameraFrame(
                frame_id=str(uuid4()),
                captured_at=datetime.now(timezone.utc),
                content=content,
                media_type=media_type,
            )
            self._last_fetch_at = monotonic()
            return self._frame

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._poll(), name="robit-frame-broker")

    async def shutdown(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _poll(self) -> None:
        while not self._stopping:
            self._rate_changed.clear()
            try:
                await self.get_frame()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._rate_changed.wait(), timeout=self.effective_interval_seconds)
            except TimeoutError:
                pass

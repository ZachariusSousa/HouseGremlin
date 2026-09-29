from __future__ import annotations

import asyncio
import heapq
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import count
from typing import AsyncIterator, Callable

from .brain_models import WorkPriority


@dataclass(order=True)
class _Waiter:
    priority: int
    order: int
    ready: asyncio.Future[None] = field(compare=False)
    queued_at: float = field(compare=False, default=0.0)


logger = logging.getLogger("uvicorn.error")


class PriorityResourceLease:
    """A small priority lease used by foreground voice and future GPU jobs."""

    def __init__(self):
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._counter = count()
        self._held = False
        self._holder_priority: int | None = None
        self._holder_cancel: Callable[[], None] | None = None
        self._holder_task: asyncio.Task | None = None
        self._holder_started_at: float | None = None
        self._foreground_barrier = False
        self._cancel_tasks: set[asyncio.Task] = set()
        self._last_queue_ms = 0.0

    @asynccontextmanager
    async def acquire(
        self,
        priority: WorkPriority,
        cancel_holder: Callable[[], None] | None = None,
    ) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            int(priority),
            next(self._counter),
            loop.create_future(),
            loop.time(),
        )
        async with self._condition:
            heapq.heappush(self._waiters, waiter)
            if self._held and self._holder_priority is not None and int(priority) < self._holder_priority:
                if self._holder_cancel:
                    self._holder_cancel()
                if (
                    self._holder_task is not None
                    and self._holder_priority >= int(WorkPriority.target_reacquisition)
                ):
                    cancellation = asyncio.create_task(
                        self._cancel_after_grace(self._holder_task)
                    )
                    self._cancel_tasks.add(cancellation)
                    cancellation.add_done_callback(self._cancel_tasks.discard)
            while self._foreground_barrier and int(priority) >= int(WorkPriority.background):
                await self._condition.wait()
            self._wake_next_locked()
            while not waiter.ready.done():
                await self._condition.wait()
                self._wake_next_locked()
            self._held = True
            self._holder_priority = int(priority)
            self._holder_cancel = cancel_holder
            self._holder_task = asyncio.current_task()
            self._holder_started_at = asyncio.get_running_loop().time()
            self._last_queue_ms = (
                self._holder_started_at - waiter.queued_at
            ) * 1000.0
            if self._last_queue_ms >= 5.0:
                logger.info(
                    "workload.acquired priority=%s queue_ms=%.2f",
                    int(priority),
                    self._last_queue_ms,
                )
        try:
            yield
        finally:
            async with self._condition:
                self._held = False
                self._holder_priority = None
                self._holder_cancel = None
                self._holder_task = None
                self._holder_started_at = None
                self._wake_next_locked()
                self._condition.notify_all()

    def _wake_next_locked(self) -> None:
        if not self._held and self._waiters:
            if (
                self._foreground_barrier
                and self._waiters[0].priority >= int(WorkPriority.background)
            ):
                return
            waiter = heapq.heappop(self._waiters)
            if not waiter.ready.done():
                # Reserve the lease before notifying all waiters. Without this,
                # another awakened waiter can pop itself before the winner runs.
                self._held = True
                waiter.ready.set_result(None)

    async def set_foreground_barrier(self, active: bool) -> None:
        async with self._condition:
            if self._foreground_barrier == active:
                return
            self._foreground_barrier = active
            if (
                active
                and self._holder_task is not None
                and self._holder_priority is not None
                and self._holder_priority >= int(WorkPriority.background)
            ):
                if self._holder_cancel:
                    self._holder_cancel()
                cancellation = asyncio.create_task(
                    self._cancel_after_grace(self._holder_task)
                )
                self._cancel_tasks.add(cancellation)
                cancellation.add_done_callback(self._cancel_tasks.discard)
            self._condition.notify_all()

    async def _cancel_after_grace(self, task: asyncio.Task) -> None:
        await asyncio.sleep(0.05)
        if not task.done():
            task.cancel()

    def status(self) -> dict:
        now = asyncio.get_running_loop().time()
        return {
            "active": self._held,
            "priority": self._holder_priority,
            "queue_depth": len(self._waiters),
            "foreground_barrier": self._foreground_barrier,
            "active_seconds": (
                now - self._holder_started_at
                if self._holder_started_at is not None
                else 0.0
            ),
            "last_queue_ms": self._last_queue_ms,
        }

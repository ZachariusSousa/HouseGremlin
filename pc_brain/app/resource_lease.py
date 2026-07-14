from __future__ import annotations

import asyncio
import heapq
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


class PriorityResourceLease:
    """A small priority lease used by foreground voice and future GPU jobs."""

    def __init__(self):
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._counter = count()
        self._held = False
        self._holder_priority: int | None = None
        self._holder_cancel: Callable[[], None] | None = None

    @asynccontextmanager
    async def acquire(
        self,
        priority: WorkPriority,
        cancel_holder: Callable[[], None] | None = None,
    ) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        waiter = _Waiter(int(priority), next(self._counter), loop.create_future())
        async with self._condition:
            heapq.heappush(self._waiters, waiter)
            if self._held and self._holder_priority is not None and int(priority) < self._holder_priority:
                if self._holder_cancel:
                    self._holder_cancel()
            self._wake_next_locked()
            while not waiter.ready.done():
                await self._condition.wait()
                self._wake_next_locked()
            self._held = True
            self._holder_priority = int(priority)
            self._holder_cancel = cancel_holder
        try:
            yield
        finally:
            async with self._condition:
                self._held = False
                self._holder_priority = None
                self._holder_cancel = None
                self._wake_next_locked()
                self._condition.notify_all()

    def _wake_next_locked(self) -> None:
        if not self._held and self._waiters:
            waiter = heapq.heappop(self._waiters)
            if not waiter.ready.done():
                # Reserve the lease before notifying all waiters. Without this,
                # another awakened waiter can pop itself before the winner runs.
                self._held = True
                waiter.ready.set_result(None)

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from app.brain_models import BrainEvent, EventSource, WorkPriority
from app.frame_broker import FrameBroker
from app.journal import EventJournal
from app.resource_lease import PriorityResourceLease


@pytest.mark.anyio
async def test_foreground_terminates_noncooperative_background_work():
    scheduler = PriorityResourceLease()
    background_started = asyncio.Event()
    cancellation_requested = asyncio.Event()

    async def background():
        async with scheduler.acquire(
            WorkPriority.background, cancellation_requested.set
        ):
            background_started.set()
            await asyncio.Event().wait()

    async def foreground():
        async with scheduler.acquire(WorkPriority.foreground):
            return "ready"

    background_task = asyncio.create_task(background())
    await background_started.wait()
    assert await asyncio.wait_for(foreground(), 1) == "ready"
    assert cancellation_requested.is_set()
    await asyncio.gather(background_task, return_exceptions=True)
    assert background_task.cancelled()


@pytest.mark.anyio
async def test_frame_variants_share_one_camera_acquisition():
    source = BytesIO()
    Image.new("RGB", (640, 480), "navy").save(source, format="JPEG")
    fetches = 0

    async def fetch():
        nonlocal fetches
        fetches += 1
        return source.getvalue(), "image/jpeg"

    broker = FrameBroker(fetch, interval_seconds=5, max_fps=2)
    raw = await broker.get_frame()
    rotated_frame, rotated = await broker.get_rotated_jpeg(180)
    preview_frame, preview = await broker.get_preview_jpeg()

    assert fetches == 1
    assert raw.frame_id == rotated_frame.frame_id == preview_frame.frame_id
    assert rotated and preview
    assert broker.status()["last_frame_bytes"] == len(source.getvalue())


def test_journal_uses_bounded_writer_and_flushes_durable_events(tmp_path):
    journal = EventJournal(tmp_path / "brain.db", queue_limit=10)
    event = BrainEvent(
        event_type="action.completed",
        occurred_at=datetime.now(timezone.utc),
        source=EventSource.firmware,
        correlation_id="test",
        priority=WorkPriority.manual_action,
        payload={"ok": True},
    )
    committed = journal.append(event)

    assert committed.sequence is not None
    assert journal.status()["capacity"] == 10
    assert journal.latest_sequence() == committed.sequence
    journal.close()

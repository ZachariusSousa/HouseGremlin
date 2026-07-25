from __future__ import annotations

from datetime import datetime, timezone

import pytest

import app.tracking as tracking_module
from app.coordinator import BrainCoordinator
from app.frame_broker import FrameBroker
from app.journal import EventJournal
from app.tracking import (
    DetectorResult,
    PersonCandidate,
    PersonTrackingService,
)


class FakeDetector:
    available = True
    reason = None
    backend = "cuda/bfloat16"
    model = "Roboflow/rf-detr-nano"

    async def probe(self):
        return True


def candidate(box, confidence=0.9):
    return PersonCandidate(confidence=confidence, bounding_box=box)


def detection(frame_id, people):
    return DetectorResult(
        frame_id=frame_id,
        captured_at=datetime.now(timezone.utc),
        model="Roboflow/rf-detr-nano",
        backend="cuda/bfloat16",
        latency_ms=10,
        people=people,
    )


def service(tmp_path):
    heads = []
    pivots = []

    async def fetch():
        return b"jpeg", "image/jpeg"

    async def head(pan, tilt, generation, allow_search=False):
        heads.append((pan, tilt))
        return {"ok": True, "pan_target": pan, "tilt_target": tilt}

    async def move(direction, speed, duration_ms, generation):
        pivots.append((direction, speed, duration_ms))
        return {"ok": True}

    coordinator = BrainCoordinator(EventJournal(tmp_path / "brain.db"))
    broker = FrameBroker(fetch)
    tracker = PersonTrackingService(
        coordinator,
        broker,
        FakeDetector(),
        head,
        move,
    )
    return tracker, heads, pivots


@pytest.mark.anyio
async def test_requires_two_consistent_detections_and_holds_brief_occlusion(
    tmp_path, monkeypatch
):
    clock = [100.0]
    monkeypatch.setattr(tracking_module, "monotonic", lambda: clock[0])
    tracker, _, _ = service(tmp_path)
    person = candidate((0.35, 0.1, 0.65, 0.9))

    await tracker.process_result(detection("one", [person]))
    assert tracker.state == "acquiring"
    assert tracker.target is None

    clock[0] += 0.5
    await tracker.process_result(detection("two", [person]))
    assert tracker.state == "tracking"
    assert tracker.target is not None

    clock[0] += 2.4
    await tracker.process_result(detection("missing-briefly", []))
    assert tracker.state == "tracking"
    assert tracker.target is not None

    clock[0] += 0.2
    await tracker.process_result(detection("lost", []))
    assert tracker.state == "lost"
    assert tracker.target is None


@pytest.mark.anyio
async def test_lost_target_gets_one_neutral_command_after_five_seconds(
    tmp_path, monkeypatch
):
    clock = [200.0]
    monkeypatch.setattr(tracking_module, "monotonic", lambda: clock[0])
    tracker, heads, _ = service(tmp_path)

    await tracker.process_result(detection("none", []))
    clock[0] += 5.1
    await tracker.process_result(detection("still-none", []))
    await tracker.process_result(detection("still-none-2", []))

    assert heads.count((90, 90)) == 1
    assert tracker.status().command_counts["neutral"] == 1


@pytest.mark.anyio
async def test_new_person_needs_three_superior_detections_before_switch(
    tmp_path, monkeypatch
):
    clock = [300.0]
    monkeypatch.setattr(tracking_module, "monotonic", lambda: clock[0])
    tracker, _, _ = service(tmp_path)
    first = candidate((0.1, 0.1, 0.3, 0.9), 0.70)
    superior = candidate((0.7, 0.1, 0.95, 0.95), 0.95)

    await tracker.process_result(detection("a1", [first]))
    clock[0] += 0.5
    await tracker.process_result(detection("a2", [first]))
    original_track = tracker.target.track_id

    for index in range(2):
        clock[0] += 0.5
        await tracker.process_result(detection(f"s{index}", [superior]))
        assert tracker.target.track_id == original_track

    clock[0] += 0.5
    await tracker.process_result(detection("s3", [superior]))
    assert tracker.target.track_id != original_track
    assert tracker.status().association_result == "switched"

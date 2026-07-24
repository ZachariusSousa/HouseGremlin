from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO

import httpx
import pytest
from PIL import Image

from app.coordinator import BrainCoordinator
from app.frame_broker import CameraFrame, FrameBroker
from app.journal import EventJournal
from app.tracking import (
    DetectorResult,
    PersonCandidate,
    PersonTrackingService,
    RFDetrClient,
    rotate_jpeg,
)


class FakeDetector:
    available = True
    reason = None
    backend = "cuda/bfloat16"
    model = "Roboflow/rf-detr-nano"

    async def probe(self):
        return self.available


def make_service(tmp_path, *, skip_head=False):
    async def fetch():
        return b"jpeg", "image/jpeg"

    head_commands: list[tuple[int, int]] = []
    move_commands: list[tuple[str, int, int]] = []

    async def head(pan, tilt, generation, allow_search=False):
        head_commands.append((pan, tilt))
        return {"ok": False, "skipped": "authorization expired"} if skip_head else {"ok": True}

    async def move(direction, speed, duration_ms, generation):
        move_commands.append((direction, speed, duration_ms))
        return {"ok": True}

    coordinator = BrainCoordinator(EventJournal(tmp_path / "tracking.db"))
    broker = FrameBroker(fetch, interval_seconds=5.0, max_fps=3.0)
    service = PersonTrackingService(coordinator, broker, FakeDetector(), head, move)
    return service, broker, head_commands, move_commands


def result(frame_id, people, latency_ms=15.0):
    return DetectorResult(
        frame_id=frame_id,
        captured_at=datetime.now(timezone.utc),
        model="Roboflow/rf-detr-nano",
        backend="cuda/bfloat16",
        latency_ms=latency_ms,
        people=people,
    )


def person(box, confidence=0.9):
    return PersonCandidate(confidence=confidence, bounding_box=box)


@pytest.mark.anyio
async def test_brief_detection_miss_keeps_target_and_sustained_loss_reacquires(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    await service.process_result(
        result(
            "one",
            [
                person((0.35, 0.1, 0.65, 0.9)),
                person((0.75, 0.2, 0.95, 0.8)),
            ],
        )
    )
    assert service.target is not None
    track_id = service.target.track_id

    await service.process_result(result("two", []))
    assert service.mode == "track"
    assert service.state == "tracking"
    assert service.target is not None
    assert service.target.track_id == track_id

    await service.process_result(result("three", [person((0.32, 0.1, 0.62, 0.9))]))
    assert service.mode == "track"
    assert service.state == "tracking"
    assert service.target is not None
    assert service.target.track_id == track_id

    for frame_id in ("four", "five", "six", "seven", "eight", "nine"):
        await service.process_result(result(frame_id, []))
    assert service.state == "searching"
    assert service.target is None

    await service.process_result(result("ten", [person((0.0, 0.1, 0.2, 0.9))]))
    assert service.target is not None
    assert service.target.track_id != track_id


@pytest.mark.anyio
async def test_head_steps_are_bounded_and_default_tracking_can_pivot(tmp_path):
    service, _, head_commands, move_commands = make_service(tmp_path)
    for index in range(2):
        await service.process_result(result(str(index), [person((0.72, 0.2, 0.98, 0.8))]))
        service._last_head_command_at = 0.0

    assert head_commands
    assert head_commands[0][0] == 104
    tracking_updates = head_commands[:-1]
    assert all(
        abs(current[0] - previous[0]) <= 18
        for previous, current in zip([(90, 90), *tracking_updates], tracking_updates)
    )
    assert head_commands[-1][0] == 90
    assert ("left", 170, 650) in move_commands


@pytest.mark.anyio
async def test_head_aims_at_face_height_instead_of_person_box_center(tmp_path):
    service, _, head_commands, _ = make_service(tmp_path)

    await service.process_result(result("one", [person((0.30, 0.10, 0.70, 0.90))]))

    assert head_commands == [(90, 79)]


@pytest.mark.anyio
async def test_body_pivots_when_head_nears_pan_limit_even_with_centered_person(tmp_path):
    service, _, _, move_commands = make_service(tmp_path)
    service.head_pan = 120
    service._last_head_command_at = float("inf")

    for index in range(2):
        await service.process_result(result(str(index), [person((0.35, 0.2, 0.65, 0.9))]))

    assert ("left", 170, 600) in move_commands


@pytest.mark.anyio
async def test_body_pivot_direction_reverses_at_opposite_pan_limit(tmp_path):
    service, _, _, move_commands = make_service(tmp_path)
    service.head_pan = 60
    service._last_head_command_at = float("inf")

    for index in range(2):
        await service.process_result(result(str(index), [person((0.35, 0.2, 0.65, 0.9))]))

    assert ("right", 170, 600) in move_commands


@pytest.mark.anyio
async def test_target_latch_prefers_matching_person_over_larger_new_person(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    service._last_head_command_at = float("inf")
    await service.process_result(result("one", [person((0.05, 0.15, 0.35, 0.85))]))

    matching_person = person((0.08, 0.15, 0.38, 0.85), confidence=0.72)
    larger_new_person = person((0.45, 0.05, 0.95, 0.95), confidence=0.98)
    await service.process_result(result("two", [larger_new_person, matching_person]))

    assert service.target is not None
    assert service.target.bounding_box == matching_person.bounding_box


def test_initial_target_prefers_upright_upper_body_over_wide_reclined_box(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    wide_full_body = person((0.03, 0.40, 0.62, 0.79), confidence=0.66)
    upright_upper_body = person((0.50, 0.41, 0.61, 0.68), confidence=0.46)

    selected = service._select_candidate([wide_full_body, upright_upper_body])

    assert selected == upright_upper_body


def test_initial_target_groups_person_boxes_and_ignores_separate_false_person(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    actual_upper_body = person((0.02, 0.50, 0.35, 0.89), confidence=0.66)
    actual_full_body = person((0.02, 0.50, 0.66, 0.99), confidence=0.57)
    separate_dark_object = person((0.59, 0.63, 0.82, 0.99), confidence=0.84)

    selected = service._select_candidate(
        [separate_dark_object, actual_full_body, actual_upper_body]
    )

    assert selected in (actual_upper_body, actual_full_body)


@pytest.mark.anyio
async def test_searching_recenters_head_once_without_authorizing_body_motion(tmp_path):
    service, _, head_commands, move_commands = make_service(tmp_path)
    service.head_pan = 135

    await service.process_result(result("missing-one", []))
    await service.process_result(result("missing-two", []))

    assert head_commands == [(90, 90)]
    assert service.head_pan == 90
    assert not move_commands


@pytest.mark.anyio
async def test_tracking_estimates_motion_and_leads_the_smoothed_aim_point(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    service._last_head_command_at = float("inf")
    first = result("one", [person((0.25, 0.10, 0.55, 0.90))])
    second = result("two", [person((0.35, 0.10, 0.65, 0.90))])
    second.captured_at = first.captured_at + timedelta(seconds=1 / 3)

    await service.process_result(first)
    await service.process_result(second)

    assert service._aim_velocity[0] > 0
    assert service._smoothed_center is not None
    assert service._smoothed_center[0] > 0.45


def test_tracking_preprocessing_applies_mild_contrast_boost():
    image = Image.new("RGB", (32, 16), (80, 80, 80))
    for x in range(16, 32):
        for y in range(16):
            image.putpixel((x, y), (160, 160, 160))
    source = BytesIO()
    image.save(source, format="JPEG", quality=100)

    processed = Image.open(BytesIO(rotate_jpeg(source.getvalue(), 0))).convert("RGB")

    assert processed.getpixel((4, 8))[0] < 80
    assert processed.getpixel((28, 8))[0] > 160


@pytest.mark.anyio
async def test_stop_and_enable_are_the_only_mode_transitions(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    assert service.mode == "track"
    assert service.state == "searching"

    stopped = await service.stop(reason="test")
    assert stopped.mode == "off"
    assert stopped.state == "off"
    assert stopped.stop_reason == "test"

    enabled = await service.enable()
    assert enabled.mode == "track"
    assert enabled.state == "searching"
    assert enabled.stop_reason is None


@pytest.mark.anyio
async def test_detection_finishing_after_stop_cannot_restart_tracking(tmp_path):
    service, _, head_commands, move_commands = make_service(tmp_path)
    await service.stop(reason="test")

    await service.process_result(result("late", [person((0.72, 0.2, 0.98, 0.8))]))

    assert service.mode == "off"
    assert service.state == "off"
    assert service.target is None
    assert not head_commands
    assert not move_commands


@pytest.mark.anyio
async def test_tracking_rate_lease_and_manual_suspension(tmp_path):
    service, broker, _, _ = make_service(tmp_path)
    assert service.mode == "track"
    assert broker.effective_fps == pytest.approx(3.0)

    service.suspend_for_manual_control()
    assert service.mode == "track"
    assert broker.effective_fps == pytest.approx(3.0)


@pytest.mark.anyio
async def test_stale_detection_never_moves_and_slow_cpu_disables_body_motion(tmp_path):
    service, broker, head_commands, move_commands = make_service(tmp_path)
    stale = result("stale", [person((0.72, 0.2, 0.98, 0.8))])
    stale.captured_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    await service.process_result(stale)

    assert service.target is None
    assert not head_commands
    assert not move_commands

    for index in range(3):
        slow_cpu = result(str(index), [person((0.72, 0.2, 0.98, 0.8))], latency_ms=300.0)
        slow_cpu.backend = "cpu/float32"
        await service.process_result(slow_cpu)
        service._last_head_command_at = 0.0

    assert head_commands
    assert not move_commands
    assert broker.effective_fps == pytest.approx(1.0)


@pytest.mark.anyio
async def test_missing_target_immediately_revokes_motion_authorization(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    await service.process_result(result("one", [person((0.35, 0.2, 0.65, 0.8))]))
    generation = service._motion_generation
    assert service.motion_authorized(generation, body=True)

    await service.process_result(result("two", []))

    assert service.state == "tracking"
    assert service.target is not None
    assert not service.motion_authorized(generation, body=True)


@pytest.mark.anyio
async def test_manual_or_emergency_change_invalidates_queued_motion(tmp_path):
    service, _, _, _ = make_service(tmp_path)
    await service.process_result(result("one", [person((0.35, 0.2, 0.65, 0.8))]))
    generation = service._motion_generation
    assert service.motion_authorized(generation, body=True)

    service.suspend_for_manual_control()
    assert not service.motion_authorized(generation, body=True)
    assert service.mode == "track"

    await service.enable()
    await service.process_result(result("two", [person((0.35, 0.2, 0.65, 0.8))]))
    generation = service._motion_generation
    assert service.motion_authorized(generation, body=True)

    await service.stop(reason="test stop")
    assert not service.motion_authorized(generation, body=True)

    await service.enable()
    await service.process_result(result("three", [person((0.35, 0.2, 0.65, 0.8))]))
    generation = service._motion_generation
    assert service.motion_authorized(generation, body=True)

    service.emergency_disable()
    assert not service.motion_authorized(generation, body=False)
    assert service.mode == "off"
    assert service.state == "off"


@pytest.mark.anyio
async def test_detector_client_rejects_mismatched_frame(monkeypatch):
    captured_at = datetime.now(timezone.utc)
    frame = CameraFrame("expected-frame", captured_at, b"jpeg")

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, content, headers):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "frame_id": "wrong-frame",
                    "captured_at": captured_at.isoformat(),
                    "model": "Roboflow/rf-detr-nano",
                    "backend": "cuda/bfloat16",
                    "latency_ms": 10.0,
                    "people": [],
                },
            )

    monkeypatch.setattr("app.tracking.httpx.AsyncClient", lambda **_kwargs: FakeAsyncClient())

    with pytest.raises(RuntimeError, match="different camera frame"):
        await RFDetrClient("http://tracking").detect(frame, frame.content, 0.55)


@pytest.mark.anyio
async def test_skipped_head_command_does_not_drift_internal_position(tmp_path):
    service, _, head_commands, _ = make_service(tmp_path, skip_head=True)

    await service.process_result(result("one", [person((0.72, 0.2, 0.98, 0.8))]))

    assert head_commands == [(104, 82)]
    assert (service.head_pan, service.head_tilt) == (90, 90)


@pytest.mark.anyio
async def test_completed_head_command_syncs_even_if_detection_ages_out_in_flight(tmp_path):
    service, _, _, _ = make_service(tmp_path)

    async def delayed_head(_pan, _tilt, _generation, _allow_search=False):
        assert service.target is not None
        service.target.observed_at -= timedelta(seconds=1)
        return {"ok": True, "pan": 94, "tilt": 88}

    service.head_command = delayed_head

    await service.process_result(result("one", [person((0.72, 0.2, 0.98, 0.8))]))

    assert (service.head_pan, service.head_tilt) == (94, 88)


@pytest.mark.anyio
async def test_manual_override_during_head_command_prevents_stale_position_update(tmp_path):
    service, _, _, _ = make_service(tmp_path)

    async def interrupted_head(_pan, _tilt, _generation, _allow_search=False):
        service.suspend_for_manual_control()
        return {"ok": True, "pan": 93, "tilt": 90}

    service.head_command = interrupted_head

    await service.process_result(result("one", [person((0.72, 0.2, 0.98, 0.8))]))

    assert (service.head_pan, service.head_tilt) == (90, 90)


@pytest.mark.anyio
async def test_completed_pivot_starts_cooldown_even_if_detection_ages_out_in_flight(tmp_path):
    service, _, _, move_commands = make_service(tmp_path)

    async def delayed_move(direction, speed, duration_ms, _generation):
        move_commands.append((direction, speed, duration_ms))
        assert service.target is not None
        service.target.observed_at -= timedelta(seconds=1)
        return {"ok": True}

    service.move_command = delayed_move
    service._last_head_command_at = float("inf")
    service._pivot_error_frames = 1

    await service.process_result(result("one", [person((0.72, 0.2, 0.98, 0.8))]))

    assert move_commands == [("left", 170, 434)]
    assert service._last_pivot_at > 0
    assert service._pivot_error_frames == 0

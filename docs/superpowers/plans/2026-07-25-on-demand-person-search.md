# On-Demand RF-DETR Person Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove YuNet and continuous tracking, replacing them with a ten-second RF-DETR-only command that turns right until the first person is visible.

**Architecture:** A compact `PersonSearchService` owns one explicit finite search operation and is idle otherwise. It positions the head once, alternates fresh RF-DETR checks with bounded right pivots, and terminates on the first person, timeout, cancellation, or fault. The tracking sidecar returns only RF-DETR person detections.

**Tech Stack:** Python 3.11, asyncio, FastAPI/Pydantic, HTTPX, RF-DETR, existing frame and actuator brokers, pytest/AnyIO.

## Global Constraints

- Search is explicit only; no idle/background detector loop remains.
- Search timeout is `10` seconds.
- Head-height pose is pan `90`, tilt `75`.
- Each miss authorizes one right pivot at speed `140` for `350` ms.
- Wait `500` ms after the bounded movement ends before capturing.
- The first RF-DETR person at or above configured confidence ends the search.
- Do not center, associate, estimate, predict, or follow a detected person.
- Manual head, drive, and stop cancel the active search.
- Stale results from a cancelled generation cannot issue actuator commands.
- Routine image bytes remain memory-only.
- Preserve the existing persistent ESP control channel and browser robot APIs.
- Stage only files named by the active task because the worktree contains prior reliability changes.

## File Structure

- `pc_tracking/app/main.py`: RF-DETR-only sidecar and person-only response schema.
- `pc_tracking/app/yunet.py`: delete.
- `pc_tracking/app/model_assets.py`: delete.
- `pc_tracking/requirements.txt`: remove OpenCV.
- `pc_tracking/tests/test_main.py`: RF-DETR-only API/backend tests.
- `pc_tracking/tests/test_yunet.py`: delete.
- `pc_tracking/tests/test_model_assets.py`: delete.
- `pc_brain/app/tracking.py`: replace temporal tracker with detector client and `PersonSearchService`.
- `pc_brain/app/config.py`: retain only search configuration.
- `pc_brain/app/main.py`: construct search service, route commands, cancel on manual actions, and expose search APIs/status.
- `pc_brain/tests/test_tracking.py`: focused person-search state, freshness, timeout, cancellation, and failure tests.
- `pc_brain/tests/test_tracking_temporal.py`: delete.
- `pc_brain/tests/test_config.py`: search defaults/validation and removed-mode regression.
- `pc_brain/tests/test_main_endpoints.py`: endpoint, deterministic text, and manual cancellation tests.
- `README.md`, `docs/architecture.md`, `pc_brain/README.md`, `pc_brain/.env.example`, `pc_tracking/README.md`: remove YuNet/continuous tracking documentation and describe the one-shot command.

---

### Task 1: RF-DETR-Only Tracking Sidecar

**Files:**
- Modify: `pc_tracking/app/main.py`
- Modify: `pc_tracking/requirements.txt`
- Modify: `pc_tracking/tests/test_main.py`
- Delete: `pc_tracking/app/yunet.py`
- Delete: `pc_tracking/app/model_assets.py`
- Delete: `pc_tracking/tests/test_yunet.py`
- Delete: `pc_tracking/tests/test_model_assets.py`

**Interfaces:**
- Produces: `RFDetrBackend.detect(jpeg: bytes, threshold: float) -> tuple[list[PersonDetection], float]`.
- Produces: `POST /detect` response with `frame_id`, `captured_at`, `model`, `backend`, `latency_ms`, `queue_ms`, `request_bytes`, and `people`.
- Removes: face fields, detector modes, YuNet model loading, and minimum-face-size header.

- [ ] **Step 1: Rewrite sidecar tests to require person-only behavior**

Add literal tests asserting:

```python
def test_health_is_rfdetr_only(monkeypatch):
    monkeypatch.setattr(main, "backend", FakeBackend())
    payload = TestClient(main.app).get("/health").json()
    assert payload["model"] == "Roboflow/rf-detr-nano"
    assert "face" not in payload
    assert "detector_mode" not in payload


def test_detect_returns_only_people(monkeypatch):
    monkeypatch.setattr(main, "backend", FakeBackend(
        people=[PersonDetection(confidence=0.8, bounding_box=(0.1, 0.1, 0.4, 0.9))]
    ))
    response = TestClient(main.app).post(
        "/detect",
        content=b"jpeg",
        headers={
            "X-Robit-Frame-Id": "frame-1",
            "X-Robit-Captured-At": "2026-07-25T00:00:00Z",
            "X-Robit-Threshold": "0.4",
        },
    )
    assert response.status_code == 200
    assert len(response.json()["people"]) == 1
    assert "faces" not in response.json()
```

Also retain device selection, unavailable backend, serialized inference, empty-body, normalized-box, and frame metadata tests.

- [ ] **Step 2: Run the sidecar tests to verify RED**

Run in PowerShell:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests\test_main.py -q
```

Expected: failures because health and detection responses still expose YuNet/mode fields.

- [ ] **Step 3: Collapse `pc_tracking/app/main.py` to RF-DETR only**

Delete `TrackingBackend`, `DetectionBatch`, `DetectorMode`, face imports, and face constants. Instantiate:

```python
backend = RFDetrBackend()
```

Define the response as:

```python
class DetectionResponse(BaseModel):
    frame_id: str
    captured_at: str
    model: str = MODEL_NAME
    backend: str
    latency_ms: float
    queue_ms: float = 0.0
    request_bytes: int = 0
    people: list[PersonDetection] = Field(default_factory=list)
```

Run `backend.detect(jpeg, x_robit_threshold)` in `asyncio.to_thread`, preserve the inference lock and metrics, and expose only RF-DETR status from `/health`.

- [ ] **Step 4: Delete YuNet code/tests and remove OpenCV**

Delete the four YuNet/model-asset files listed above. Remove:

```text
opencv-python-headless==4.12.0.88
```

from `pc_tracking/requirements.txt`.

- [ ] **Step 5: Verify the sidecar**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_tracking\.venv\Scripts\python.exe -m compileall -q pc_tracking\app pc_tracking\tests
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
```

Expected: compile exits zero and all remaining sidecar tests pass.

### Task 2: Bounded Person Search Service

**Files:**
- Replace: `pc_brain/app/tracking.py`
- Replace: `pc_brain/tests/test_tracking.py`
- Delete: `pc_brain/tests/test_tracking_temporal.py`

**Interfaces:**
- Produces: `PersonCandidate`, `DetectorResult`, and `RFDetrClient`.
- Produces: `PersonSearchStatus`.
- Produces: `PersonSearchService.search()`, `cancel(reason)`, `cancel_now(reason)`, `status()`, `motion_authorized(generation, body)`, `start()`, and `shutdown()`.
- Consumes: `FrameBroker.get_rotated_jpeg(degrees, force_fresh=True)`.
- Consumes callbacks:
  `head_command(pan, tilt, generation)`,
  `move_command(direction, speed, duration_ms, generation)`, and
  `stop_command(generation)`.

- [ ] **Step 1: Write the bounded-search tests**

Use a fake clock/frame broker/detector and command recorders. Cover these exact sequences:

```python
await service.search()
assert commands[:2] == [("head", 90, 75), ("stop",)]
assert detector.frames == ["fresh-after-head"]
assert service.status().state == "found"
```

For an initial miss then detection:

```python
assert commands == [
    ("head", 90, 75),
    ("move", "right", 140, 350),
    ("stop",),
]
assert detector.frames == ["after-head", "after-settling"]
```

Also assert edge-of-frame detection produces no head/body correction, timeout is bounded, cancellation invalidates a delayed detector result, duplicate callers share one search, manual cancellation stops future commands, idle performs no detection, and detector/camera/actuator errors yield `faulted` with stage and exception class.

- [ ] **Step 2: Run focused tests to verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests\test_tracking.py -q
```

Expected: imports and behavior fail because `PersonSearchService` does not exist.

- [ ] **Step 3: Replace temporal types with compact person-only models**

Keep normalized person box validation and detector frame/capture identity checks:

```python
class PersonCandidate(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: tuple[float, float, float, float]


class DetectorResult(BaseModel):
    frame_id: str
    captured_at: datetime
    model: str
    backend: str
    latency_ms: float
    queue_ms: float = 0.0
    people: list[PersonCandidate] = Field(default_factory=list)
```

`RFDetrClient.detect` sends only frame ID, capture time, and person threshold headers.

- [ ] **Step 4: Implement `PersonSearchService`**

Use states `idle`, `positioning`, `detecting`, `turning`, `found`, `timed_out`, `cancelled`, and `faulted`. `search()` creates or joins one task. `_run_search(generation)`:

```python
deadline = monotonic() + self.timeout_seconds
await self._authorized_head(90, 75, generation)
while self._authorized(generation):
    if monotonic() >= deadline:
        return await self._finish("timed_out", generation)
    frame, content = await self.broker.get_rotated_jpeg(
        self.rotate_degrees,
        force_fresh=True,
    )
    result = await self.detector.detect(frame, content, self.confidence)
    if result.people:
        self.last_detection_confidence = result.people[0].confidence
        return await self._finish("found", generation)
    await self._authorized_move("right", 140, 350, generation)
    await asyncio.sleep(0.35 + self.settling_seconds)
```

Before and after every await, verify the generation remains active. `_finish` sends ordinary stop before recording success/timeout. Cancellation increments generation, cancels the task, and prevents stale completion.

- [ ] **Step 5: Verify the service**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests\test_tracking.py -q
```

Expected: all bounded-search tests pass.

### Task 3: Commands, Configuration, and Manual Cancellation

**Files:**
- Modify: `pc_brain/app/config.py`
- Modify: `pc_brain/app/main.py`
- Modify: `pc_brain/tests/test_config.py`
- Modify: `pc_brain/tests/test_main_endpoints.py`

**Interfaces:**
- Consumes Task 2 `PersonSearchService`.
- Produces: `POST /tracking/look-at-me`, `GET /tracking/status`, and cancellation-only `POST /tracking/stop`.
- Produces deterministic text/voice commands `look at me` and `find me`.

- [ ] **Step 1: Write failing configuration and endpoint tests**

Assert exact defaults:

```python
assert settings.tracking_search_timeout_seconds == 10.0
assert settings.tracking_head_height_pan == 90
assert settings.tracking_head_height_tilt == 75
assert settings.tracking_search_pivot_speed == 140
assert settings.tracking_search_pivot_duration_ms == 350
assert settings.tracking_search_settling_seconds == 0.5
```

Endpoint tests require `/tracking/start` to return `404`, `/tracking/look-at-me` to await `search()`, `/tracking/stop` to call `cancel("API request")`, and manual drive/head/stop to call `cancel_now` before actuator execution. Text tests require “look at me” and “find me” to return deterministic found/timeout/failure wording.

- [ ] **Step 2: Run tests to verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests\test_config.py pc_brain\tests\test_main_endpoints.py -q
```

Expected: failures because old continuous settings/routes remain.

- [ ] **Step 3: Prune configuration and wire the service**

Remove detector mode, face, angular, cadence, association, and continuous pivot settings. Add validated search settings using existing integer/float helpers. Construct `PersonSearchService` with the exact global defaults and a stop callback that routes through `ActuatorBroker.stop()`.

Change tracking head publication to `wait_for_ack=True`. Use generation authorization for head, move, and stop callbacks.

- [ ] **Step 4: Replace deterministic commands and APIs**

Use:

```python
PERSON_SEARCH_MARKERS = re.compile(r"\b(?:look at me|find me)\b", re.IGNORECASE)
PERSON_SEARCH_CANCEL_MARKERS = re.compile(
    r"\b(?:stop looking|stop searching|cancel (?:the )?search)\b",
    re.IGNORECASE,
)
```

Remove continuous start/enable routing and its capability advertisement. Map terminal results to:

- found: `I found you.`
- timed out: `I couldn't find anyone.`
- cancelled: `I stopped looking.`
- faulted: `I couldn't look for you: <error>.`

- [ ] **Step 5: Verify configuration and endpoints**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests\test_config.py pc_brain\tests\test_main_endpoints.py -q
```

Expected: all selected tests pass.

### Task 4: Remove Stale Surface and Verify the Hard Cleanup

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `pc_brain/README.md`
- Modify: `pc_brain/.env.example`
- Modify: `pc_tracking/README.md`
- Test: `pc_brain/tests`
- Test: `pc_tracking/tests`

**Interfaces:**
- Documents PowerShell startup, explicit search, status, cancellation, and the person-only sidecar.

- [ ] **Step 1: Remove stale documentation and environment variables**

Document:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\Scripts\run.bat
Invoke-RestMethod -Method Post http://localhost:8080/tracking/look-at-me
Invoke-RestMethod http://localhost:8080/tracking/status |
    ConvertTo-Json -Depth 6
```

Remove all YuNet, face-only, shadow, continuous following, angular calibration, and `/tracking/start` references. Add the six search defaults to `.env.example`.

- [ ] **Step 2: Prove deleted concepts are absent**

Run:

```powershell
cd C:\Users\z1sou\HouseGremlin
rg -n "YuNet|yunet|face_only|shadow|FaceCandidate|gaze_calculation|TRACKING_DETECTOR_MODE|tracking/start" `
  pc_brain pc_tracking README.md docs\architecture.md
```

Expected: no matches outside historical design/plan documents and virtual environments.

- [ ] **Step 3: Run fresh full verification**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m compileall -q pc_brain\app pc_brain\tests
.\pc_tracking\.venv\Scripts\python.exe -m compileall -q pc_tracking\app pc_tracking\tests
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests -q
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
git diff --check
```

Expected: both compile commands exit zero, both full suites pass, and the whitespace check reports no errors.

- [ ] **Step 4: Run the attended PowerShell acceptance**

After stopping the existing supervisor with `Ctrl+C`:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\Scripts\run.bat
```

Say “look at me” from outside the initial camera view. Confirm Robit sets the head-height pose, turns only right in bounded steps, stops at the first person detection, performs no centering correction, remains idle afterward, and times out within ten seconds when no person is present.

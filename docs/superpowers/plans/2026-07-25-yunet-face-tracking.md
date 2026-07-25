# YuNet Face Tracking Implementation Plan

> **For implementation:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use subagent-driven development only when the user explicitly requests delegated execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a reversible YuNet face-tracking trial, prove it against RF-DETR in shadow mode, cut normal control to face-only gaze, and remove the RF-DETR runtime after physical acceptance.

**Architecture:** The existing `pc_tracking` sidecar gains a CPU YuNet backend and a mode-aware response that can expose face and person observations from the same shared frame during the trial. PC Brain selects `shadow`, `face_only`, or `person_only`; only the selected controller path can publish gaze. The accepted end state retains YuNet, the temporal controller, and the existing actuator broker while deleting RF-DETR-specific dependencies and startup behavior.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, OpenCV `FaceDetectorYN`, YuNet ONNX, pytest/AnyIO, existing PC Brain frame broker and actuator broker.

## Global Constraints

- Robot camera acquisition remains globally capped at 2 FPS.
- Routine camera frames remain memory-only and are never journaled.
- YuNet runs on CPU and must not allocate GPU memory.
- Face tracking uses no recognition model, embedding, identity database, or biometric persistence.
- A face target requires two consistent detections within two seconds.
- Switching faces requires three consistently superior detections.
- Missing-face grace remains 2.5 seconds; prediction is capped at 0.5 seconds.
- Five seconds without a face issues one neutral target and never starts body search movement.
- Tracking head commands remain brokered at no more than 4 Hz.
- Frames captured during pivot or settling cannot seed the next gaze estimate.
- Browser-facing `/tracking/*` endpoints remain compatible.
- The trial is reversible with `ROBIT_TRACKING_DETECTOR_MODE=person_only`.
- RF-DETR removal occurs only after the shadow, ten-minute, and thirty-minute acceptance gates pass.
- Before implementation, preserve the current reliability-redesign work as its own
  reviewed baseline. Every task commit below must stage only its listed files so
  pre-existing user changes cannot be absorbed accidentally.

## File Structure

- `pc_tracking/app/model_assets.py`: deterministic YuNet model download and checksum validation.
- `pc_tracking/app/yunet.py`: synchronous CPU-only YuNet wrapper and normalized face result conversion.
- `pc_tracking/app/main.py`: mode-aware detector startup, `/detect` response, timing, and health.
- `pc_tracking/tests/test_model_assets.py`: checksum and atomic model-install tests.
- `pc_tracking/tests/test_yunet.py`: YuNet output conversion and invalid-result tests.
- `pc_tracking/tests/test_main.py`: sidecar mode, response, and failure contract tests.
- `pc_brain/app/config.py`: detector mode and face-specific thresholds.
- `pc_brain/app/tracking.py`: face data models, client parsing, association, gaze, loss behavior, and shadow metrics.
- `pc_brain/app/main.py`: configuration wiring and public health/status exposure.
- `pc_brain/tests/test_config.py`: mode and threshold defaults.
- `pc_brain/tests/test_tracking.py`: face aiming, association, grace, neutral, and pivot behavior.
- `pc_brain/tests/test_tracking_temporal.py`: face replay sequences and shadow isolation.
- `pc_brain/tests/test_main_endpoints.py`: public compatibility and new status fields.
- `Scripts/setup.bat`: install pinned OpenCV wheel and verified YuNet asset.
- `Scripts/supervisor.py`: pass detector mode and report the correct tracking backend.
- `pc_brain/.env.example`, `pc_brain/README.md`, `README.md`, `docs/architecture.md`: trial configuration and operational documentation.

## Shared Test Fixtures

Create fixtures in the test module where they are first used; move them to that
package's `conftest.py` only when a second module needs them.

- `jpeg_bytes(width: int = 64, height: int = 64) -> bytes`: returns an in-memory
  valid JPEG with the requested dimensions.
- `FakeDetector(rows: np.ndarray | None)`: implements `setInputSize`,
  `setScoreThreshold`, and `detect`, returning the supplied YuNet rows.
- `reload_main()`: removes `app.main` from `sys.modules`, imports it again after
  environment monkeypatching, and returns the module.
- `make_face_service(tmp_path, mode)`: builds the tracking service with an
  in-memory detector result source and recording head/move callbacks; returns the
  service followed by those command lists.
- `face_candidate(...)`, `centered_face()`, `face_result(...)`,
  `combined_result(...)`, and `person(...)`: construct literal normalized model
  objects; they must not bypass model validation.
- `acquire_face(service, face)`: submits exactly the configured two consistent
  face detections with distinct frame IDs and returns only after acquisition.

---

### Task 1: Verified YuNet asset and dependency

**Files:**
- Create: `pc_tracking/app/model_assets.py`
- Create: `pc_tracking/tests/test_model_assets.py`
- Modify: `pc_tracking/requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `ensure_yunet_model(model_path: Path, download_url: str = YUNET_URL) -> Path`
- Produces: `YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"`
- Consumes: only Python standard-library `hashlib`, `os`, `pathlib`, `tempfile`, and `urllib.request`.

- [ ] **Step 1: Write the failing asset tests**

```python
def test_existing_yunet_model_with_expected_hash_is_reused(tmp_path, monkeypatch):
    content = b"known-model"
    monkeypatch.setattr(model_assets, "YUNET_SHA256", sha256(content).hexdigest())
    path = tmp_path / "yunet.onnx"
    path.write_bytes(content)
    assert model_assets.ensure_yunet_model(path, "https://unused.invalid") == path


def test_bad_download_is_deleted_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(model_assets, "YUNET_SHA256", sha256(b"expected").hexdigest())
    monkeypatch.setattr(model_assets, "_download", lambda _url, path: path.write_bytes(b"bad"))
    path = tmp_path / "yunet.onnx"
    with pytest.raises(RuntimeError, match="checksum"):
        model_assets.ensure_yunet_model(path, "https://model.invalid/yunet.onnx")
    assert not path.exists()
    assert not path.with_suffix(".onnx.part").exists()
```

- [ ] **Step 2: Run the asset tests and verify RED**

Run in PowerShell:

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_tracking
.\.venv\Scripts\python.exe -m pytest tests\test_model_assets.py -q
```

Expected: collection fails because `app.model_assets` does not exist.

- [ ] **Step 3: Implement atomic download and checksum validation**

```python
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def ensure_yunet_model(model_path: Path, download_url: str = YUNET_URL) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and _digest(model_path) == YUNET_SHA256:
        return model_path
    partial = model_path.with_suffix(model_path.suffix + ".part")
    partial.unlink(missing_ok=True)
    _download(download_url, partial)
    if _digest(partial) != YUNET_SHA256:
        partial.unlink(missing_ok=True)
        model_path.unlink(missing_ok=True)
        raise RuntimeError("YuNet model checksum mismatch")
    os.replace(partial, model_path)
    return model_path
```

Pin `opencv-python-headless==4.12.0.88` in `pc_tracking/requirements.txt` and ignore `pc_tracking/models/*.onnx`.

- [ ] **Step 4: Run the asset tests and complete sidecar tests**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_tracking
.\.venv\Scripts\python.exe -m pytest tests\test_model_assets.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: both commands pass.

- [ ] **Step 5: Commit Task 1**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add .gitignore pc_tracking\requirements.txt pc_tracking\app\model_assets.py pc_tracking\tests\test_model_assets.py
git commit -m "build: add verified YuNet model asset"
```

### Task 2: CPU YuNet backend and sidecar contract

**Files:**
- Create: `pc_tracking/app/yunet.py`
- Create: `pc_tracking/tests/test_yunet.py`
- Modify: `pc_tracking/app/main.py`
- Modify: `pc_tracking/tests/test_main.py`

**Interfaces:**
- Produces: `FaceDetection` with normalized `bounding_box`, `right_eye`, `left_eye`, `nose`, `right_mouth`, and `left_mouth`.
- Produces: `YuNetBackend.load() -> None` and `YuNetBackend.detect(jpeg: bytes, threshold: float) -> tuple[list[FaceDetection], float]`.
- Produces: `/detect` response fields `faces`, `people`, `detector_mode`, `face_latency_ms`, and existing RF timing.
- Consumes: `ensure_yunet_model`.

- [ ] **Step 1: Write failing YuNet conversion tests**

```python
def test_yunet_normalizes_box_and_five_landmarks(monkeypatch, jpeg_bytes):
    fake = np.array([[32, 24, 64, 80, 48, 44, 72, 44, 60, 58, 50, 78, 70, 78, 0.93]])
    backend = YuNetBackend(model_path=Path("unused.onnx"), detector=FakeDetector(fake))
    faces, _ = backend.detect(jpeg_bytes(width=160, height=120), 0.5)
    assert faces[0].bounding_box == pytest.approx((0.2, 0.2, 0.6, 0.8666667))
    assert faces[0].right_eye == pytest.approx((0.3, 0.3666667))
    assert faces[0].left_eye == pytest.approx((0.45, 0.3666667))


def test_yunet_rejects_non_finite_or_inverted_face_rows(jpeg_bytes):
    rows = np.array([[20, 20, -5, 30, *([10] * 10), float("nan")]])
    backend = YuNetBackend(model_path=Path("unused.onnx"), detector=FakeDetector(rows))
    faces, _ = backend.detect(jpeg_bytes(), 0.5)
    assert faces == []
```

- [ ] **Step 2: Run YuNet tests and verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_tracking
.\.venv\Scripts\python.exe -m pytest tests\test_yunet.py -q
```

Expected: import fails because `app.yunet` does not exist.

- [ ] **Step 3: Implement the CPU-only backend**

```python
detector = cv2.FaceDetectorYN.create(
    str(model_path),
    "",
    (320, 320),
    score_threshold=0.5,
    nms_threshold=0.3,
    top_k=32,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU,
)


def detect(self, jpeg: bytes, threshold: float) -> tuple[list[FaceDetection], float]:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG could not be decoded")
    height, width = image.shape[:2]
    self.detector.setInputSize((width, height))
    self.detector.setScoreThreshold(threshold)
    started = perf_counter()
    _, rows = self.detector.detect(image)
    latency_ms = (perf_counter() - started) * 1000.0
    return normalize_faces(rows, width, height), latency_ms
```

Never select an OpenCV CUDA backend or target.

- [ ] **Step 4: Write failing sidecar mode tests**

```python
@pytest.mark.anyio
async def test_face_only_runs_yunet_without_loading_rfdetr(monkeypatch):
    monkeypatch.setenv("ROBIT_TRACKING_DETECTOR_MODE", "face_only")
    app = reload_main()
    assert app.backend.person_backend is None
    assert app.backend.face_backend is not None


def test_detect_response_preserves_frame_identity_and_exposes_faces(client):
    response = client.post(
        "/detect",
        content=b"jpeg",
        headers={
            "X-Robit-Frame-Id": "frame-7",
            "X-Robit-Captured-At": "2026-07-25T00:00:00Z",
            "X-Robit-Threshold": "0.5",
        },
    )
    payload = response.json()
    assert payload["frame_id"] == "frame-7"
    assert payload["detector_mode"] == "face_only"
    assert payload["people"] == []
    assert payload["faces"][0]["right_eye"] == [0.3, 0.36]
```

- [ ] **Step 5: Implement mode-aware sidecar startup and response**

Use the exact modes `shadow`, `face_only`, and `person_only`. Load YuNet in
`shadow` and `face_only`; load RF-DETR in `shadow` and `person_only`. In shadow
mode, run both backends sequentially under the existing inference lock against
the same JPEG. Return partial results if one shadow backend fails, but return
HTTP 503 if the active backend for `face_only` or `person_only` is unavailable.

- [ ] **Step 6: Run sidecar tests**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_tracking
.\.venv\Scripts\python.exe -m pytest tests -q
```

Expected: all sidecar tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add pc_tracking\app\yunet.py pc_tracking\app\main.py pc_tracking\tests\test_yunet.py pc_tracking\tests\test_main.py
git commit -m "feat: add YuNet face detection backend"
```

### Task 3: PC Brain face contract and configuration

**Files:**
- Modify: `pc_brain/app/config.py`
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/app/main.py`
- Modify: `pc_brain/tests/test_config.py`
- Modify: `pc_brain/tests/test_tracking.py`

**Interfaces:**
- Produces: `TrackingDetectorMode = Literal["shadow", "face_only", "person_only"]`.
- Produces: `FaceCandidate`, `FaceObservation`, and face fields on `DetectorResult`.
- Produces settings: `tracking_detector_mode`, `tracking_face_confidence`, and `tracking_min_face_pixels`.
- Consumes the Task 2 `/detect` response.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_face_tracking_defaults(monkeypatch):
    monkeypatch.delenv("ROBIT_TRACKING_DETECTOR_MODE", raising=False)
    settings = load_settings()
    assert settings.tracking_detector_mode == "shadow"
    assert settings.tracking_face_confidence == pytest.approx(0.60)
    assert settings.tracking_min_face_pixels == 12


def test_invalid_tracking_detector_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("ROBIT_TRACKING_DETECTOR_MODE", "both")
    with pytest.raises(ValueError, match="shadow, face_only, or person_only"):
        load_settings()
```

- [ ] **Step 2: Run configuration tests and verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_config.py -q
```

Expected: settings fields do not exist.

- [ ] **Step 3: Add validated settings**

```python
def _tracking_mode_env() -> str:
    mode = os.getenv("ROBIT_TRACKING_DETECTOR_MODE", "shadow").strip().lower()
    if mode not in {"shadow", "face_only", "person_only"}:
        raise ValueError(
            "ROBIT_TRACKING_DETECTOR_MODE must be shadow, face_only, or person_only"
        )
    return mode
```

Set face confidence to `0.60` and minimum face size to `12` pixels.

- [ ] **Step 4: Write failing response-parsing tests**

```python
def test_detector_result_accepts_normalized_face_landmarks():
    result = DetectorResult.model_validate(
        {
            "frame_id": "f1",
            "captured_at": "2026-07-25T00:00:00Z",
            "model": "opencv/yunet",
            "backend": "opencv/cpu",
            "latency_ms": 3.0,
            "detector_mode": "face_only",
            "faces": [{
                "confidence": 0.9,
                "bounding_box": [0.2, 0.2, 0.6, 0.7],
                "right_eye": [0.3, 0.35],
                "left_eye": [0.5, 0.35],
                "nose": [0.4, 0.45],
                "right_mouth": [0.33, 0.55],
                "left_mouth": [0.47, 0.55],
            }],
            "people": [],
        }
    )
    assert result.faces[0].eye_midpoint == pytest.approx((0.4, 0.35))
```

- [ ] **Step 5: Add face models and client validation**

`FaceCandidate.eye_midpoint` must be derived from the two literal eye points,
not from the face box. Validate every normalized coordinate in `[0, 1]`, reject
inverted boxes, reject non-finite numbers, and preserve the existing frame ID
and capture timestamp mismatch checks.

- [ ] **Step 6: Wire settings into `get_tracking_service` and run tests**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_tracking.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add pc_brain\app\config.py pc_brain\app\tracking.py pc_brain\app\main.py pc_brain\tests\test_config.py pc_brain\tests\test_tracking.py
git commit -m "feat: add face detector contract and modes"
```

### Task 4: Face association, gaze, and calm loss behavior

**Files:**
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/tests/test_tracking.py`
- Modify: `pc_brain/tests/test_tracking_temporal.py`

**Interfaces:**
- Produces: face-first state flow `face_searching`, `face_acquiring`, `face_tracking`, `face_grace`, `face_lost`.
- Produces: `target_source` values `face` and `none`.
- Consumes: `FaceCandidate.eye_midpoint`, existing alpha-beta estimator, actuator broker callbacks, and detector mode.

- [ ] **Step 1: Write failing face-gaze tests**

```python
@pytest.mark.anyio
async def test_face_tracking_aims_between_eyes_not_at_box_center(tmp_path):
    service, head_commands = make_face_service(tmp_path, mode="face_only")
    face = face_candidate(
        box=(0.20, 0.10, 0.80, 0.90),
        right_eye=(0.38, 0.32),
        left_eye=(0.46, 0.32),
    )
    await acquire_face(service, face)
    assert head_commands[-1] == (90, 83)


@pytest.mark.anyio
async def test_face_loss_holds_then_neutralizes_once_without_body_search(tmp_path):
    service, head_commands, move_commands = make_face_service(tmp_path, mode="face_only")
    service.missing_grace_seconds = 0.02
    service.lost_neutral_seconds = 0.04
    await acquire_face(service, centered_face())
    await asyncio.sleep(0.03)
    await service.process_result(face_result("missing-1", []))
    assert service.state == "face_lost"
    await asyncio.sleep(0.03)
    await service.process_result(face_result("missing-2", []))
    await service.process_result(face_result("missing-3", []))
    assert head_commands.count((90, 90)) == 1
    assert move_commands == []
```

- [ ] **Step 2: Run face behavior tests and verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_tracking.py -k "face_tracking or face_loss" -q
```

Expected: failures show person-box aim or missing face states.

- [ ] **Step 3: Implement face-first selection and association**

Use eye midpoint as the measurement. Preserve current IoU/distance latch logic,
but score candidates using face confidence, predicted midpoint distance, box
scale consistency, and center proximity. The active face's track ID remains
stable unless three consecutive frames confirm a superior face.

- [ ] **Step 4: Implement grace, loss, and neutral behavior**

In `face_only`, an absent face must never call `move_command`. Preserve the last
gaze through the 2.5-second grace, discard the estimator afterward, and issue
exactly one `(90, 90)` target after five seconds. Fresh acquisition requires
two frames and resets neutral state.

- [ ] **Step 5: Write and run temporal replay tests**

Add literal sequences for stationary jitter, profile disappearance, occlusion,
two-face crossing, pivot settling, and manual override. Assert track ID,
state, target source, maximum head command count, and zero body commands during
face loss.

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_tracking.py tests\test_tracking_temporal.py -q
```

Expected: all tracking tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add pc_brain\app\tracking.py pc_brain\tests\test_tracking.py pc_brain\tests\test_tracking_temporal.py
git commit -m "feat: track facial gaze with calm loss behavior"
```

### Task 5: Shadow isolation and measurable trial status

**Files:**
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/app/main.py`
- Modify: `pc_brain/tests/test_tracking_temporal.py`
- Modify: `pc_brain/tests/test_main_endpoints.py`

**Interfaces:**
- Produces `/tracking/status` fields `detector_mode`, `target_source`, `face_box`, `face_landmarks`, `face_confidence`, `face_latency_ms`, `face_queue_ms`, and `shadow`.
- Produces bounded 1 Hz `tracking.sample` face geometry without image bytes.
- Guarantees shadow face observations cannot call head or move callbacks.

- [ ] **Step 1: Write failing shadow-isolation test**

```python
@pytest.mark.anyio
async def test_shadow_face_measurements_never_control_actuators(tmp_path):
    service, head_commands, move_commands = make_face_service(tmp_path, mode="shadow")
    result = combined_result(
        people=[person((0.45, 0.1, 0.75, 0.9))],
        faces=[face_candidate(eyes=((0.1, 0.3), (0.2, 0.3)))],
    )
    await service.process_result(result)
    assert service.status().shadow["face_detected"] is True
    assert all(command[0] > 90 for command in head_commands)
    assert not any(command[0] < 90 for command in head_commands)
    assert move_commands == []
```

- [ ] **Step 2: Run test and verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_tracking_temporal.py -k shadow -q
```

Expected: shadow metrics do not exist or face data affects the controller.

- [ ] **Step 3: Add passive shadow accumulator**

Track counts and rolling bounded samples for face detections, confidence,
eye-midpoint jitter, face latency, RF agreement, and failures. Store only
normalized numeric data and frame IDs. Cap rolling samples at 500 entries.
Only the active mode's observation is passed to `_update_head_and_body`.

- [ ] **Step 4: Add public status and event tests**

Assert `/tracking/status` remains backward compatible and returns the new
fields. Assert `tracking.sample` contains no keys named `image`, `jpeg`,
`bytes`, or `content`.

- [ ] **Step 5: Run endpoint and tracking suites**

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\python.exe -m pytest tests\test_main_endpoints.py tests\test_tracking.py tests\test_tracking_temporal.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 5**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add pc_brain\app\tracking.py pc_brain\app\main.py pc_brain\tests\test_tracking_temporal.py pc_brain\tests\test_main_endpoints.py
git commit -m "feat: expose isolated face tracking shadow metrics"
```

### Task 6: Setup, supervision, documentation, and trial verification

**Files:**
- Modify: `Scripts/setup.bat`
- Modify: `Scripts/supervisor.py`
- Modify: `pc_brain/.env.example`
- Modify: `pc_brain/README.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `pc_tracking/README.md`
- Test: `pc_brain/tests`
- Test: `pc_tracking/tests`

**Interfaces:**
- Consumes: Task 1 model installer and Tasks 2–5 mode contract.
- Produces: a runnable `shadow` default and documented PowerShell cutover to `face_only`.

- [ ] **Step 1: Update setup and supervisor behavior**

`Scripts/setup.bat` must install `pc_tracking/requirements.txt`, change into
`pc_tracking`, and call this batch command:

```bat
".venv\Scripts\python.exe" -c "from pathlib import Path; from app.model_assets import ensure_yunet_model; ensure_yunet_model(Path('models/face_detection_yunet_2023mar.onnx'))"
```

from `pc_tracking`, then validate `cv2.__version__ == "4.12.0"`. The supervisor
must pass `ROBIT_TRACKING_DETECTOR_MODE` to both the sidecar and PC Brain and
must display the selected mode during readiness.

- [ ] **Step 2: Document exact trial commands**

Document shadow mode:

```powershell
cd C:\Users\z1sou\HouseGremlin
$env:ROBIT_TRACKING_DETECTOR_MODE = "shadow"
.\Scripts\run.bat
```

Document face-only mode:

```powershell
cd C:\Users\z1sou\HouseGremlin
$env:ROBIT_TRACKING_DETECTOR_MODE = "face_only"
.\Scripts\run.bat
```

Document rollback:

```powershell
cd C:\Users\z1sou\HouseGremlin
$env:ROBIT_TRACKING_DETECTOR_MODE = "person_only"
.\Scripts\run.bat
```

Explain that each block is run in a new PowerShell window after stopping the
previous supervisor with `Ctrl+C`.

- [ ] **Step 3: Run complete automated verification**

```powershell
cd C:\Users\z1sou\HouseGremlin
python -m pytest pc_brain\tests -q
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
git diff --check
```

Expected: both suites pass, no test failures, and `git diff --check` reports no
whitespace errors.

- [ ] **Step 4: Run the shadow gate**

Start `shadow`, collect 300–500 `tracking.sample` events across the scenarios
in the design, and verify:

- visible-face retention at least 95%
- false-positive frames below 1%
- stationary eye-midpoint jitter below 0.03 p95
- face inference below 20 ms p95
- zero actuator commands attributable to shadow face results

Do not proceed to `face_only` if any threshold fails.

- [ ] **Step 5: Run attended face-only acceptance**

Use face-only mode for the ten-minute stationary test and thirty-minute
voice-plus-tracking test. Verify no centered-face pivot, no duplicate
post-pivot turn, no more than four tracking head commands per second, and no
camera/control failure.

- [ ] **Step 6: Commit Task 6**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add Scripts\setup.bat Scripts\supervisor.py pc_brain\.env.example pc_brain\README.md pc_tracking\README.md README.md docs\architecture.md
git commit -m "docs: add YuNet tracking trial workflow"
```

### Task 7: Post-acceptance RF-DETR removal

**Gate:** Execute this task only after Task 6's shadow and physical acceptance
criteria have passed and the accepted test evidence has been recorded.

**Files:**
- Modify: `pc_tracking/requirements.txt`
- Modify: `pc_tracking/app/main.py`
- Delete: RF-DETR backend code from `pc_tracking/app/main.py`
- Modify: `Scripts/setup.bat`
- Modify: `Scripts/supervisor.py`
- Modify: `pc_brain/app/config.py`
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/.env.example`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: affected tests under `pc_brain/tests` and `pc_tracking/tests`

**Interfaces:**
- Produces: one YuNet-only tracking sidecar and one `face_only` controller path.
- Removes: `shadow`, `person_only`, RF-DETR/PyTorch/CUDA detector startup, and person-only response fields.

- [ ] **Step 1: Write failing end-state tests**

```python
def test_tracking_settings_have_no_rfdetr_modes(monkeypatch):
    monkeypatch.delenv("ROBIT_TRACKING_DETECTOR_MODE", raising=False)
    settings = load_settings()
    assert settings.tracking_detector_mode == "face_only"
    assert "person_only" not in get_args(TrackingDetectorMode)
    assert "shadow" not in get_args(TrackingDetectorMode)


def test_sidecar_health_reports_yunet_only(client):
    health = client.get("/health").json()
    assert health["model"] == "opencv/yunet/2023mar"
    assert "rfdetr" not in json.dumps(health).lower()
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin
python -m pytest pc_brain\tests\test_config.py -q
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
```

Expected: old modes and RF fields remain.

- [ ] **Step 3: Remove RF-DETR runtime and compatibility branches**

Remove `rfdetr`, detector Torch, and `torchvision` from the tracking
requirements and setup path. Retain neither dormant imports nor a disabled
backend class. Simplify `/detect` to return faces only, default PC Brain to
`face_only`, and remove shadow/person compatibility status.

- [ ] **Step 4: Run complete end-state verification**

```powershell
cd C:\Users\z1sou\HouseGremlin
python -m pytest pc_brain\tests -q
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 5: Recreate the tracking environment to prove dependency removal**

In PowerShell, stop Robit with `Ctrl+C`, then run:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\Scripts\setup.bat
.\Scripts\run.bat
```

Confirm `/health` reports YuNet CPU, no RF-DETR process starts, and the tracking
environment imports neither `rfdetr` nor detector-specific Torch.

- [ ] **Step 6: Commit Task 7**

```powershell
cd C:\Users\z1sou\HouseGremlin
git add pc_tracking\requirements.txt pc_tracking\app\main.py pc_tracking\tests\test_main.py pc_brain\app\config.py pc_brain\app\tracking.py pc_brain\tests\test_config.py pc_brain\tests\test_tracking.py pc_brain\tests\test_tracking_temporal.py Scripts\setup.bat Scripts\supervisor.py pc_brain\.env.example README.md docs\architecture.md
git commit -m "refactor: remove RF-DETR tracking runtime"
```

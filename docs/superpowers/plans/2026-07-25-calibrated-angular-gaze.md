# Calibrated Angular Gaze Implementation Plan

> **For implementation:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> task-by-task. Use delegated execution only when the user explicitly requests
> subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Center a freshly acquired face with one calculated absolute head
target, followed by at most one residual correction, while firmware preserves
fluid acceleration-limited motion.

**Architecture:** Add a pure pinhole-camera angle helper and validated
calibration settings to the existing tracker. Extend the persistent control
channel with telemetry receive age, then make face-only gaze originate from
fresh actual servo telemetry or the last acknowledged fallback position.
Existing association, broker coalescing, pivot settling, and loss behavior
remain unchanged.

**Tech Stack:** Python 3.11, asyncio, FastAPI/Pydantic, existing NDJSON control
channel, pytest/AnyIO.

## Global Constraints

- Horizontal FOV defaults to `62.0` degrees and vertical FOV to `48.5`.
- Pan and tilt angle gains default to `1.0`.
- Angular deadband defaults to `2.0` degrees.
- Telemetry is fresh only while connected and no more than one second old.
- Final targets remain clamped to pan `55..135` and tilt `35..115`.
- Initial face acquisition uses fresh eye geometry, not a smoothed center.
- No gaze update is allowed during pivot settling.
- Tracking publication remains capped by the actuator broker at 4 Hz.
- No image bytes are logged or journaled.
- Stage only files listed by each task; preserve the dirty reliability work.

## File Structure

- `pc_brain/app/config.py`: validated camera and servo calibration settings.
- `pc_brain/app/control_channel.py`: telemetry receive time and age-aware head
  state.
- `pc_brain/app/tracking.py`: pure angle calculation, absolute target
  calculation, controller integration, and diagnostics.
- `pc_brain/app/main.py`: passes settings and the age-aware telemetry provider.
- `pc_brain/tests/test_config.py`: invalid/default calibration tests.
- `pc_brain/tests/test_control_channel.py`: hello/telemetry age tests.
- `pc_brain/tests/test_tracking.py`: geometry, origin, clamping, deadband, and
  initial-command tests.
- `pc_brain/tests/test_tracking_temporal.py`: residual and settling regression.
- `pc_brain/.env.example`, `pc_brain/README.md`: calibration documentation.

---

### Task 1: Pure Camera Geometry and Validated Calibration

**Files:**
- Modify: `pc_brain/app/config.py`
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/tests/test_config.py`
- Modify: `pc_brain/tests/test_tracking.py`

**Interfaces:**
- Produces:
  `camera_angle_degrees(image_error: float, fov_degrees: float) -> float`.
- Produces settings `tracking_camera_horizontal_fov_degrees`,
  `tracking_camera_vertical_fov_degrees`, `tracking_pan_angle_gain`,
  `tracking_tilt_angle_gain`, and `tracking_angular_deadband_degrees`.

- [ ] **Step 1: Write failing geometry and configuration tests**

```python
def test_pinhole_camera_angle_matches_edge_geometry():
    assert camera_angle_degrees(0.40, 62.0) == pytest.approx(25.68, abs=0.05)
    assert camera_angle_degrees(-0.40, 62.0) == pytest.approx(-25.68, abs=0.05)
    assert camera_angle_degrees(0.0, 62.0) == 0.0


def test_angular_tracking_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBIT_DATA_DIR", str(tmp_path))
    for name in (
        "ROBIT_TRACKING_CAMERA_HORIZONTAL_FOV_DEGREES",
        "ROBIT_TRACKING_CAMERA_VERTICAL_FOV_DEGREES",
        "ROBIT_TRACKING_PAN_ANGLE_GAIN",
        "ROBIT_TRACKING_TILT_ANGLE_GAIN",
        "ROBIT_TRACKING_ANGULAR_DEADBAND_DEGREES",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.tracking_camera_horizontal_fov_degrees == 62.0
    assert settings.tracking_camera_vertical_fov_degrees == 48.5
    assert settings.tracking_pan_angle_gain == 1.0
    assert settings.tracking_tilt_angle_gain == 1.0
    assert settings.tracking_angular_deadband_degrees == 2.0
```

Parameterize invalid FOV values `0`, `180`, and `nan`; invalid gain values `0`,
`-1`, and `nan`; and invalid deadband values `-1` and `nan`.

- [ ] **Step 2: Verify RED**

Run in PowerShell:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_config.py `
  pc_brain\tests\test_tracking.py `
  -k "angular or pinhole" -q
```

Expected: imports/settings fail because the helper and settings do not exist.

- [ ] **Step 3: Implement the pure helper and settings validation**

```python
def camera_angle_degrees(image_error: float, fov_degrees: float) -> float:
    half_fov = radians(fov_degrees) / 2.0
    return degrees(atan(2.0 * image_error * tan(half_fov)))
```

Add a finite range validator in `config.py`:

```python
def _finite_float_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    value = _float_env(name, default)
    valid_minimum = value >= minimum if minimum_inclusive else value > minimum
    if (
        not math.isfinite(value)
        or not valid_minimum
        or (maximum is not None and value >= maximum)
    ):
        raise ValueError(f"{name} has an invalid value")
    return value
```

Use it for the five exact defaults in the global constraints.

- [ ] **Step 4: Verify GREEN**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_config.py `
  pc_brain\tests\test_tracking.py `
  -k "angular or pinhole" -q
```

Expected: selected tests pass.

### Task 2: Fresh Actual Head Telemetry

**Files:**
- Modify: `pc_brain/app/control_channel.py`
- Modify: `pc_brain/app/main.py`
- Modify: `pc_brain/tests/test_control_channel.py`

**Interfaces:**
- Produces: `EspControlChannel.head_state() -> dict[str, Any]`.
- Returned metadata keys:
  `control_connected: bool`, `telemetry_age_seconds: float | None`,
  plus the latest telemetry fields.
- Consumes: `time.monotonic()`.

- [ ] **Step 1: Write failing telemetry freshness test**

```python
@pytest.mark.anyio
async def test_head_state_reports_connection_and_telemetry_age(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(control_channel.time, "monotonic", lambda: clock[0])
    channel = EspControlChannel("127.0.0.1")
    channel.telemetry = {"pan_actual": 91, "tilt_actual": 88}
    channel._telemetry_received_at = clock[0]
    channel._ready.set()

    assert channel.head_state()["control_connected"] is True
    assert channel.head_state()["telemetry_age_seconds"] == 0.0
    clock[0] += 1.2
    assert channel.head_state()["telemetry_age_seconds"] == pytest.approx(1.2)
```

Also assert both hello state and later telemetry update
`_telemetry_received_at`.

- [ ] **Step 2: Verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_control_channel.py `
  -k "head_state or telemetry_age" -q
```

Expected: `head_state` or `_telemetry_received_at` is missing.

- [ ] **Step 3: Implement age-aware state**

Initialize `_telemetry_received_at = 0.0`, update it whenever hello state or
`telemetry` is accepted, and add:

```python
def head_state(self) -> dict[str, Any]:
    received_at = self._telemetry_received_at
    return {
        **self.telemetry,
        "control_connected": self._ready.is_set(),
        "telemetry_age_seconds": (
            max(0.0, time.monotonic() - received_at)
            if received_at > 0.0
            else None
        ),
    }
```

Change `get_tracking_service` to provide
`control_channel.head_state()` when a channel exists.

- [ ] **Step 4: Verify GREEN**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_control_channel.py -q
```

Expected: all control-channel tests pass.

### Task 3: Absolute Face Target and Diagnostics

**Files:**
- Modify: `pc_brain/app/tracking.py`
- Modify: `pc_brain/app/main.py`
- Modify: `pc_brain/tests/test_tracking.py`
- Modify: `pc_brain/tests/test_tracking_temporal.py`
- Modify: `pc_brain/tests/test_main_endpoints.py`

**Interfaces:**
- Produces:
  `HeadTargetCalculation(origin_pan, origin_tilt, target_pan, target_tilt,
  horizontal_error_degrees, vertical_error_degrees, origin_source)`.
- Produces:
  `calculate_face_head_target(...) -> HeadTargetCalculation`.
- Adds `gaze_calculation` to `TrackingStatus` and `tracking.sample`.
- Consumes Task 1 settings and Task 2 `head_state()`.

- [ ] **Step 1: Write failing absolute-target tests**

```python
def test_face_target_uses_fresh_actual_telemetry():
    result = calculate_face_head_target(
        eye_midpoint=(0.90, 0.50),
        telemetry={
            "control_connected": True,
            "telemetry_age_seconds": 0.2,
            "pan_actual": 80,
            "tilt_actual": 90,
        },
        fallback=(100, 85),
        horizontal_fov_degrees=62.0,
        vertical_fov_degrees=48.5,
        pan_gain=1.0,
        tilt_gain=1.0,
        pan_sign=1,
        tilt_sign=1,
    )
    assert result.origin_source == "actual"
    assert result.origin_pan == 80
    assert result.target_pan == 106


def test_face_target_falls_back_and_clamps():
    result = calculate_face_head_target(
        eye_midpoint=(1.0, 0.0),
        telemetry={"control_connected": False},
        fallback=(130, 40),
        horizontal_fov_degrees=62.0,
        vertical_fov_degrees=48.5,
        pan_gain=1.0,
        tilt_gain=1.0,
        pan_sign=1,
        tilt_sign=1,
    )
    assert result.origin_source == "acknowledged"
    assert (result.target_pan, result.target_tilt) == (135, 35)
```

Add async tests proving centered eyes emit no command and initial acquisition
at `x=0.90` emits one absolute target based on supplied actual telemetry rather
than the old 18-degree/smoothed step.

- [ ] **Step 2: Verify RED**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_tracking.py `
  -k "face_target or angular_deadband or initial_angular" -q
```

Expected: calculation type/function and absolute behavior are absent.

- [ ] **Step 3: Implement calculation and face-only integration**

Create a frozen dataclass and pure function. Round final servo targets once,
after gains/signs, and clamp to existing ranges. Treat telemetry as actual only
when connected, age is numeric and `<= 1.0`, and actual pan/tilt are non-boolean
integers.

Pass all calibration settings into `PersonTrackingService`. In
`_update_head_and_body`, use the target's fresh `eye_midpoint` for a
`FaceObservation`; retain the existing person controller for shadow and
person-only modes. Emit no command when both calculated angular errors are
within the angular deadband.

Set `self._last_gaze_calculation` before command publication. Do not update
`self.head_pan` or `self.head_tilt` unless the command completes under the
existing generation check.

- [ ] **Step 4: Add residual, status, and pivot tests**

Add literal tests that:

- the next fresh face frame produces only the residual correction;
- an in-deadband residual emits no command;
- settling frames retain the previous calculation and estimator;
- `/tracking/status` returns `gaze_calculation`;
- `tracking.sample` includes calculation numbers and no image keys;
- a just-issued head target does not immediately authorize a body pivot.

- [ ] **Step 5: Verify GREEN**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m pytest `
  pc_brain\tests\test_tracking.py `
  pc_brain\tests\test_tracking_temporal.py `
  pc_brain\tests\test_main_endpoints.py -q
```

Expected: all selected tests pass.

### Task 4: Configuration Handoff, Documentation, and Full Verification

**Files:**
- Modify: `pc_brain/.env.example`
- Modify: `pc_brain/README.md`
- Test: `pc_brain/tests`
- Test: `pc_tracking/tests`

**Interfaces:**
- Documents the five calibration variables and PowerShell trial commands.
- Preserves all existing browser and tracking endpoints.

- [ ] **Step 1: Add exact environment defaults**

```dotenv
ROBIT_TRACKING_CAMERA_HORIZONTAL_FOV_DEGREES=62
ROBIT_TRACKING_CAMERA_VERTICAL_FOV_DEGREES=48.5
ROBIT_TRACKING_PAN_ANGLE_GAIN=1.0
ROBIT_TRACKING_TILT_ANGLE_GAIN=1.0
ROBIT_TRACKING_ANGULAR_DEADBAND_DEGREES=2
```

Document that gains above `1.0` increase correction and values below `1.0`
reduce it. Tell operators to change one axis at a time in `face_only`.

- [ ] **Step 2: Run fresh full verification**

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe -m compileall -q pc_brain\app pc_brain\tests
.\pc_tracking\.venv\Scripts\python.exe -m compileall -q pc_tracking\app pc_tracking\tests
.\pc_brain\.venv\Scripts\python.exe -m pytest pc_brain\tests -q
.\pc_tracking\.venv\Scripts\python.exe -m pytest pc_tracking\tests -q
git diff --check
```

Expected: both compile commands exit zero, all tests pass, and the whitespace
check reports no errors.

- [ ] **Step 3: Run attended PowerShell trial**

After stopping the previous supervisor with `Ctrl+C`:

```powershell
cd C:\Users\z1sou\HouseGremlin
$env:ROBIT_TRACKING_DETECTOR_MODE = "face_only"
.\Scripts\run.bat
```

Verify the first post-acquisition movement centers the face with no more than
one residual correction and remains physically acceleration-limited. Inspect:

```powershell
Invoke-RestMethod http://localhost:8080/tracking/status |
    ConvertTo-Json -Depth 8
```


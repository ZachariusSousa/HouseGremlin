# Restore Full Robit Console Functionality Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate camera, tracking, voice, chat, telemetry, memory, diagnostics, and manual robot controls into the approved RB-01 retro-anime HUD.

**Architecture:** Keep `web_control/index.html` as a dependency-free single-file client served by FastAPI. Six in-page subsystem screens share one browser state model and the existing HTTP/WebSocket interfaces; no backend schemas change.

**Tech Stack:** HTML5, CSS, browser JavaScript, FastAPI, pytest.

**Spec:** User-approved restore plan from 2026-09-07.

## Global Constraints

- Preserve the approved CRT console shell and visual grammar.
- Camera starts automatically; tracking requires explicit activation.
- Voice requires an explicit connect gesture and uses the brain WebSocket gateway.
- Manual control uses a simple session-local arm toggle and always stops on release/disarm.
- Browser-side voice tools and `/tracking/approach` remain absent.

---

### Task 1: Restore the browser contract

- [x] Add failing contract assertions for every required endpoint, frame matching, voice capture, control disarm, and six mode panels.
- [x] Confirm the approved static mock fails the operational contract.
- [x] Retain the visual identity assertions.

### Task 2: Build the six live subsystem screens

- [x] Preserve Overview and add separate Optical, Manual, Voice, Memory, and Diagnostic workspaces.
- [x] Make F1-F6 and the mode buttons switch exactly one workspace at a time.
- [x] Prevent decorative overlays from intercepting navigation.
- [x] Replace Overview's fictional subsystem indicators with live state summaries.

### Task 3: Restore camera, tracking, and perception

- [x] Restore low-rate camera capture, frame headers, object URL lifecycle, and visibility-aware polling.
- [x] Restore frame-matched normalized bounding boxes and tracking telemetry.
- [x] Keep tracking off until explicit start and expose explicit stop.
- [x] Restore perception summary and entities.

### Task 4: Restore control, chat, voice, memory, and diagnostics

- [x] Restore arm-gated drive and head commands with queued requests and automatic stop behavior.
- [x] Connect both emergency-stop controls to `/robot/stop` and disarm the UI.
- [x] Restore text chat and conversation rendering.
- [x] Restore realtime microphone capture, PCM16 streaming, transcript events, playback, and reconnect cleanup.
- [x] Restore read-only memory and action journal views.
- [x] Restore structured diagnostic refresh and reconnect controls.

### Task 5: Verify

- [x] Validate JavaScript syntax and static operational contracts.
- [x] Exercise all mode controls at the rendered desktop viewport with no horizontal overflow.
- [x] Verify camera, frame-matched overlay, tracking, chat, drive, release-stop, head, emergency stop, memory, and diagnostics against a mock brain.
- [ ] Run pytest after restoring the repository's broken Python 3.11 virtual environment.
- [ ] Run hardware-in-the-loop checks through `Scripts\run.bat` when Robit is available.

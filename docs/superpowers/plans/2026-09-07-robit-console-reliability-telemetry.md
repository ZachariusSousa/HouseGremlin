# Robit Console Reliability and Telemetry Repair

## Goal

Repair the frontend as one cohesive monitoring-and-control console. Overview owns the sole live video and tracking overlay, F2 becomes Telemetry, every displayed value has a real source, LLM failures are observable, and X eyes appear only for a currently active critical robot-control fault.

## Global Constraints

- Preserve the retro visual language, six-key navigation, existing public control/chat routes, the 12-second ESP heartbeat watchdog, and compatibility with older firmware telemetry.
- Keep `/health.ok` as PC-process liveness, add truthful readiness/components, and never infer robot or LLM readiness from configuration alone.
- Poll browser telemetry once per second only while visible and retain 60 samples in memory; do not persist one-second telemetry.
- Report unsupported or missing measurements as `null`/`N/A`; never invent values or expose secrets.
- Execute no movement from malformed or unvalidated LLM output.
- Use tests first and preserve all unrelated user changes.

## Task 1: Fault lifecycle and canonical robot status

- Replace the persisted global safety latch with an active fault registry containing source, severity, message, and timestamp.
- Do not reactivate journaled transient faults at startup. Only critical actuation, ESP-watchdog, or confirmed robot-control faults force X eyes; LLM, voice, camera, and tracking failures are degraded services.
- Clear source-specific faults on recovery and send an immediate heartbeat/current desired state after reconnect.
- Normalize TCP, HTTP, and cached robot status into one non-blocking shape with source and age. Use online at <=3 seconds, stale at >3 through 15 seconds, offline after 15 seconds/disconnect, and fault for an active critical robot fault.
- Expose actual/requested eyes and the active override reason.
- Add red/green tests for restart, fault priority/recovery, noncritical degradation, transport normalization, freshness thresholds, and non-blocking cached status.

## Task 2: System telemetry and truthful readiness

- Add `GET /system/telemetry` with `host`, `llm`, `robot`, `tracking`, and active `faults` sections.
- Sample host CPU/RAM/process CPU and RSS, event-loop lag, network totals/rates, and optional NVIDIA GPU utilization/VRAM using a bounded `nvidia-smi` call.
- Explicitly pin psutil. Unsupported GPU collection returns structured unavailable data without failing the endpoint.
- Keep `/health.ok` as liveness and add `ready` plus per-component status, reason, latency, and check time.
- Probe LLM `/v1/models` on startup and every 10 seconds without token generation; track last inference latency/result.
- Add red/green tests for sampler deltas, counter reset, unavailable GPU, telemetry shape, readiness truth, and stale component state.

## Task 3: Reliable LLM action output

- Use the existing action Pydantic model as strict JSON Schema for `/chat/action`, matching the working vision pattern.
- Retry once without `response_format` only when a provider rejects that capability, while retaining identical Pydantic validation.
- Surface warm-up/probe failures rather than swallowing them and record request outcome/latency.
- Never execute invalid output.
- Add red/green tests for schema request, unsupported-schema fallback, valid action, malformed output, timeout, and recovery.

## Task 4: ESP32 telemetry extension

- Add bounded additive telemetry for uptime, Wi-Fi mode/RSSI, heap and minimum heap, total heap, PSRAM free/total, camera state, and heartbeat armed/fault flags.
- Preserve protocol size and old-client compatibility; do not change the watchdog safety behavior.
- Update PC normalization tests with both old and new packets. Hardware compile/flash and watchdog validation remain manual if no CLI is available.

## Task 5: Frontend rebuild

- Move the sole real camera and frame-matched tracking overlay into Overview with retry, tracking toggle, scene summary, and entities.
- Repurpose F2 Optical as Telemetry. Move overview-only wires/stamp/registration into the Overview panel.
- Replace hard-coded 73% with real GPU VRAM percentage or N/A. Replace the static plot with a rolling 60-second RX/TX graph and add robot RTT history on Telemetry.
- Present PC/LLM metrics separately from ESP metrics, including explicit loading/unknown/stale/offline/fault labels.
- Poll `/system/telemetry` at 1 Hz while visible, immediately refresh on visibility regain, and cap histories at 60 samples.
- Add chat pending/elapsed/error/retry behavior, disable duplicate sends, and guarantee WebSocket errors leave Connecting state.
- Add red/green frontend contract tests and verify extracted JavaScript with Node plus browser interactions across all tabs.

## Task 6: Integrated verification

- Run all Python suites, frontend syntax/contract tests, endpoint smoke checks, and browser checks.
- Verify exact acceptance requirements against the approved plan.
- Record unavailable hardware-only checks explicitly; do not claim them complete without an ESP flash and controlled heartbeat test.

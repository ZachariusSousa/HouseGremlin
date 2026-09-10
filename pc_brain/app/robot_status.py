from __future__ import annotations

from typing import Any


ONLINE_MAX_AGE_MS = 3000.0
STALE_MAX_AGE_MS = 15000.0


def _first(sample: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def normalize_robot_status(
    sample: dict[str, Any] | None,
    *,
    source: str,
    sample_age_ms: float | None,
    connected: bool,
    ready: bool,
    desired_head: dict[str, Any] | None = None,
    desired_eyes: str | None = None,
    override_reason: str | None = None,
    control: dict[str, Any] | None = None,
    critical_fault: bool = False,
) -> dict[str, Any]:
    """Normalize robot samples without performing transport I/O."""
    raw = sample or {}
    desired_head = desired_head or {}
    control = control or {}
    age_ms = max(0.0, float(sample_age_ms)) if sample_age_ms is not None else None
    usable_sample = bool(raw) and raw.get("ok") is not False

    actual_pan = _first(raw, "pan_actual", "pan")
    actual_tilt = _first(raw, "tilt_actual", "tilt")
    target_pan = _first(raw, "pan_target")
    target_tilt = _first(raw, "tilt_target")
    if target_pan is None:
        target_pan = desired_head.get("pan", actual_pan)
    if target_tilt is None:
        target_tilt = desired_head.get("tilt", actual_tilt)

    actual_eyes = _first(raw, "eyes", "eye_expression")
    requested_eyes = desired_eyes
    if requested_eyes is None:
        requested_eyes = _first(raw, "requested_eyes", "eyes_target", "eyes")

    heartbeat_armed = _first(
        raw, "heartbeat_armed", "brain_heartbeat_armed", "ha"
    )
    heartbeat_fault = _first(raw, "heartbeat_fault", "brain_heartbeat_fault", "fault")
    watchdog_fault = heartbeat_fault is True
    if override_reason is None and watchdog_fault:
        override_reason = "esp_watchdog"

    if critical_fault or watchdog_fault:
        status = "fault"
    elif not usable_sample or age_ms is None or age_ms > STALE_MAX_AGE_MS:
        status = "offline"
    elif age_ms > ONLINE_MAX_AGE_MS:
        status = "stale"
    elif connected and ready:
        status = "online"
    else:
        status = "offline"

    movement = _first(raw, "movement", "move")
    speed = _first(raw, "speed", "motor_speed")
    camera_enabled = _first(raw, "camera_enabled", "camera")
    protocol = _first(raw, "protocol", "protocol_version")
    if protocol is None:
        protocol = control.get("protocol")
    wifi_mode = _first(raw, "wifi_mode", "mode", "wm")

    return {
        "ok": status in {"online", "stale"},
        "status": status,
        "connected": bool(connected),
        "ready": bool(ready),
        "source": source,
        "sample_age_ms": age_ms,
        "movement": movement,
        "speed": speed,
        "head": {
            "actual": {"pan": actual_pan, "tilt": actual_tilt},
            "target": {"pan": target_pan, "tilt": target_tilt},
        },
        "eyes": {
            "actual": actual_eyes,
            "requested": requested_eyes,
            "override_reason": override_reason,
        },
        "firmware": {
            "version": _first(raw, "firmware_version", "firmware"),
            "protocol": protocol,
            "uptime_ms": _first(raw, "uptime_ms", "uptime", "u"),
            "ip": _first(raw, "ip"),
            "hostname": _first(raw, "hostname"),
            "wifi": {
                "mode": wifi_mode,
                "rssi": _first(raw, "wifi_rssi", "rssi"),
            },
            "memory": {
                "heap_free_bytes": _first(
                    raw, "heap_free_bytes", "free_heap_bytes", "hf"
                ),
                "heap_min_free_bytes": _first(
                    raw, "heap_min_free_bytes", "minimum_free_heap_bytes", "hm"
                ),
                "heap_total_bytes": _first(
                    raw, "heap_total_bytes", "total_heap_bytes", "ht"
                ),
                "psram_free_bytes": _first(
                    raw, "psram_free_bytes", "free_psram_bytes", "pf"
                ),
                "psram_total_bytes": _first(
                    raw, "psram_total_bytes", "total_psram_bytes", "pt"
                ),
            },
            "camera_enabled": camera_enabled,
        },
        "watchdog": {
            "armed": heartbeat_armed,
            "fault": heartbeat_fault,
            "control_last_receive_age_ms": _first(
                raw,
                "control_last_receive_age_ms",
                "last_control_receive_age_ms",
                "ca",
            ),
        },
        "control": {
            "round_trip_ms": control.get("last_round_trip_ms"),
            "reconnect_count": control.get("reconnect_count", 0),
            "error": control.get("last_error"),
        },
        # Flat aliases keep the current public status consumers working while
        # new telemetry uses the canonical nested structures above.
        "control_channel": "ready" if connected and ready else "disconnected",
        "mode": wifi_mode,
        "pan": actual_pan,
        "tilt": actual_tilt,
        "pan_actual": actual_pan,
        "tilt_actual": actual_tilt,
        "pan_target": target_pan,
        "tilt_target": target_tilt,
        "camera": camera_enabled,
    }

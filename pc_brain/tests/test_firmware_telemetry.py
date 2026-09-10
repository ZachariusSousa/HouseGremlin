from __future__ import annotations

import json

from app.robot_status import normalize_robot_status


def normalize_tcp(sample: dict) -> dict:
    return normalize_robot_status(
        sample,
        source="tcp",
        sample_age_ms=10.0,
        connected=True,
        ready=True,
    )


def test_maximum_extended_telemetry_packet_stays_below_protocol_bound():
    # This is the largest supported protocol-v1 envelope: a 48-byte session,
    # 32-bit counters, and the longest normal state labels.
    packet = {
        "v": 1,
        "type": "telemetry",
        "session": "s" * 48,
        "last_seq": 4_294_967_295,
        "state": {
            "movement": "reverse",
            "speed": 255,
            "pan_actual": 135,
            "tilt_actual": 115,
            "pan_target": 135,
            "tilt_target": 115,
            "eyes": "concerned",
            "wifi_rssi": -128,
            "camera": True,
            "fault": True,
            "u": 4_294_967_295,
            "wm": "sta",
            "hf": 4_294_967_295,
            "hm": 4_294_967_295,
            "ht": 4_294_967_295,
            "pf": 4_294_967_295,
            "pt": 4_294_967_295,
            "ha": True,
            "ca": 4_294_967_295,
        },
    }

    encoded = json.dumps(packet, separators=(",", ":")).encode()

    assert len(encoded) < 512


def test_legacy_tcp_packet_normalization_is_unchanged():
    normalized = normalize_tcp(
        {
            "movement": "stop",
            "speed": 0,
            "pan_actual": 91,
            "tilt_actual": 89,
            "pan_target": 95,
            "tilt_target": 85,
            "eyes": "happy",
            "wifi_rssi": -45,
            "camera": True,
            "fault": False,
        }
    )

    assert normalized["head"] == {
        "actual": {"pan": 91, "tilt": 89},
        "target": {"pan": 95, "tilt": 85},
    }
    assert normalized["firmware"]["wifi"] == {"mode": None, "rssi": -45}
    assert normalized["firmware"]["memory"] == {
        "heap_free_bytes": None,
        "heap_min_free_bytes": None,
        "heap_total_bytes": None,
        "psram_free_bytes": None,
        "psram_total_bytes": None,
    }
    assert normalized["firmware"]["camera_enabled"] is True
    assert normalized["watchdog"] == {
        "armed": None,
        "fault": False,
        "control_last_receive_age_ms": None,
    }


def test_extended_compact_tcp_packet_normalizes_to_descriptive_firmware_health():
    normalized = normalize_tcp(
        {
            "movement": "stop",
            "speed": 0,
            "pan_actual": 90,
            "tilt_actual": 90,
            "pan_target": 90,
            "tilt_target": 90,
            "eyes": "neutral",
            "wifi_rssi": -57,
            "camera": True,
            "fault": False,
            "u": 123_456,
            "wm": "ap",
            "hf": 201_000,
            "hm": 180_000,
            "ht": 327_680,
            "pf": 0,
            "pt": 0,
            "ha": True,
            "ca": 125,
        }
    )

    assert normalized["firmware"] == {
        "version": None,
        "protocol": None,
        "uptime_ms": 123_456,
        "ip": None,
        "hostname": None,
        "wifi": {"mode": "ap", "rssi": -57},
        "memory": {
            "heap_free_bytes": 201_000,
            "heap_min_free_bytes": 180_000,
            "heap_total_bytes": 327_680,
            "psram_free_bytes": 0,
            "psram_total_bytes": 0,
        },
        "camera_enabled": True,
    }
    assert normalized["watchdog"] == {
        "armed": True,
        "fault": False,
        "control_last_receive_age_ms": 125,
    }


def test_status_json_never_carries_credential_values_from_a_raw_sample():
    secret_ssid = "HouseGremlin-private-network"
    secret_password = "correct-horse-battery-staple"
    status_json = json.dumps(
        normalize_tcp(
            {
                "ok": True,
                "movement": "stop",
                "ssid": secret_ssid,
                "password": secret_password,
                "ROBIT_STA_SSID": secret_ssid,
                "ROBIT_STA_PASSWORD": secret_password,
            }
        )
    )

    assert secret_ssid not in status_json
    assert secret_password not in status_json
    assert "ROBIT_STA_SSID" not in status_json
    assert "ROBIT_STA_PASSWORD" not in status_json

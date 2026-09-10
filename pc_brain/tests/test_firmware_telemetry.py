from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.robot_status import normalize_robot_status


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIRMWARE_DIRECTORY = REPOSITORY_ROOT / "firmware" / "robit_controller"
HARNESS_SOURCE = FIRMWARE_DIRECTORY / "tests" / "telemetry_serialization_harness.cpp"
SERIALIZER_SOURCE = FIRMWARE_DIRECTORY / "telemetry_serialization.cpp"
VS_DEVELOPER_COMMAND = Path(
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)


@pytest.fixture(scope="session")
def firmware_serialized_samples(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output_directory = tmp_path_factory.mktemp("firmware-telemetry")
    executable = output_directory / "telemetry_serialization_harness.exe"
    command = (
        f'call "{VS_DEVELOPER_COMMAND}" -arch=x64 -host_arch=x64 >nul '
        f'&& cl /nologo /std:c++17 /EHsc /W4 /I"{FIRMWARE_DIRECTORY}" '
        f'"{HARNESS_SOURCE}" "{SERIALIZER_SOURCE}" /Fe"{executable}"'
    )
    build = subprocess.run(
        command,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        cwd=output_directory,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    run = subprocess.run(
        [str(executable)], check=False, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(run.stdout)


def normalize_tcp(sample: dict) -> dict:
    return normalize_robot_status(
        sample,
        source="tcp",
        sample_age_ms=10.0,
        connected=True,
        ready=True,
    )


def test_shared_firmware_serializer_emits_bounded_extended_telemetry(
    firmware_serialized_samples: dict,
):
    packet = firmware_serialized_samples["tcp"]

    assert firmware_serialized_samples["tcp_bytes"] < 512
    assert firmware_serialized_samples["tcp_bytes"] == len(
        json.dumps(packet, separators=(",", ":")).encode()
    )
    assert packet["v"] == 1
    assert packet["type"] == "telemetry"
    assert packet["session"] == "s" * 48
    assert packet["last_seq"] == 4_294_967_295
    assert packet["state"] == {
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
        "wm": "ap",
        "hf": 4_294_967_295,
        "hm": 4_294_967_295,
        "ht": 4_294_967_295,
        "pf": 0,
        "pt": 0,
        "ha": True,
        "ca": 4_294_967_295,
    }


def test_shared_firmware_serializer_legacy_state_normalizes_unchanged(
    firmware_serialized_samples: dict,
):
    normalized = normalize_tcp(firmware_serialized_samples["legacy"])

    assert normalized["head"] == {
        "actual": {"pan": 135, "tilt": 115},
        "target": {"pan": 135, "tilt": 115},
    }
    assert normalized["firmware"]["wifi"] == {"mode": None, "rssi": -128}
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
        "fault": True,
        "control_last_receive_age_ms": None,
    }


def test_shared_firmware_serializer_extended_packet_normalizes_to_health(
    firmware_serialized_samples: dict,
):
    normalized = normalize_tcp(firmware_serialized_samples["tcp"]["state"])

    assert normalized["firmware"] == {
        "version": None,
        "protocol": None,
        "uptime_ms": 4_294_967_295,
        "ip": None,
        "hostname": None,
        "wifi": {"mode": "ap", "rssi": -128},
        "memory": {
            "heap_free_bytes": 4_294_967_295,
            "heap_min_free_bytes": 4_294_967_295,
            "heap_total_bytes": 4_294_967_295,
            "psram_free_bytes": 0,
            "psram_total_bytes": 0,
        },
        "camera_enabled": True,
    }
    assert normalized["watchdog"] == {
        "armed": True,
        "fault": True,
        "control_last_receive_age_ms": 4_294_967_295,
    }
    assert normalized["mode"] == "ap"


def test_shared_firmware_serializer_emits_descriptive_http_without_credentials(
    firmware_serialized_samples: dict,
):
    http = firmware_serialized_samples["http"]
    rendered = json.dumps(firmware_serialized_samples, separators=(",", ":"))

    assert http["ok"] is True
    assert http["wifi_mode"] == "ap"
    assert http["uptime_ms"] == 4_294_967_295
    assert http["heap_free_bytes"] == 4_294_967_295
    assert http["psram_free_bytes"] == 0
    assert http["camera_enabled"] is True
    assert http["heartbeat_armed"] is True
    assert http["heartbeat_fault"] is True
    assert http["control_last_receive_age_ms"] == 4_294_967_295
    assert firmware_serialized_samples["escaped_http"] == {
        **http,
        "movement": 'move"\\path',
        "move": 'move"\\path',
        "eyes": 'eyes"\\path',
        "ip": '10.0.0.1"\\path',
        "hostname": 'robit"\\node.local',
    }
    assert "HouseGremlin-private-network" not in rendered
    assert "correct-horse-battery-staple" not in rendered
    assert "ROBIT_STA_SSID" not in rendered
    assert "ROBIT_STA_PASSWORD" not in rendered

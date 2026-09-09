from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main


class RecordingLlmHealth:
    def __init__(self, provider, model):
        self.data = {
            "status": "unknown",
            "provider": provider,
            "model": model,
            "last_probe_at": None,
            "last_probe_latency_ms": None,
            "last_probe_error": None,
            "last_inference_at": None,
            "last_inference_latency_ms": None,
            "last_inference_error": None,
            "last_inference_success": None,
        }

    def snapshot(self):
        return dict(self.data)

    def record_probe(self, *, success, checked_at, latency_ms, error=None):
        self.data.update(
            status="ready" if success else "unavailable",
            last_probe_at=checked_at,
            last_probe_latency_ms=latency_ms,
            last_probe_error=error,
        )

    def record_inference(self, *, success, checked_at, latency_ms, error=None):
        self.data.update(
            status="ready" if success else "degraded",
            last_inference_at=checked_at,
            last_inference_latency_ms=latency_ms,
            last_inference_error=error,
            last_inference_success=success,
        )


class FakeTelemetryService:
    async def latest_host(self):
        return {
            "sampled_at": "2026-09-08T00:00:00+00:00",
            "sequence": 7,
            "host": {
                "cpu_percent": 12.5,
                "memory": {"used_bytes": 500, "total_bytes": 1_000, "percent": 50.0},
                "process": {"cpu_percent": 7.5, "rss_bytes": 256},
                "event_loop_lag_ms": 2.5,
                "network": {
                    "rx_bytes": 1_000,
                    "tx_bytes": 2_000,
                    "rx_bytes_per_second": 100.0,
                    "tx_bytes_per_second": 50.0,
                },
                "gpu": {
                    "available": False,
                    "name": None,
                    "utilization_percent": None,
                    "memory_used_bytes": None,
                    "memory_total_bytes": None,
                    "reason": "nvidia-smi not found",
                },
            },
        }


class FakeTrackingService:
    def status(self):
        return SimpleNamespace(
            available=False,
            enabled=False,
            state="off",
            mode="off",
            reason="detector unavailable",
            detector_latency_ms=None,
            detector_queue_ms=None,
            detector_cadence_fps=0.0,
            backend=None,
            target_age_seconds=None,
        )


class FakeFrameBroker:
    latest = None

    def status(self):
        return {
            "effective_fps": 0.2,
            "last_acquisition_ms": None,
            "last_frame_bytes": None,
            "frame_id": None,
            "last_success_at": None,
            "last_error_at": None,
            "last_error": None,
            "frame_age_seconds": None,
        }


class FakeActuatorBroker:
    channel = SimpleNamespace(
        status=lambda: {
            "ready": False,
            "last_error": "control disconnected",
            "last_round_trip_ms": None,
        }
    )

    def status(self):
        return {}


def configure_readiness_fakes(monkeypatch, llm_health):
    monkeypatch.setattr(main, "telemetry_service", FakeTelemetryService(), raising=False)
    monkeypatch.setattr(main, "llm_health", llm_health, raising=False)
    monkeypatch.setattr(main, "get_tracking_service", lambda: FakeTrackingService())
    monkeypatch.setattr(main, "get_frame_broker", lambda: FakeFrameBroker())
    monkeypatch.setattr(main, "get_actuator_broker", lambda: FakeActuatorBroker())

    async def offline_robot():
        return {
            "ok": False,
            "status": "offline",
            "source": "none",
            "sample_age_ms": None,
            "control": {"round_trip_ms": None, "error": "control disconnected"},
        }

    monkeypatch.setattr(main, "robot_status", offline_robot)


@pytest.mark.anyio
async def test_lifespan_cleans_partial_startup_without_masking_start_error(monkeypatch):
    events = []

    class Client:
        async def aclose(self):
            events.append("client.closed")

    class Telemetry:
        async def start(self):
            events.append("telemetry.started")

        async def shutdown(self):
            events.append("telemetry.stopped")

    class Actuators:
        async def start(self):
            events.append("actuators.starting")
            raise RuntimeError("actuator start failed")

        async def shutdown(self):
            events.append("actuators.stopped")
            raise RuntimeError("actuator shutdown failed")

    client = Client()
    monkeypatch.setattr(main, "ensure_data_dirs", lambda path: None)
    monkeypatch.setattr(main, "get_brain_coordinator", lambda: SimpleNamespace())
    monkeypatch.setattr(main.httpx, "AsyncClient", lambda timeout: client)
    monkeypatch.setattr(main, "telemetry_service", Telemetry())
    monkeypatch.setattr(main, "get_actuator_broker", lambda: Actuators())

    with pytest.raises(RuntimeError, match="actuator start failed"):
        async with main.lifespan(main.app):
            raise AssertionError("startup failure must prevent serving")

    assert events == [
        "telemetry.started",
        "actuators.starting",
        "actuators.stopped",
        "telemetry.stopped",
        "client.closed",
    ]
    assert main.event_loop_monitor_task is None
    assert main.robot_http_client is None


def test_system_telemetry_has_approved_sections_canonical_robot_and_no_secrets(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    configure_readiness_fakes(monkeypatch, health)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot-user:supersecret@private-robot",
            llm_base_url="http://llm-user:supersecret@private-llm/v1?api_key=supersecret",
            llm_provider="openai_compatible",
            llm_model="gemma4:e4b",
        ),
    )
    coordinator = main.get_brain_coordinator()
    coordinator.register_fault(
        "control_transport",
        "critical",
        "control disconnected",
        "corr-telemetry",
    )

    response = TestClient(main.app).get("/system/telemetry")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "sampled_at",
        "sequence",
        "host",
        "llm",
        "robot",
        "tracking",
        "faults",
    }
    assert payload["robot"]["status"] == "offline"
    assert payload["robot"]["control"]["error"] == "control disconnected"
    assert payload["tracking"] == {
        "available": False,
        "active": False,
        "latest_frame_age_seconds": None,
        "latest_result_age_seconds": None,
        "error": "detector unavailable",
    }
    assert payload["faults"][0]["source"] == "control_transport"
    assert payload["faults"][0]["severity"] == "critical"
    assert "supersecret" not in response.text
    assert "private-robot" not in response.text
    assert "private-llm" not in response.text


def test_system_telemetry_never_serializes_secrets_from_fault_messages(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    configure_readiness_fakes(monkeypatch, health)
    coordinator = main.get_brain_coordinator()
    coordinator.register_fault(
        "credential_transport",
        "critical",
        "websocket failed wss://alice:url-secret@robot.local/live"
        "?token=query-secret#fragment-secret while reconnecting "
        "api_key=plain-secret Authorization: Basic basic-secret "
        "Authorization: Bearer bearer-secret",
        "corr-secret-telemetry",
    )

    response = TestClient(main.app).get("/system/telemetry")

    assert response.status_code == 200
    fault = next(
        item for item in response.json()["faults"]
        if item["source"] == "credential_transport"
    )
    assert "[redacted-url]" in fault["message"]
    assert "Authorization: Basic [redacted]" in fault["message"]
    assert "Authorization: Bearer [redacted]" in fault["message"]
    for secret in (
        "alice",
        "url-secret",
        "robot.local",
        "query-secret",
        "fragment-secret",
        "plain-secret",
        "basic-secret",
        "bearer-secret",
    ):
        assert secret not in response.text
    assert "websocket failed" in fault["message"]
    assert "while reconnecting" in fault["message"]
    assert len(fault["message"]) <= 500


def test_health_keeps_liveness_true_but_reports_truthful_not_ready_components(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    configure_readiness_fakes(monkeypatch, health)
    monkeypatch.setattr(main.shutil, "which", lambda executable: None)
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda module: None)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot-user:supersecret@private-robot",
            llm_base_url="http://llm-user:supersecret@private-llm/v1?api_key=supersecret",
            llm_provider="openai_compatible",
            llm_model="gemma4:e4b",
            realtime_voice="serena",
            realtime_instructions="test instructions",
        ),
    )

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["ready"] is False
    assert set(payload["components"]) == {"robot", "llm", "tracking", "camera", "voice"}
    assert payload["components"]["robot"]["status"] == "offline"
    assert payload["components"]["llm"]["status"] == "unknown"
    assert payload["components"]["tracking"]["status"] == "degraded"
    assert payload["components"]["camera"]["status"] == "degraded"
    assert payload["components"]["voice"]["status"] == "degraded"
    for component in payload["components"].values():
        assert set(component) == {"status", "reason", "checked_at", "latency_ms"}
    assert payload["robot_base_url"] == "http://private-robot"
    assert payload["llm_base_url"] == "http://private-llm/v1"
    assert "supersecret" not in response.text
    assert "api_key" not in response.text


def test_health_is_ready_when_robot_and_llm_are_ready_despite_optional_degradation(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:00+00:00",
        latency_ms=12.0,
    )
    configure_readiness_fakes(monkeypatch, health)
    monkeypatch.setattr(main.shutil, "which", lambda executable: None)
    monkeypatch.setattr(main.importlib.util, "find_spec", lambda module: None)
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            robot_base_url="http://robot",
            llm_base_url="http://localhost:11434/v1",
            llm_provider="openai_compatible",
            llm_model="gemma4:e4b",
            realtime_voice="serena",
            realtime_instructions="test instructions",
        ),
    )

    async def online_robot():
        return {
            "ok": True,
            "status": "online",
            "source": "tcp",
            "sample_age_ms": 10.0,
            "control": {"round_trip_ms": 8.0, "error": None},
        }

    monkeypatch.setattr(main, "robot_status", online_robot)

    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["components"]["robot"]["status"] == "online"
    assert payload["components"]["llm"]["status"] == "ready"
    assert payload["components"]["tracking"]["status"] == "degraded"
    assert payload["components"]["camera"]["status"] == "degraded"
    assert payload["components"]["voice"]["status"] == "degraded"


def test_camera_readiness_transitions_from_ready_to_failed_to_stale_to_recovered():
    class Broker:
        def __init__(self):
            self.current_status = {}

        def status(self):
            return self.current_status

    broker = Broker()
    robot = {"status": "online", "control": {"round_trip_ms": 1.0}}
    tracking = FakeTrackingService().status()
    voice = {"status": "ok", "degraded_reasons": []}

    broker.current_status = {
        "last_success_at": "2026-09-08T00:00:00+00:00",
        "last_error_at": None,
        "last_error": None,
        "last_acquisition_ms": 12.0,
        "frame_age_seconds": 4.999,
    }
    ready = main.readiness_components(robot, tracking, broker, voice)["camera"]
    assert ready == {
        "status": "ready",
        "reason": None,
        "checked_at": "2026-09-08T00:00:00+00:00",
        "latency_ms": 12.0,
    }

    broker.current_status.update(
        last_error_at="2026-09-08T00:00:01+00:00",
        last_error="camera acquisition failed (TimeoutError)",
    )
    failed = main.readiness_components(robot, tracking, broker, voice)["camera"]
    assert failed["status"] == "degraded"
    assert failed["reason"] == "camera acquisition failed (TimeoutError)"
    assert failed["checked_at"] == "2026-09-08T00:00:01+00:00"

    broker.current_status.update(
        last_success_at="2026-09-08T00:00:02+00:00",
        last_error=None,
        frame_age_seconds=5.001,
    )
    stale = main.readiness_components(robot, tracking, broker, voice)["camera"]
    assert stale["status"] == "stale"
    assert stale["reason"] == "camera frame is older than 5 seconds"

    broker.current_status.update(
        last_success_at="2026-09-08T00:00:03+00:00",
        last_error=None,
        frame_age_seconds=0.1,
    )
    recovered = main.readiness_components(robot, tracking, broker, voice)["camera"]
    assert recovered["status"] == "ready"
    assert recovered["reason"] is None
    assert recovered["checked_at"] == "2026-09-08T00:00:03+00:00"


def test_degraded_llm_readiness_uses_one_inference_observation(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:00+00:00",
        latency_ms=10.0,
    )
    health.record_inference(
        success=False,
        checked_at="2026-09-08T00:00:01+00:00",
        latency_ms=33.0,
        error="inference failed (HTTPException)",
    )
    monkeypatch.setattr(main, "llm_health", health)

    llm = main.readiness_components(
        {"status": "online", "control": {"round_trip_ms": 1.0}},
        FakeTrackingService().status(),
        FakeFrameBroker(),
        {"status": "ok", "degraded_reasons": []},
    )["llm"]

    assert llm == {
        "status": "degraded",
        "reason": "inference failed (HTTPException)",
        "checked_at": "2026-09-08T00:00:01+00:00",
        "latency_ms": 33.0,
    }


@pytest.mark.anyio
async def test_call_llm_records_success_and_failure_without_changing_error_behavior(monkeypatch):
    health = RecordingLlmHealth("openai_compatible", "gemma4:e4b")
    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:00+00:00",
        latency_ms=1.0,
    )
    monkeypatch.setattr(main, "llm_health", health, raising=False)

    class SuccessfulClient:
        async def chat(self, text):
            return SimpleNamespace(response="hello", model="gemma4:e4b")

    monkeypatch.setattr(main, "llm_client", SuccessfulClient())
    result = await main.call_llm("chat", "hello", [], include_live_scene=False)
    assert result.response == "hello"
    assert health.snapshot()["last_inference_success"] is True

    class FailingClient:
        async def chat(self, text):
            raise HTTPException(status_code=502, detail="provider failed")

    monkeypatch.setattr(main, "llm_client", FailingClient())
    with pytest.raises(HTTPException, match="provider failed"):
        await main.call_llm("chat", "hello", [], include_live_scene=False)

    failed = health.snapshot()
    assert failed["status"] == "degraded"
    assert failed["last_inference_success"] is False
    assert failed["last_inference_error"] == "inference failed (HTTPException)"


@pytest.mark.anyio
async def test_successful_probe_uses_models_endpoint_without_generating_tokens(monkeypatch):
    requested = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"object": "list", "data": [{"id": "gemma4:e4b"}]}

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, headers):
            requested.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeAsyncClient)

    await main.llm_client.probe_models()

    assert requested == [
        (
            f"{main.settings.llm_base_url}/models",
            {"authorization": "Bearer local"},
        )
    ]

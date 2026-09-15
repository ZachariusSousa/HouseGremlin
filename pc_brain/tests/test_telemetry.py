import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.telemetry import (
    HostSampler,
    LlmHealthState,
    NvidiaGpuSampler,
    SystemTelemetryService,
)


GPU_UNAVAILABLE = {
    "available": False,
    "name": None,
    "utilization_percent": None,
    "memory_used_bytes": None,
    "memory_total_bytes": None,
    "reason": "not installed",
}


class FakeProcessMetrics:
    def cpu_percent(self, interval=None):
        assert interval is None
        return 7.5

    def memory_info(self):
        return SimpleNamespace(rss=256)


class FakePsutil:
    def __init__(self):
        self.network = iter(
            (
                SimpleNamespace(bytes_recv=1_000, bytes_sent=2_000),
                SimpleNamespace(bytes_recv=1_400, bytes_sent=2_200),
                SimpleNamespace(bytes_recv=100, bytes_sent=50),
            )
        )

    def Process(self):
        return FakeProcessMetrics()

    def cpu_percent(self, interval=None):
        assert interval is None
        return 12.5

    def virtual_memory(self):
        return SimpleNamespace(used=500, total=1_000, percent=50.0)

    def net_io_counters(self):
        return next(self.network)


class StaticGpuSampler:
    async def sample(self):
        return dict(GPU_UNAVAILABLE)


@pytest.mark.anyio
async def test_host_samples_use_literal_network_rates_and_reset_without_negative_values():
    clock = iter((100.0, 104.0, 106.0))
    sampler = HostSampler(
        psutil_module=FakePsutil(),
        gpu_sampler=StaticGpuSampler(),
        monotonic=lambda: next(clock),
    )

    first = await sampler.sample(event_loop_lag_ms=2.5)
    second = await sampler.sample(event_loop_lag_ms=3.5)
    reset = await sampler.sample(event_loop_lag_ms=4.5)

    assert first["network"] == {
        "rx_bytes": 1_000,
        "tx_bytes": 2_000,
        "rx_bytes_per_second": 0.0,
        "tx_bytes_per_second": 0.0,
    }
    assert second["network"] == {
        "rx_bytes": 1_400,
        "tx_bytes": 2_200,
        "rx_bytes_per_second": 100.0,
        "tx_bytes_per_second": 50.0,
    }
    assert reset["network"] == {
        "rx_bytes": 100,
        "tx_bytes": 50,
        "rx_bytes_per_second": 0.0,
        "tx_bytes_per_second": 0.0,
    }
    assert first["cpu_percent"] == 12.5
    assert first["memory"] == {
        "used_bytes": 500,
        "total_bytes": 1_000,
        "percent": 50.0,
    }
    assert first["process"] == {"cpu_percent": 7.5, "rss_bytes": 256}
    assert first["event_loop_lag_ms"] == 2.5


class FakeGpuProcess:
    def __init__(self, stdout=b"", stderr=b"", *, block=False):
        self.stdout = stdout
        self.stderr = stderr
        self.block = block
        self.returncode = 0
        self.killed = False
        self.waited = False
        self.communicating = asyncio.Event()

    async def communicate(self):
        self.communicating.set()
        if self.block:
            await asyncio.Event().wait()
        return self.stdout, self.stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        self.returncode = -9


@pytest.mark.anyio
async def test_nvidia_gpu_sample_parses_one_bounded_non_shell_query():
    calls = []

    async def create_process(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeGpuProcess(stdout=b"NVIDIA RTX, 25, 1024, 4096\n")

    result = await NvidiaGpuSampler(create_process=create_process).sample()

    assert result == {
        "available": True,
        "name": "NVIDIA RTX",
        "utilization_percent": 25.0,
        "memory_used_bytes": 1_073_741_824,
        "memory_total_bytes": 4_294_967_296,
        "reason": None,
    }
    assert calls == [
        (
            (
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ),
            {"stdout": asyncio.subprocess.PIPE, "stderr": asyncio.subprocess.PIPE},
        )
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stdout",
    [
        b"NVIDIA RTX, NaN, 1024, 4096\n",
        b"NVIDIA RTX, Infinity, 1024, 4096\n",
        b"NVIDIA RTX, -0.1, 1024, 4096\n",
        b"NVIDIA RTX, 100.1, 1024, 4096\n",
        b"NVIDIA RTX, 25, -1, 4096\n",
        b"NVIDIA RTX, 25, 4097, 4096\n",
        b"NVIDIA RTX, 25, 1024, 1e309\n",
    ],
)
async def test_nvidia_gpu_rejects_non_finite_out_of_range_and_impossible_values(stdout):
    async def create_process(*args, **kwargs):
        return FakeGpuProcess(stdout=stdout)

    result = await NvidiaGpuSampler(create_process=create_process).sample()

    assert result == {
        "available": False,
        "name": None,
        "utilization_percent": None,
        "memory_used_bytes": None,
        "memory_total_bytes": None,
        "reason": "malformed nvidia-smi output",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "reason_fragment"),
    (("missing", "not found"), ("timeout", "timed out"), ("malformed", "malformed")),
)
async def test_nvidia_gpu_failures_return_structured_unavailable_data(failure, reason_fragment):
    process = FakeGpuProcess(
        stdout=b"not,enough,fields\n" if failure == "malformed" else b"",
        block=failure == "timeout",
    )

    async def create_process(*args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("nvidia-smi")
        return process

    result = await NvidiaGpuSampler(
        create_process=create_process,
        timeout_seconds=0.01,
    ).sample()

    assert result["available"] is False
    assert result["name"] is None
    assert result["utilization_percent"] is None
    assert result["memory_used_bytes"] is None
    assert result["memory_total_bytes"] is None
    assert reason_fragment in result["reason"]
    if failure == "timeout":
        assert process.killed is True


@pytest.mark.anyio
async def test_nvidia_gpu_cancellation_kills_and_reaps_child_process():
    process = FakeGpuProcess(block=True)

    async def create_process(*args, **kwargs):
        return process

    task = asyncio.create_task(
        NvidiaGpuSampler(create_process=create_process).sample()
    )
    await process.communicating.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
    assert process.waited is True


def test_llm_health_probe_recovery_preserves_newer_inference_failure():
    health = LlmHealthState(provider="openai_compatible", model="gemma4:e4b")

    assert health.snapshot()["status"] == "unknown"

    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:00+00:00",
        latency_ms=10.0,
    )
    assert health.snapshot()["status"] == "ready"

    health.record_inference(
        success=False,
        checked_at="2026-09-08T00:00:01+00:00",
        latency_ms=20.0,
        error="inference failed (HTTPException)",
    )
    assert health.snapshot()["status"] == "degraded"
    assert health.snapshot()["last_inference_success"] is False

    health.record_probe(
        success=False,
        checked_at="2026-09-08T00:00:02+00:00",
        latency_ms=30.0,
        error="probe failed (ConnectError)",
    )
    assert health.snapshot()["status"] == "unavailable"

    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:03+00:00",
        latency_ms=40.0,
    )
    recovered = health.snapshot()
    assert recovered["status"] == "degraded"
    assert recovered["last_probe_error"] is None
    assert recovered["last_probe_latency_ms"] == 40.0
    assert recovered["last_inference_error"] == "inference failed (HTTPException)"

    health.record_inference(
        success=True,
        checked_at="2026-09-08T00:00:04+00:00",
        latency_ms=50.0,
    )
    assert health.snapshot()["status"] == "ready"


def test_malformed_inference_degrades_a_reachable_llm_without_losing_latency():
    health = LlmHealthState(provider="openai_compatible", model="gemma4:e4b")
    health.record_probe(
        success=True,
        checked_at="2026-09-08T00:00:00+00:00",
        latency_ms=10.0,
    )
    health.record_inference(
        success=True,
        checked_at="2026-09-08T00:00:01+00:00",
        latency_ms=25.0,
    )

    health.mark_last_inference_malformed("malformed action response")

    snapshot = health.snapshot()
    assert snapshot["status"] == "degraded"
    assert snapshot["last_inference_success"] is False
    assert snapshot["last_inference_error"] == "malformed action response"
    assert snapshot["last_inference_latency_ms"] == 25.0


@pytest.mark.anyio
async def test_probe_failure_is_unavailable_and_does_not_expose_exception_secrets():
    clock = iter((100.0, 100.025))

    async def failing_probe():
        raise RuntimeError("http://user:supersecret@private-llm/v1/models")

    health = LlmHealthState("openai_compatible", "gemma4:e4b")
    service = SystemTelemetryService(
        host_sampler=SimpleNamespace(),
        llm_health=health,
        llm_probe=failing_probe,
        event_loop_lag=lambda: 0.0,
        monotonic=lambda: next(clock),
        utc_now=lambda: datetime(2026, 9, 8, tzinfo=timezone.utc),
    )

    await service.probe_llm_once()

    snapshot = health.snapshot()
    assert snapshot["status"] == "unavailable"
    assert snapshot["last_probe_latency_ms"] == pytest.approx(25.0)
    assert snapshot["last_probe_error"] == "probe failed (RuntimeError)"
    assert "supersecret" not in str(snapshot)


@pytest.mark.anyio
async def test_delayed_llm_probe_is_cancelled_within_budget_and_loop_continues():
    probe_calls = 0
    cancelled_probes = 0
    sleep_delays = []
    keep_sleeping = asyncio.Event()

    async def delayed_probe():
        nonlocal probe_calls, cancelled_probes
        probe_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_probes += 1
            raise

    async def recording_sleep(delay):
        sleep_delays.append(delay)
        if len(sleep_delays) > 1:
            await keep_sleeping.wait()

    health = LlmHealthState("openai_compatible", "gemma4:e4b")
    service = SystemTelemetryService(
        host_sampler=SimpleNamespace(),
        llm_health=health,
        llm_probe=delayed_probe,
        event_loop_lag=lambda: 0.0,
        probe_interval_seconds=0.05,
        probe_timeout_seconds=0.01,
        monotonic=lambda: 100.0,
        sleep=recording_sleep,
    )

    loop_task = asyncio.create_task(service._probe_loop())
    for _ in range(100):
        if len(sleep_delays) >= 2:
            break
        await asyncio.sleep(0.002)

    assert probe_calls == 2
    assert cancelled_probes == 2
    assert len(sleep_delays) == 2
    assert 0.0 <= sleep_delays[0] <= 0.05
    assert health.snapshot()["last_probe_error"] == "probe failed (TimeoutError)"

    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)


@pytest.mark.anyio
async def test_start_samples_and_probes_immediately_then_uses_one_and_ten_second_cadences():
    sleep_calls = {}
    block = asyncio.Event()

    class ImmediateHostSampler:
        async def sample(self, *, event_loop_lag_ms):
            return {"event_loop_lag_ms": event_loop_lag_ms}

    async def successful_probe():
        return None

    async def recording_sleep(delay):
        sleep_calls[asyncio.current_task().get_name()] = delay
        await block.wait()

    health = LlmHealthState("openai_compatible", "gemma4:e4b")
    service = SystemTelemetryService(
        host_sampler=ImmediateHostSampler(),
        llm_health=health,
        llm_probe=successful_probe,
        event_loop_lag=lambda: 2.0,
        monotonic=lambda: 100.0,
        utc_now=lambda: datetime(2026, 9, 8, tzinfo=timezone.utc),
        sleep=recording_sleep,
    )

    await service.start()
    for _ in range(20):
        if len(sleep_calls) == 2:
            break
        await asyncio.sleep(0)

    assert (await service.latest_host())["sequence"] == 1
    assert health.snapshot()["status"] == "ready"
    assert sleep_calls == {
        "system-telemetry-host": 1.0,
        "system-telemetry-llm": 10.0,
    }
    await service.shutdown()


@pytest.mark.anyio
async def test_concurrent_reads_share_latest_sample_without_resampling_gpu():
    gpu_calls = 0

    class CountingHostSampler:
        async def sample(self, *, event_loop_lag_ms):
            nonlocal gpu_calls
            gpu_calls += 1
            return {"gpu": {"available": False, "reason": "test"}}

    service = SystemTelemetryService(
        host_sampler=CountingHostSampler(),
        llm_health=LlmHealthState("openai_compatible", "gemma4:e4b"),
        llm_probe=None,
        event_loop_lag=lambda: 1.0,
        utc_now=lambda: datetime(2026, 9, 8, tzinfo=timezone.utc),
    )
    await service.sample_host_once()

    first, second = await asyncio.gather(service.latest_host(), service.latest_host())

    assert gpu_calls == 1
    assert first == second
    assert first["sequence"] == 1
    assert first["sampled_at"] == "2026-09-08T00:00:00+00:00"

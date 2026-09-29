from __future__ import annotations

import asyncio
import copy
import math
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal

import psutil


GpuSnapshot = dict[str, Any]
LlmStatus = Literal["unknown", "ready", "degraded", "unavailable"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _unavailable_gpu(reason: str) -> GpuSnapshot:
    return {
        "available": False,
        "name": None,
        "utilization_percent": None,
        "memory_used_bytes": None,
        "memory_total_bytes": None,
        "reason": reason,
    }


class NvidiaGpuSampler:
    """Collect one NVIDIA GPU sample without invoking a shell."""

    def __init__(
        self,
        *,
        create_process=asyncio.create_subprocess_exec,
        timeout_seconds: float = 2.0,
    ):
        self.create_process = create_process
        self.timeout_seconds = max(0.01, float(timeout_seconds))

    async def sample(self) -> GpuSnapshot:
        try:
            process = await self.create_process(
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return _unavailable_gpu("nvidia-smi not found")
        except OSError:
            return _unavailable_gpu("nvidia-smi could not be started")

        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout_seconds
            )
        except asyncio.CancelledError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            return _unavailable_gpu("nvidia-smi timed out")

        if process.returncode:
            return _unavailable_gpu(
                f"nvidia-smi exited with code {process.returncode}"
            )

        try:
            line = next(
                line.strip()
                for line in stdout.decode("utf-8", errors="replace").splitlines()
                if line.strip()
            )
            name, utilization, memory_used_mib, memory_total_mib = (
                value.strip() for value in line.split(",")
            )
            utilization_percent = float(utilization)
            memory_used = float(memory_used_mib)
            memory_total = float(memory_total_mib)
            if (
                not name
                or not all(
                    math.isfinite(value)
                    for value in (utilization_percent, memory_used, memory_total)
                )
                or not 0.0 <= utilization_percent <= 100.0
                or memory_used < 0.0
                or memory_total <= 0.0
                or memory_used > memory_total
            ):
                raise ValueError("invalid GPU values")
            memory_used_bytes = round(memory_used * 1024 * 1024)
            memory_total_bytes = round(memory_total * 1024 * 1024)
        except (StopIteration, ValueError, TypeError, OverflowError):
            return _unavailable_gpu("malformed nvidia-smi output")

        return {
            "available": True,
            "name": name,
            "utilization_percent": utilization_percent,
            "memory_used_bytes": memory_used_bytes,
            "memory_total_bytes": memory_total_bytes,
            "reason": None,
        }


class HostSampler:
    """Sample host and process utilization while retaining network baselines."""

    def __init__(
        self,
        *,
        psutil_module=psutil,
        gpu_sampler: NvidiaGpuSampler | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.psutil = psutil_module
        self.gpu_sampler = gpu_sampler or NvidiaGpuSampler()
        self.monotonic = monotonic
        self.process = self.psutil.Process()
        self._previous_network: tuple[float, int, int] | None = None

    async def sample(self, *, event_loop_lag_ms: float | None) -> dict[str, Any]:
        sampled_monotonic = self.monotonic()
        network = self.psutil.net_io_counters()
        rx_rate = 0.0
        tx_rate = 0.0
        if self._previous_network is not None:
            previous_at, previous_rx, previous_tx = self._previous_network
            elapsed = sampled_monotonic - previous_at
            if elapsed > 0:
                if network.bytes_recv >= previous_rx:
                    rx_rate = (network.bytes_recv - previous_rx) / elapsed
                if network.bytes_sent >= previous_tx:
                    tx_rate = (network.bytes_sent - previous_tx) / elapsed
        self._previous_network = (
            sampled_monotonic,
            int(network.bytes_recv),
            int(network.bytes_sent),
        )

        memory = self.psutil.virtual_memory()
        process_memory = self.process.memory_info()
        gpu = await self.gpu_sampler.sample()
        return {
            "cpu_percent": float(self.psutil.cpu_percent(interval=None)),
            "memory": {
                "used_bytes": int(memory.used),
                "total_bytes": int(memory.total),
                "percent": float(memory.percent),
            },
            "process": {
                "cpu_percent": float(self.process.cpu_percent(interval=None)),
                "rss_bytes": int(process_memory.rss),
            },
            "event_loop_lag_ms": (
                max(0.0, float(event_loop_lag_ms))
                if event_loop_lag_ms is not None
                else None
            ),
            "network": {
                "rx_bytes": int(network.bytes_recv),
                "tx_bytes": int(network.bytes_sent),
                "rx_bytes_per_second": max(0.0, rx_rate),
                "tx_bytes_per_second": max(0.0, tx_rate),
            },
            "gpu": gpu,
        }


class LlmHealthState:
    """Record probe and inference outcomes without retaining endpoint secrets."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.status: LlmStatus = "unknown"
        self.last_probe_at: str | None = None
        self.last_probe_latency_ms: float | None = None
        self.last_probe_error: str | None = None
        self.last_inference_at: str | None = None
        self.last_inference_latency_ms: float | None = None
        self.last_inference_error: str | None = None
        self.last_inference_success: bool | None = None

    def record_probe(
        self,
        *,
        success: bool,
        checked_at: str,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        self.last_probe_at = checked_at
        self.last_probe_latency_ms = max(0.0, float(latency_ms))
        self.last_probe_error = None if success else error
        self._refresh_status()

    def record_inference(
        self,
        *,
        success: bool,
        checked_at: str,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        self.last_inference_at = checked_at
        self.last_inference_latency_ms = max(0.0, float(latency_ms))
        self.last_inference_error = None if success else error
        self.last_inference_success = success
        self._refresh_status()

    def mark_last_inference_malformed(self, error: str) -> None:
        if self.last_inference_at is None:
            return
        self.last_inference_error = error
        self.last_inference_success = False
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self.last_probe_at is None:
            self.status = "unknown"
        elif self.last_probe_error is not None:
            self.status = "unavailable"
        elif self.last_inference_success is False:
            self.status = "degraded"
        else:
            self.status = "ready"

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "last_probe_at": self.last_probe_at,
            "last_probe_latency_ms": self.last_probe_latency_ms,
            "last_probe_error": self.last_probe_error,
            "last_inference_at": self.last_inference_at,
            "last_inference_latency_ms": self.last_inference_latency_ms,
            "last_inference_error": self.last_inference_error,
            "last_inference_success": self.last_inference_success,
        }


class SystemTelemetryService:
    """Refresh expensive telemetry in background tasks and serve cached snapshots."""

    def __init__(
        self,
        *,
        host_sampler: HostSampler,
        llm_health: LlmHealthState,
        llm_probe: Callable[[], Awaitable[None]] | None,
        event_loop_lag: Callable[[], float | None],
        host_interval_seconds: float = 1.0,
        probe_interval_seconds: float = 10.0,
        probe_timeout_seconds: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.host_sampler = host_sampler
        self.llm_health = llm_health
        self.llm_probe = llm_probe
        self.event_loop_lag = event_loop_lag
        self.host_interval_seconds = max(0.05, float(host_interval_seconds))
        self.probe_interval_seconds = max(0.05, float(probe_interval_seconds))
        self.probe_timeout_seconds = max(0.01, float(probe_timeout_seconds))
        self.monotonic = monotonic
        self.utc_now = utc_now
        self.sleep = sleep
        self._sample_lock = asyncio.Lock()
        self._latest_host: dict[str, Any] = {
            "sampled_at": None,
            "sequence": 0,
            "host": None,
        }
        self._tasks: list[asyncio.Task[None]] = []

    async def sample_host_once(self) -> None:
        async with self._sample_lock:
            host = await self.host_sampler.sample(
                event_loop_lag_ms=self.event_loop_lag()
            )
            self._latest_host = {
                "sampled_at": self.utc_now().isoformat(),
                "sequence": self._latest_host["sequence"] + 1,
                "host": host,
            }

    async def latest_host(self) -> dict[str, Any]:
        return copy.deepcopy(self._latest_host)

    async def probe_llm_once(self) -> None:
        if self.llm_probe is None:
            return
        started_at = self.monotonic()
        checked_at = self.utc_now().isoformat()
        try:
            await asyncio.wait_for(
                self.llm_probe(), timeout=self.probe_timeout_seconds
            )
        except Exception as exc:
            self.llm_health.record_probe(
                success=False,
                checked_at=checked_at,
                latency_ms=(self.monotonic() - started_at) * 1000.0,
                error=f"probe failed ({type(exc).__name__})",
            )
            return
        self.llm_health.record_probe(
            success=True,
            checked_at=checked_at,
            latency_ms=(self.monotonic() - started_at) * 1000.0,
        )

    async def start(self) -> None:
        if any(not task.done() for task in self._tasks):
            return
        self._tasks = [
            asyncio.create_task(self._host_loop(), name="system-telemetry-host"),
            asyncio.create_task(self._probe_loop(), name="system-telemetry-llm"),
        ]

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def _host_loop(self) -> None:
        while True:
            started_at = self.monotonic()
            try:
                await self.sample_host_once()
            except Exception:
                pass
            delay = max(
                0.0,
                self.host_interval_seconds - (self.monotonic() - started_at),
            )
            await self.sleep(delay)

    async def _probe_loop(self) -> None:
        while True:
            started_at = self.monotonic()
            await self.probe_llm_once()
            delay = max(
                0.0,
                self.probe_interval_seconds - (self.monotonic() - started_at),
            )
            await self.sleep(delay)

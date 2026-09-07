from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger("uvicorn.error")
MAX_MESSAGE_BYTES = 512
PROTOCOL_VERSION = 1


class ControlChannelError(RuntimeError):
    pass


@dataclass
class ControlChannelStats:
    connected: bool = False
    ready: bool = False
    reconnect_count: int = 0
    sent: int = 0
    acknowledged: int = 0
    rejected: int = 0
    stale_rejections: int = 0
    last_round_trip_ms: float | None = None
    last_error: str | None = None
    connected_at: float | None = None


class ControlChannelClient:
    """One persistent, versioned PC-to-ESP control connection."""

    def __init__(
        self,
        host: str,
        port: int = 82,
        *,
        heartbeat_interval_seconds: float = 1.0,
        command_timeout_seconds: float = 1.5,
    ):
        self.host = host
        self.port = port
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.session_id = uuid.uuid4().hex
        self.stats = ControlChannelStats()
        self.telemetry: dict[str, Any] = {}
        self.telemetry_received_at: float | None = None
        self._watchdog_fault_observed = False
        self._watchdog_recovered = False
        self._sequence = 0
        self._server_clock_offset_ms = 0.0
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._runner: asyncio.Task[None] | None = None
        self._heartbeat: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._stopping = False
        self._send_lock = asyncio.Lock()
        self._pending: dict[int, tuple[asyncio.Future[dict[str, Any]], float]] = {}
        self._connection_listeners: list[Callable[[bool], Awaitable[None] | None]] = []

    def add_connection_listener(
        self, listener: Callable[[bool], Awaitable[None] | None]
    ) -> None:
        self._connection_listeners.append(listener)

    async def start(self) -> None:
        if self._runner is not None:
            return
        self._stopping = False
        self._runner = asyncio.create_task(self._run(), name="esp-control-channel")

    async def shutdown(self) -> None:
        self._stopping = True
        self._ready.clear()
        tasks = [task for task in (self._runner, self._heartbeat) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._close()
        self._runner = None
        self._heartbeat = None

    async def wait_ready(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(
            self._ready.wait(),
            timeout=timeout or self.command_timeout_seconds,
        )

    async def send(
        self,
        command_type: str,
        *,
        ttl_ms: int = 1000,
        timeout: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        await self.wait_ready(timeout)
        async with self._send_lock:
            if not self._writer or self._writer.is_closing():
                raise ControlChannelError("control channel disconnected")
            self._sequence += 1
            sequence = self._sequence
            message = {
                "v": PROTOCOL_VERSION,
                "type": command_type,
                "session": self.session_id,
                "seq": sequence,
                "ttl_ms": max(1, min(int(ttl_ms), 5000)),
                "expires_at_ms": round(
                    time.monotonic() * 1000.0
                    + self._server_clock_offset_ms
                    + max(1, min(int(ttl_ms), 5000))
                )
                % (2**32),
                **payload,
            }
            encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode()
            if len(encoded) > MAX_MESSAGE_BYTES + 1:
                raise ControlChannelError("control message exceeds 512 bytes")
            loop = asyncio.get_running_loop()
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[sequence] = (future, time.monotonic())
            try:
                self._writer.write(encoded)
                await self._writer.drain()
                self.stats.sent += 1
            except Exception:
                self._pending.pop(sequence, None)
                raise

        try:
            result = await asyncio.wait_for(
                future,
                timeout=timeout or self.command_timeout_seconds,
            )
        except TimeoutError as exc:
            self._pending.pop(sequence, None)
            raise ControlChannelError(
                f"control acknowledgement timeout: {command_type}"
            ) from exc
        if result.get("status") != "accepted":
            raise ControlChannelError(
                f"control command rejected: {result.get('status', 'unknown')}"
            )
        return {"ok": True, **result.get("state", {}), "ack": result}

    async def _run(self) -> None:
        backoff = 0.25
        has_connected = False
        while not self._stopping:
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self.host, self.port
                )
                if has_connected:
                    self.stats.reconnect_count += 1
                has_connected = True
                self.stats.connected = True
                self.stats.connected_at = time.time()
                self.stats.last_error = None
                self._sequence = 0
                await self._write_hello()
                hello = await asyncio.wait_for(
                    self._read_message(), timeout=self.command_timeout_seconds
                )
                if (
                    hello.get("type") != "hello_ack"
                    or hello.get("status") != "ready"
                    or hello.get("protocol") != PROTOCOL_VERSION
                    or hello.get("session") != self.session_id
                ):
                    raise ControlChannelError(
                        f"firmware protocol mismatch: {hello!r}"
                    )
                self._store_telemetry(hello.get("state"))
                self._server_clock_offset_ms = (
                    float(hello.get("uptime_ms") or 0.0)
                    - time.monotonic() * 1000.0
                )
                self.stats.ready = True
                self._ready.set()
                reader_task = asyncio.create_task(
                    self._read_loop(), name="esp-control-reader"
                )
                await self._notify_connection(True)
                backoff = 0.25
                self._heartbeat = asyncio.create_task(
                    self._heartbeat_loop(), name="esp-control-heartbeat"
                )
                await reader_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "control.disconnected stage=connection error_class=%s error=%s",
                    type(exc).__name__,
                    exc,
                )
            finally:
                was_ready = self.stats.ready
                self.stats.connected = False
                self.stats.ready = False
                self._ready.clear()
                if self._heartbeat is not None:
                    self._heartbeat.cancel()
                    await asyncio.gather(self._heartbeat, return_exceptions=True)
                    self._heartbeat = None
                self._fail_pending(ControlChannelError("control channel disconnected"))
                await self._close()
                if was_ready:
                    await self._notify_connection(False)
            if not self._stopping:
                await asyncio.sleep(backoff)
                backoff = min(5.0, backoff * 2.0)

    async def _write_hello(self) -> None:
        assert self._writer is not None
        hello = {
            "v": PROTOCOL_VERSION,
            "type": "hello",
            "session": self.session_id,
            "client": "pc_brain",
            "client_version": "0.2.0",
            "required_protocol": PROTOCOL_VERSION,
        }
        self._writer.write(
            (json.dumps(hello, separators=(",", ":")) + "\n").encode()
        )
        await self._writer.drain()

    async def _read_loop(self) -> None:
        while not self._stopping:
            message = await self._read_message()
            message_type = message.get("type")
            if message_type == "telemetry":
                self._store_telemetry(message.get("state"))
                self.telemetry["last_seq"] = message.get("last_seq")
                continue
            if message_type != "ack":
                continue
            sequence = int(message.get("seq") or 0)
            pending = self._pending.pop(sequence, None)
            if pending is None:
                continue
            future, sent_at = pending
            self._store_telemetry(message.get("state"))
            self.stats.last_round_trip_ms = (
                time.monotonic() - sent_at
            ) * 1000.0
            status = message.get("status")
            if status == "accepted":
                self.stats.acknowledged += 1
            else:
                self.stats.rejected += 1
                if status == "stale":
                    self.stats.stale_rejections += 1
            if not future.done():
                future.set_result(message)

    async def _read_message(self) -> dict[str, Any]:
        assert self._reader is not None
        line = await self._reader.readline()
        if not line:
            raise ControlChannelError("firmware closed control channel")
        if len(line) > MAX_MESSAGE_BYTES + 1:
            raise ControlChannelError("firmware message exceeds 512 bytes")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ControlChannelError("invalid firmware JSON") from exc
        if not isinstance(message, dict) or message.get("v") != PROTOCOL_VERSION:
            raise ControlChannelError("invalid firmware protocol envelope")
        return message

    async def _heartbeat_loop(self) -> None:
        while True:
            started = time.monotonic()
            try:
                await self.send("ping", ttl_ms=1500)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "control.heartbeat_failed stage=heartbeat error_class=%s error=%s",
                    type(exc).__name__,
                    exc,
                )
                if self._writer:
                    self._writer.close()
                return
            await asyncio.sleep(
                max(0.0, self.heartbeat_interval_seconds - (time.monotonic() - started))
            )

    async def _notify_connection(self, connected: bool) -> None:
        for listener in tuple(self._connection_listeners):
            try:
                result = listener(connected)
                if result is not None:
                    await result
            except Exception:
                logger.exception("control.connection_listener_failed")

    def _store_telemetry(self, sample: Any) -> None:
        if not isinstance(sample, dict) or not sample:
            return
        previous_fault = self._watchdog_fault_observed
        self.telemetry = dict(sample)
        self.telemetry_received_at = time.monotonic()
        heartbeat_fault = self.telemetry.get(
            "heartbeat_fault",
            self.telemetry.get(
                "brain_heartbeat_fault",
                self.telemetry.get("fault"),
            ),
        )
        if heartbeat_fault is True:
            self._watchdog_fault_observed = True
        elif heartbeat_fault is False and previous_fault:
            self._watchdog_fault_observed = False
            self._watchdog_recovered = True
        if self.telemetry.get("heartbeat_recovered") is True:
            self._watchdog_recovered = True

    def consume_watchdog_recovery(self) -> bool:
        recovered = self._watchdog_recovered
        self._watchdog_recovered = False
        return recovered

    def _fail_pending(self, exc: Exception) -> None:
        pending, self._pending = self._pending, {}
        for future, _ in pending.values():
            if not future.done():
                future.set_exception(exc)

    async def _close(self) -> None:
        writer, self._writer = self._writer, None
        self._reader = None
        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    def status(self) -> dict[str, Any]:
        return {
            **vars(self.stats),
            "host": self.host,
            "port": self.port,
            "protocol": PROTOCOL_VERSION,
            "session": self.session_id,
            "telemetry": self.telemetry,
            "telemetry_received_at": self.telemetry_received_at,
            "watchdog_recovered": self._watchdog_recovered,
        }


@dataclass
class ActuatorBrokerStats:
    queued: int = 0
    sent: int = 0
    coalesced: int = 0
    suppressed_tracking: int = 0
    failures: int = 0
    last_error: str | None = None


@dataclass
class _HeadTarget:
    pan: int
    tilt: int
    source: str
    requested_at: float = field(default_factory=time.monotonic)
    waiters: list[asyncio.Future[dict[str, Any]]] = field(default_factory=list)
    authorization: Callable[[], bool] | None = None


class ActuatorBroker:
    """The only PC component allowed to produce ESP actuator commands."""

    def __init__(
        self,
        channel: ControlChannelClient,
        *,
        tracking_rate_hz: float = 4.0,
        manual_rate_hz: float = 10.0,
        minimum_head_change_degrees: float = 2.0,
        manual_lease_seconds: float = 3.0,
    ):
        self.channel = channel
        self.tracking_rate_hz = tracking_rate_hz
        self.manual_rate_hz = manual_rate_hz
        self.minimum_head_change_degrees = minimum_head_change_degrees
        self.manual_lease_seconds = manual_lease_seconds
        self.stats = ActuatorBrokerStats()
        self._head_event = asyncio.Event()
        self._head_target: _HeadTarget | None = None
        self._head_worker: asyncio.Task[None] | None = None
        self._last_head_sent: tuple[int, int] | None = None
        self._last_head_sent_at = 0.0
        self._manual_lease_until = 0.0
        self._desired_head: tuple[int, int] = (90, 90)
        self._desired_eyes: tuple[str, int] = ("neutral", 0)
        channel.add_connection_listener(self._on_connection)

    async def start(self) -> None:
        await self.channel.start()
        if self._head_worker is None:
            self._head_worker = asyncio.create_task(
                self._head_loop(), name="actuator-head-broker"
            )

    async def shutdown(self) -> None:
        if self._head_worker is not None:
            self._head_worker.cancel()
            await asyncio.gather(self._head_worker, return_exceptions=True)
            self._head_worker = None
        await self.channel.shutdown()

    def tracking_allowed(self) -> bool:
        return time.monotonic() >= self._manual_lease_until

    def manual_lease_remaining(self) -> float:
        return max(0.0, self._manual_lease_until - time.monotonic())

    async def head_target(
        self,
        pan: int,
        tilt: int,
        *,
        source: str,
        wait_for_ack: bool = True,
        authorization: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if source == "tracking" and not self.tracking_allowed():
            self.stats.suppressed_tracking += 1
            return {"ok": False, "skipped": "manual control lease active"}
        if source != "tracking":
            self._manual_lease_until = time.monotonic() + self.manual_lease_seconds
        pan = max(55, min(135, int(pan)))
        tilt = max(35, min(115, int(tilt)))
        self._desired_head = (pan, tilt)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        target = _HeadTarget(
            pan=pan,
            tilt=tilt,
            source=source,
            authorization=authorization,
        )
        target.waiters.append(future)
        previous = self._head_target
        if previous is not None:
            target.waiters[:0] = previous.waiters
            self.stats.coalesced += 1
        self._head_target = target
        self.stats.queued += 1
        self._head_event.set()
        if not wait_for_ack:
            return {"ok": True, "queued": True, "pan": pan, "tilt": tilt}
        return await future

    async def drive(
        self, direction: str, speed: int, duration_ms: int
    ) -> dict[str, Any]:
        self._manual_lease_until = time.monotonic() + self.manual_lease_seconds
        return await self._send(
            "drive",
            ttl_ms=max(250, min(duration_ms + 500, 2000)),
            direction=direction,
            speed=max(0, min(int(speed), 255)),
            duration_ms=max(0, min(int(duration_ms), 1000)),
        )

    async def tracking_drive(
        self,
        direction: str,
        speed: int,
        duration_ms: int,
        authorization: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if not self.tracking_allowed():
            self.stats.suppressed_tracking += 1
            return {"ok": False, "skipped": "manual control lease active"}
        if authorization is not None and not authorization():
            return {"ok": False, "skipped": "tracking authorization expired"}
        await self.channel.wait_ready()
        if authorization is not None and not authorization():
            return {"ok": False, "skipped": "tracking authorization expired"}
        return await self._send(
            "drive",
            ttl_ms=max(250, min(duration_ms + 500, 2000)),
            direction=direction,
            speed=max(0, min(int(speed), 255)),
            duration_ms=max(0, min(int(duration_ms), 1000)),
        )

    async def stop(self) -> dict[str, Any]:
        self._manual_lease_until = time.monotonic() + self.manual_lease_seconds
        return await self._send("stop", ttl_ms=1000)

    async def eyes(self, expression: str, duration_ms: int = 0) -> dict[str, Any]:
        self._desired_eyes = (expression, duration_ms)
        return await self._send(
            "eyes",
            ttl_ms=1000,
            expression=expression,
            duration_ms=max(0, min(int(duration_ms), 10000)),
        )

    async def _head_loop(self) -> None:
        while True:
            await self._head_event.wait()
            while self._head_target is not None:
                target, self._head_target = self._head_target, None
                interval = 1.0 / (
                    self.tracking_rate_hz
                    if target.source == "tracking"
                    else self.manual_rate_hz
                )
                delay = self._last_head_sent_at + interval - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                try:
                    if (
                        target.authorization is not None
                        and not target.authorization()
                    ):
                        result = {
                            "ok": False,
                            "skipped": "tracking authorization expired",
                        }
                    elif (
                        target.source == "tracking"
                        and not self.tracking_allowed()
                    ):
                        result = {"ok": False, "skipped": "manual control lease active"}
                        self.stats.suppressed_tracking += 1
                    elif (
                        target.source == "tracking"
                        and self._last_head_sent is not None
                        and math.hypot(
                            target.pan - self._last_head_sent[0],
                            target.tilt - self._last_head_sent[1],
                        )
                        < self.minimum_head_change_degrees
                    ):
                        result = {"ok": True, "coalesced": True}
                        self.stats.coalesced += 1
                    else:
                        result = await self._send(
                            "head_target",
                            ttl_ms=1000,
                            pan=target.pan,
                            tilt=target.tilt,
                        )
                        if result.get("ok"):
                            self._last_head_sent = (target.pan, target.tilt)
                            self._last_head_sent_at = time.monotonic()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                for future in target.waiters:
                    if not future.done():
                        future.set_result(result)
            self._head_event.clear()
            if self._head_target is not None:
                self._head_event.set()

    async def _send(self, command_type: str, **payload: Any) -> dict[str, Any]:
        try:
            result = await self.channel.send(command_type, **payload)
            self.stats.sent += 1
            return result
        except Exception as exc:
            self.stats.failures += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            raise

    async def _on_connection(self, connected: bool) -> None:
        if not connected:
            self._last_head_sent = None
            return
        # Head and eyes are latest-value state and are safe to restore. Body
        # commands are intentionally absent: movement must never replay.
        pan, tilt = self._desired_head
        expression, duration_ms = self._desired_eyes
        await self.channel.send(
            "head_target", ttl_ms=1000, pan=pan, tilt=tilt
        )
        await self.channel.send(
            "eyes",
            ttl_ms=1000,
            expression=expression,
            duration_ms=duration_ms,
        )

    def status(self) -> dict[str, Any]:
        return {
            **vars(self.stats),
            "queue_depth": 1 if self._head_target is not None else 0,
            "manual_lease_remaining_seconds": self.manual_lease_remaining(),
            "desired_head": {
                "pan": self._desired_head[0],
                "tilt": self._desired_head[1],
            },
            "desired_eyes": self._desired_eyes[0],
        }

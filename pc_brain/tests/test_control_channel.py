from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.control_channel import ActuatorBroker, ControlChannelClient
from fake_esp import FakeEspControlServer


@pytest.mark.anyio
async def test_handshake_ack_telemetry_and_bounded_commands():
    fake = FakeEspControlServer()
    await fake.start()
    channel = ControlChannelClient(
        "127.0.0.1",
        fake.port,
        heartbeat_interval_seconds=10,
    )
    broker = ActuatorBroker(channel)
    await broker.start()
    try:
        await channel.wait_ready(1)
        result = await broker.head_target(105, 80, source="manual")
        assert result["ok"] is True
        head_command = next(
            command
            for command in reversed(fake.commands)
            if command["type"] == "head_target"
            and command["pan"] == 105
            and command["tilt"] == 80
        )
        assert len(
            __import__("json").dumps(head_command, separators=(",", ":")).encode()
        ) <= 512
    finally:
        await broker.shutdown()
        await fake.stop()


@pytest.mark.anyio
async def test_reconnect_restores_latest_state_but_never_replays_drive():
    fake = FakeEspControlServer()
    await fake.start()
    channel = ControlChannelClient(
        "127.0.0.1",
        fake.port,
        heartbeat_interval_seconds=10,
    )
    broker = ActuatorBroker(channel)
    await broker.start()
    try:
        await channel.wait_ready(1)
        await broker.head_target(110, 75, source="manual")
        await broker.eyes("happy")
        await broker.drive("left", 140, 300)
        drives_before = sum(command["type"] == "drive" for command in fake.commands)
        pings_before = sum(command["type"] == "ping" for command in fake.commands)
        await fake.disconnect_clients()
        for _ in range(30):
            if channel.stats.reconnect_count and channel.stats.ready:
                break
            await asyncio.sleep(0.1)
        assert channel.stats.ready
        assert sum(command["type"] == "drive" for command in fake.commands) == drives_before
        assert sum(command["type"] == "ping" for command in fake.commands) > pings_before
        assert any(
            command["type"] == "head_target"
            and command["pan"] == 110
            and command["tilt"] == 75
            for command in fake.commands
        )
        assert any(
            command["type"] == "eyes" and command["expression"] == "happy"
            for command in fake.commands
        )
    finally:
        await broker.shutdown()
        await fake.stop()


@pytest.mark.anyio
async def test_fake_firmware_rejects_stale_sequence_and_expired_ttl():
    fake = FakeEspControlServer()
    await fake.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", fake.port)
    try:
        writer.write(b'{"v":1,"type":"hello","session":"test"}\n')
        await writer.drain()
        await reader.readline()

        valid = {
            "v": 1,
            "type": "ping",
            "session": "test",
            "seq": 1,
            "ttl_ms": 1000,
            "expires_at_ms": round(time.monotonic() * 1000) + 1000,
        }
        writer.write((json.dumps(valid) + "\n").encode())
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "accepted"

        writer.write((json.dumps(valid) + "\n").encode())
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "stale"

        expired = {
            **valid,
            "seq": 2,
            "expires_at_ms": round(time.monotonic() * 1000) - 1,
        }
        writer.write((json.dumps(expired) + "\n").encode())
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "expired"
    finally:
        writer.close()
        await writer.wait_closed()
        await fake.stop()


class RecordingChannel:
    def __init__(self):
        self.commands = []
        self.listeners = []

    def add_connection_listener(self, listener):
        self.listeners.append(listener)

    async def start(self):
        return None

    async def shutdown(self):
        return None

    async def send(self, command_type, **payload):
        self.commands.append((command_type, payload))
        return {"ok": True, "status": "accepted", "state": {}}

    def status(self):
        return {}


def test_control_channel_tracks_sample_time_and_consumes_real_watchdog_recovery(monkeypatch):
    channel = ControlChannelClient("robot.local")
    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)

    channel._store_telemetry({"heartbeat_fault": True, "eyes": "fault"})
    now = 101.5
    channel._store_telemetry({"heartbeat_fault": False, "eyes": "neutral"})

    assert channel.status()["telemetry_received_at"] == 101.5
    assert channel.consume_watchdog_recovery() is True
    assert channel.consume_watchdog_recovery() is False


@pytest.mark.anyio
async def test_tracking_targets_are_coalesced_and_manual_lease_suppresses_tracking():
    channel = RecordingChannel()
    broker = ActuatorBroker(
        channel,
        tracking_rate_hz=4,
        manual_rate_hz=10,
        manual_lease_seconds=3,
    )
    await broker.start()
    try:
        first = await broker.head_target(
            92, 90, source="tracking", wait_for_ack=False
        )
        second = await broker.head_target(
            110, 90, source="tracking", wait_for_ack=False
        )
        assert first["queued"] and second["queued"]
        await asyncio.sleep(0.3)
        assert [
            payload["pan"]
            for command, payload in channel.commands
            if command == "head_target"
        ][-1] == 110

        await broker.head_target(100, 90, source="manual")
        suppressed = await broker.head_target(120, 90, source="tracking")
        assert suppressed["ok"] is False
        assert broker.stats.suppressed_tracking == 1
    finally:
        await broker.shutdown()

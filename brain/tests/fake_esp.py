from __future__ import annotations

import asyncio
import json
import time
from typing import Any


class FakeEspControlServer:
    """Protocol-v1 fake used by transport and reconnect tests."""

    def __init__(self):
        self.server: asyncio.Server | None = None
        self.port = 0
        self.commands: list[dict[str, Any]] = []
        self.connections = 0
        self.last_sequence = 0
        self.state = {
            "movement": "stop",
            "speed": 0,
            "pan_actual": 90,
            "tilt_actual": 90,
            "pan_target": 90,
            "tilt_target": 90,
            "eyes": "neutral",
            "wifi_rssi": -45,
            "camera": True,
            "fault": False,
        }
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._client, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])

    async def stop(self) -> None:
        for writer in tuple(self._writers):
            writer.close()
            await writer.wait_closed()
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def disconnect_clients(self) -> None:
        for writer in tuple(self._writers):
            writer.close()
            await writer.wait_closed()

    async def _client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        self._writers.add(writer)
        session = ""
        self.last_sequence = 0
        try:
            while line := await reader.readline():
                message = json.loads(line)
                if message.get("type") == "hello":
                    session = str(message["session"])
                    await self._send(
                        writer,
                        {
                            "v": 1,
                            "type": "hello_ack",
                            "status": "ready",
                            "session": session,
                            "protocol": 1,
                            "uptime_ms": round(time.monotonic() * 1000),
                            "state": self.state,
                        },
                    )
                    continue
                sequence = int(message.get("seq", 0))
                status = "accepted"
                if sequence <= self.last_sequence:
                    status = "stale"
                elif int(message.get("expires_at_ms", 0)) <= round(
                    time.monotonic() * 1000
                ):
                    status = "expired"
                else:
                    self.last_sequence = sequence
                    self.commands.append(message)
                    self._apply(message)
                await self._send(
                    writer,
                    {
                        "v": 1,
                        "type": "ack",
                        "session": session,
                        "seq": sequence,
                        "status": status,
                        "state": self.state,
                    },
                )
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self._writers.discard(writer)
            writer.close()

    def _apply(self, message: dict[str, Any]) -> None:
        message_type = message["type"]
        if message_type == "head_target":
            self.state["pan_target"] = message["pan"]
            self.state["tilt_target"] = message["tilt"]
        elif message_type == "drive":
            self.state["movement"] = message["direction"]
            self.state["speed"] = message["speed"]
        elif message_type == "stop":
            self.state["movement"] = "stop"
        elif message_type == "eyes":
            self.state["eyes"] = message["expression"]

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
        writer.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
        await writer.drain()

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx
import websocket


def gpu_snapshot() -> dict:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(command, text=True, timeout=10).strip()
        name, used, total, utilization = [part.strip() for part in output.split(",", 3)]
        return {
            "name": name,
            "memory_used_mb": int(used),
            "memory_total_mb": int(total),
            "utilization_percent": int(utilization),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"available": False}


def websocket_url(brain_url: str) -> str:
    parsed = urlparse(brain_url)
    return urlunparse(("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/v1/realtime", "", "", ""))


def wait_for_session(connection) -> None:
    deadline = time.perf_counter() + 20
    while time.perf_counter() < deadline:
        event = json.loads(connection.recv())
        if event.get("type") == "session.created":
            return
        if event.get("type") == "error":
            raise RuntimeError(event.get("error"))
    raise TimeoutError("Realtime session was not created")


def measure_turn(connection, text: str) -> dict:
    started = time.perf_counter()
    connection.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }
        )
    )
    connection.send(json.dumps({"type": "response.create"}))
    first_audio = None
    transcript = ""
    while True:
        event = json.loads(connection.recv())
        event_type = event.get("type", "")
        if event_type == "response.output_audio.delta" and first_audio is None:
            first_audio = time.perf_counter()
        if event_type.endswith("transcript.done"):
            transcript = event.get("transcript", transcript)
        if event_type == "response.done":
            finished = time.perf_counter()
            return {
                "first_audio_seconds": round(first_audio - started, 3) if first_audio else None,
                "total_seconds": round(finished - started, 3),
                "transcript": transcript,
            }
        if event_type == "error":
            raise RuntimeError(event.get("error"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Gate 1 warm/cold realtime latency and GPU use.")
    parser.add_argument("--brain-url", default="http://localhost:8080")
    parser.add_argument("--output", type=Path, default=Path("pc_brain/data/gate1-benchmark.json"))
    args = parser.parse_args()

    health = httpx.get(f"{args.brain_url.rstrip('/')}/health", timeout=10).json()
    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "brain_url": args.brain_url,
        "health": health,
        "before": gpu_snapshot(),
        "turns": [],
    }
    connection = websocket.create_connection(
        websocket_url(args.brain_url),
        timeout=180,
        http_proxy_host=None,
    )
    try:
        wait_for_session(connection)
        result["turns"].append({"kind": "cold", **measure_turn(connection, "Say ready in one short sentence.")})
        result["turns"].append({"kind": "warm", **measure_turn(connection, "Say ready again in one short sentence.")})
    finally:
        connection.close()
    result["after"] = gpu_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

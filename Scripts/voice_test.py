from __future__ import annotations

import argparse
import base64
import json
import socket
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import os


SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
BLOCK_SECONDS = 0.04
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_SECONDS)


def default_instructions() -> str:
    return (
        "You are Robit, a small helpful home robot. "
        "Talk like Rocky from Project Hail Mary. "
        "Be concise and use plain spoken text only."
    )


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


class RealtimeVoiceTester:
    def __init__(self, url: str, voice: str, instructions: str, sidecar_log: Path | None, connect_timeout: float):
        self.url = url
        self.voice = voice
        self.instructions = instructions
        self.sidecar_log = sidecar_log
        self.connect_timeout = connect_timeout
        self.ws_app: object | None = None
        self.connected = threading.Event()
        self.stopping = threading.Event()
        self.send_lock = threading.Lock()
        self.playback = bytearray()
        self.playback_lock = threading.Lock()
        self.user_transcript = ""
        self.assistant_transcript = ""
        self.last_error = ""
        self.last_log_offset = 0
        self.last_log_report = 0.0
        self.suppress_mic_until = 0.0

    def session_update(self) -> dict:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self.instructions,
                "audio": {
                    "output": {
                        "voice": self.voice,
                    },
                },
            },
        }

    def send(self, payload: dict) -> None:
        if not self.ws_app or not self.connected.is_set():
            return
        with self.send_lock:
            self.ws_app.send(json.dumps(payload))

    def on_open(self, ws) -> None:
        self.connected.set()
        self.send(self.session_update())
        print(f"[voice-test] connected {self.url}")
        print("[voice-test] talk into your microphone; press Ctrl+C to stop")

    def on_close(self, ws, close_status_code, close_msg) -> None:
        self.connected.clear()
        if not self.stopping.is_set() and close_status_code is not None:
            print(f"[voice-test] disconnected status={close_status_code} message={close_msg}")

    def on_error(self, ws, error) -> None:
        self.last_error = str(error)

    def on_message(self, ws, message: str) -> None:
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            print(f"[voice-test] non-json event: {message[:120]}")
            return

        event_type = event.get("type", "")
        if event_type == "session.created":
            self.send(self.session_update())
            return
        if event_type == "input_audio_buffer.speech_started":
            with self.playback_lock:
                self.playback.clear()
            print("[voice-test] listening")
            return
        if event_type == "input_audio_buffer.speech_stopped":
            print("[voice-test] thinking")
            return
        if event_type == "response.output_audio.delta":
            audio = event.get("delta")
            if audio:
                with self.playback_lock:
                    self.playback.extend(base64.b64decode(audio))
                    self.suppress_mic_until = time.monotonic() + 1.0
            return
        if "input_audio_transcription" in event_type and (event.get("delta") or event.get("transcript")):
            if event_type.endswith(".delta"):
                self.user_transcript += event.get("delta", "")
            else:
                text = event.get("transcript") or self.user_transcript
                if text:
                    print(f"[you] {text}")
                self.user_transcript = ""
            return
        if "transcript" in event_type and (event.get("delta") or event.get("transcript")):
            if event_type.endswith(".delta"):
                self.assistant_transcript += event.get("delta", "")
            else:
                text = event.get("transcript") or self.assistant_transcript
                if text:
                    print(f"[robit] {text}")
                self.assistant_transcript = ""
            return
        if event_type == "error":
            print(f"[voice-test][server-error] {event.get('error') or event}")

    def input_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[voice-test][mic] {status}", file=sys.stderr)
        if not self.connected.is_set():
            return
        with self.playback_lock:
            robit_is_speaking = bool(self.playback) or time.monotonic() < self.suppress_mic_until
        if robit_is_speaking:
            return
        audio = base64.b64encode(bytes(indata)).decode("ascii")
        self.send({"type": "input_audio_buffer.append", "audio": audio})

    def output_callback(self, outdata, frames, time_info, status) -> None:
        if status:
            print(f"[voice-test][speaker] {status}", file=sys.stderr)
        needed = frames * CHANNELS * SAMPLE_WIDTH_BYTES
        with self.playback_lock:
            chunk = self.playback[:needed]
            del self.playback[: len(chunk)]
            if chunk:
                self.suppress_mic_until = time.monotonic() + 1.0
        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))
        outdata[:] = chunk

    def wait_for_realtime_port(self, deadline: float) -> bool:
        parsed = urlparse(self.url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        printed_wait = False
        while time.monotonic() < deadline and not self.stopping.is_set():
            if not printed_wait:
                print(f"[voice-test] waiting up to {self.connect_timeout:.0f}s for realtime server at {self.url}")
                printed_wait = True
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except OSError as exc:
                self.last_error = str(exc)
                self.report_sidecar_progress()
                time.sleep(1.0)
        return False

    def report_sidecar_progress(self) -> None:
        if not self.sidecar_log or not self.sidecar_log.exists():
            return
        now = time.monotonic()
        if now - self.last_log_report < 5.0:
            return
        self.last_log_report = now
        try:
            text = self.sidecar_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if len(text) <= self.last_log_offset:
            return
        new_text = text[self.last_log_offset :]
        self.last_log_offset = len(text)
        lines = [line for line in new_text.splitlines() if line.strip()]
        for line in lines[-8:]:
            print(f"[sidecar] {line}")

    def run(self) -> int:
        import sounddevice as sd
        import websocket

        deadline = time.monotonic() + self.connect_timeout
        if not self.wait_for_realtime_port(deadline):
            print(f"[voice-test][error] timed out waiting for realtime server at {self.url}")
            if self.last_error:
                print(f"[voice-test][last-error] {self.last_error}")
            if self.sidecar_log:
                print(f"[voice-test] sidecar log: {self.sidecar_log}")
            return 1

        self.ws_app = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        ws_thread.start()
        while not self.connected.is_set() and ws_thread.is_alive() and time.monotonic() < deadline:
            self.report_sidecar_progress()
            time.sleep(0.05)

        if not self.connected.is_set():
            print(f"[voice-test][error] timed out connecting to {self.url}")
            if self.last_error:
                print(f"[voice-test][last-error] {self.last_error}")
            if self.sidecar_log:
                print(f"[voice-test] sidecar log: {self.sidecar_log}")
                if self.sidecar_log.exists():
                    lines = self.sidecar_log.read_text(encoding="utf-8", errors="replace").splitlines()
                    for line in lines[-30:]:
                        print(f"[sidecar] {line}")
            return 1

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=self.input_callback,
        ), sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=self.output_callback,
        ):
            while not self.stopping.is_set():
                time.sleep(0.1)
        return 0

    def stop(self) -> None:
        self.stopping.set()
        if self.ws_app:
            self.ws_app.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    load_env_file(root / "pc_brain" / ".env")
    parser = argparse.ArgumentParser(description="Standalone Robit realtime voice tester.")
    parser.add_argument(
        "--url",
        default=os.getenv("ROBIT_REALTIME_WS_URL", "ws://localhost:7861/v1/realtime"),
        help="Realtime WebSocket URL.",
    )
    parser.add_argument(
        "--voice",
        default=os.getenv("ROBIT_REALTIME_VOICE", "serena"),
        help="Realtime output voice name.",
    )
    parser.add_argument(
        "--instructions",
        default=os.getenv("ROBIT_REALTIME_INSTRUCTIONS", default_instructions()),
        help="Session instructions sent to the realtime sidecar.",
    )
    parser.add_argument(
        "--sidecar-log",
        type=Path,
        default=None,
        help="Optional sidecar log to print when connection fails.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=float(os.getenv("ROBIT_VOICE_TEST_CONNECT_TIMEOUT", "600")),
        help="Seconds to wait for the realtime sidecar to finish cold-start model downloads and open its websocket.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tester = RealtimeVoiceTester(
        url=args.url,
        voice=args.voice,
        instructions=args.instructions,
        sidecar_log=args.sidecar_log,
        connect_timeout=args.connect_timeout,
    )

    def handle_stop(signum, frame) -> None:
        tester.stop()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    try:
        return tester.run()
    except KeyboardInterrupt:
        tester.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

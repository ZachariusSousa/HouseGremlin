from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from logging import Formatter, getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "pc_brain"
TRACKING = ROOT / "pc_tracking"
DATA = Path(os.getenv("ROBIT_DATA_DIR", str(BRAIN / "data"))).expanduser()
if not DATA.is_absolute():
    DATA = BRAIN / DATA
LOGS = DATA / "logs"


def cached_snapshot_or_model_id(model_id: str, cache_root: Path | None = None) -> str:
    """Prefer the checked-out main snapshot when a Hugging Face model is cached."""
    cache = cache_root or Path(
        os.getenv("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    )
    model_cache = cache / f"models--{model_id.replace('/', '--')}"
    ref = model_cache / "refs" / "main"
    try:
        revision = ref.read_text(encoding="utf-8").strip()
    except OSError:
        return model_id
    snapshot = model_cache / "snapshots" / revision
    return str(snapshot) if snapshot.is_dir() else model_id


def http_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def tcp_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_ready(name: str, readiness: Callable[[], bool], process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited during startup with code {process.returncode}")
        if readiness():
            print(f"[supervisor] {name} ready", flush=True)
            return
        time.sleep(0.5)
    raise TimeoutError(f"{name} did not become ready within {timeout:.0f}s")


@dataclass
class Service:
    name: str
    command: list[str]
    cwd: Path
    environment: dict[str, str]
    readiness: Callable[[], bool]
    startup_timeout: float
    restart_limit: int = 3
    restart_count: int = 0
    optional: bool = False
    process: subprocess.Popen | None = None
    output_thread: threading.Thread | None = None
    _handler: RotatingFileHandler | None = field(default=None, init=False)

    def start(self) -> None:
        LOGS.mkdir(parents=True, exist_ok=True)
        if self.optional:
            model_args = [a for a in self.command[1:] if a.endswith(".gguf")]
            if model_args and not Path(model_args[0]).exists():
                print(
                    f"[supervisor][warn] optional {self.name} skipped: "
                    f"model missing ({model_args[0]})",
                    flush=True,
                )
                return
        if self._handler is not None:
            self._handler.close()
        self._handler = RotatingFileHandler(
            LOGS / f"{self.name}.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=4,
            encoding="utf-8",
        )
        self._handler.setFormatter(Formatter("%(asctime)s %(message)s"))
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0
        )
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self.output_thread = threading.Thread(
            target=self._copy_output,
            name=f"{self.name}-log",
            daemon=True,
        )
        self.output_thread.start()
        print(
            f"[supervisor] starting {self.name}; log={LOGS / f'{self.name}.log'}",
            flush=True,
        )
        if self.optional:
            try:
                wait_ready(
                    self.name,
                    self.readiness,
                    self.process,
                    self.startup_timeout,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(
                    f"[supervisor][warn] optional {self.name} not ready "
                    f"({exc}); continuing without it",
                    flush=True,
                )
        else:
            wait_ready(
                self.name,
                self.readiness,
                self.process,
                self.startup_timeout,
            )

    def _copy_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        log = getLogger(f"robit.service.{self.name}.{id(self)}")
        log.propagate = False
        log.setLevel("INFO")
        if self._handler:
            log.addHandler(self._handler)
        for line in self.process.stdout:
            log.info(line.rstrip())
        if self._handler:
            self._handler.flush()

    def stop(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._handler:
            self._handler.close()

    def restart_if_failed(self) -> bool:
        if self.process is None or self.process.poll() is None:
            return True
        if self.restart_count >= self.restart_limit:
            print(
                f"[supervisor][error] {self.name} exhausted "
                f"{self.restart_limit} restarts",
                flush=True,
            )
            return False
        self.restart_count += 1
        delay = min(8.0, 2.0 ** (self.restart_count - 1))
        print(
            f"[supervisor][warn] {self.name} exited; restart "
            f"{self.restart_count}/{self.restart_limit} in {delay:.0f}s",
            flush=True,
        )
        time.sleep(delay)
        self.start()
        return True


def checked_python(path: Path, label: str) -> str:
    executable = path / "Scripts" / "python.exe"
    if not executable.exists():
        raise RuntimeError(f"{label} environment is missing: run Scripts\\setup.bat")
    result = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{label} environment is broken: run Scripts\\setup.bat")
    return str(executable)


def resolve_robot(brain_python: str, requested: str) -> str:
    value = requested
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    result = subprocess.run(
        [brain_python, str(ROOT / "Scripts" / "resolve_robot_host.py"), value],
        capture_output=True,
        text=True,
    )
    resolved = result.stdout.strip()
    if result.returncode or not resolved:
        raise RuntimeError(
            f"could not discover {value}; confirm Robit is online with protocol v1 firmware"
        )
    return resolved


def run_preparation(label: str, command: list[str], environment: dict[str, str]) -> None:
    print(f"[supervisor] {label}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        raise RuntimeError(f"{label} failed with code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise the Robit PC stack")
    parser.add_argument("robot", nargs="?", default="robit.local")
    parser.add_argument("port", nargs="?", type=int, default=8080)
    args = parser.parse_args()

    try:
        brain_python = checked_python(BRAIN / ".venv", "pc_brain")
        tracking_python = checked_python(TRACKING / ".venv", "pc_tracking")
        robot_url = resolve_robot(brain_python, args.robot)
        llama = (
            Path(r"C:\Tools\llama.cpp\llama-server.exe")
            if Path(r"C:\Tools\llama.cpp\llama-server.exe").exists()
            else shutil.which("llama-server")
        )
        if not llama:
            raise RuntimeError("llama-server was not found on PATH or C:\\Tools\\llama.cpp")
    except RuntimeError as exc:
        print(f"[supervisor][error] {exc}", file=sys.stderr)
        return 1

    environment = os.environ.copy()
    environment.update(
        {
            "ROBIT_BASE_URL": robot_url,
            "ROBIT_LLM_BASE_URL": "http://127.0.0.1:8081/v1",
            "ROBIT_LLM_MODEL": "ggml-org/gemma-4-E4B-it-GGUF:Q4_0",
            "ROBIT_REALTIME_MODEL": "ggml-org/gemma-4-E4B-it-GGUF:Q4_0",
            "ROBIT_VISION_BASE_URL": "http://127.0.0.1:8081/v1",
            "ROBIT_VISION_MODEL": "ggml-org/gemma-4-E4B-it-GGUF:Q4_0",
            "ROBIT_TRACKING_BASE_URL": "http://127.0.0.1:8091",
            "ROBIT_REALTIME_WS_URL": "ws://127.0.0.1:7861/v1/realtime",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "OPENAI_API_KEY": "local",
            "OPENAI_BASE_URL": "http://127.0.0.1:8081/v1",
        }
    )
    voice = environment.get("ROBIT_REALTIME_VOICE", "serena")
    device = environment.get("ROBIT_TRACKING_DEVICE", "auto")
    model = "ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
    tts_model = cached_snapshot_or_model_id(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    )
    services = [
        Service(
            "llama-server",
            [
                str(llama),
                "-hf",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
                "-np",
                "2",
                "-c",
                "65536",
                "-fa",
                "on",
                "--swa-full",
                "--reasoning",
                "off",
                "--image-max-tokens",
                "140",
            ],
            ROOT,
            environment,
            lambda: http_ready("http://127.0.0.1:8081/v1/models"),
            180,
        ),
        Service(
            "embedding-server",
            [
                str(llama),
                "-m",
                str(ROOT / "models" / "bekko-embedding-v1-a25m-Q8_0.gguf"),
                "--host",
                "127.0.0.1",
                "--port",
                "8093",
                "--embedding",
                "--pooling",
                "mean",
                "--embd-normalize",
                "2",
                "--ctx-size",
                "8192",
            ],
            ROOT,
            environment,
            lambda: http_ready("http://127.0.0.1:8093/v1/models"),
            120,
            optional=True,
        ),
        Service(
            "voice",
            [
                brain_python,
                "-m",
                "speech_to_speech.s2s_pipeline",
                "--mode",
                "realtime",
                "--ws_host",
                "0.0.0.0",
                "--ws_port",
                "7861",
                "--stt",
                "parakeet-tdt",
                "--parakeet_tdt_model_name",
                "nvidia/parakeet-tdt-0.6b-v3",
                "--parakeet_tdt_device",
                "cuda",
                "--parakeet_tdt_compute_type",
                "float16",
                "--llm_backend",
                "responses-api",
                "--model_name",
                model,
                "--responses_api_api_key",
                "local",
                "--responses_api_base_url",
                "http://127.0.0.1:8081/v1",
                "--responses_api_request_timeout_s",
                "180",
                "--tts",
                "qwen3",
                "--qwen3_tts_model_name",
                tts_model,
                "--qwen3_tts_device",
                "cuda",
                "--qwen3_tts_speaker",
                voice,
            ],
            BRAIN,
            environment,
            lambda: tcp_ready("127.0.0.1", 7861),
            180,
        ),
        Service(
            "tracking",
            [
                tracking_python,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8091",
            ],
            TRACKING,
            {**environment, "ROBIT_TRACKING_DEVICE": device},
            lambda: http_ready("http://127.0.0.1:8091/health"),
            120,
        ),
        Service(
            "pc-brain",
            [
                brain_python,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.port),
            ],
            BRAIN,
            environment,
            lambda: http_ready(f"http://127.0.0.1:{args.port}/health"),
            45,
            restart_limit=0,
        ),
    ]

    started: list[Service] = []
    return_code = 0
    try:
        run_preparation(
            "patching validated realtime request timeout",
            [brain_python, str(ROOT / "Scripts" / "patch_speech_to_speech_timeout.py")],
            environment,
        )
        for service in services:
            started.append(service)
            service.start()
            if service.name == "llama-server":
                run_preparation(
                    "prewarming shared language/vision model",
                    [
                        brain_python,
                        str(ROOT / "Scripts" / "prewarm_responses.py"),
                        "--base-url",
                        "http://127.0.0.1:8081/v1",
                        "--model",
                        model,
                        "--attempts",
                        "4",
                        "--sleep",
                        "2",
                        "--target-seconds",
                        "180",
                    ],
                    environment,
                )
                run_preparation(
                    "checking voice sidecar models",
                    [
                        brain_python,
                        str(ROOT / "Scripts" / "download_voice_models.py"),
                        "nvidia/parakeet-tdt-0.6b-v3",
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    ],
                    environment,
                )
        webbrowser.open(f"http://localhost:{args.port}")
        print("[supervisor] Robit is ready. Press Ctrl+C for clean shutdown.", flush=True)
        while True:
            time.sleep(1)
            for service in services[:-1]:
                if service.optional:
                    continue
                if not service.restart_if_failed():
                    raise RuntimeError(f"required sidecar failed: {service.name}")
            if services[-1].process and services[-1].process.poll() is not None:
                raise RuntimeError("PC Brain exited")
    except KeyboardInterrupt:
        print("[supervisor] shutting down", flush=True)
        return_code = 0
    except (RuntimeError, TimeoutError, OSError) as exc:
        print(f"[supervisor][error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return_code = 1
    finally:
        for service in reversed(started):
            service.stop()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

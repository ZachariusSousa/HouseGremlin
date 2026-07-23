# HouseGremlin / Robit

Robit should be built as two cooperating codebases:

- **Robot firmware**: runs on the ESP board, drives motors/servos, exposes a small HTTP API, and stays responsive.
- **PC brain**: runs on your computer, handles camera streaming, LLM/tool calling, speech, logging, and higher-level autonomy.

Do not put vision or LLM work on the motor controller. Keep the robot firmware boring and real-time-ish; offload expensive work to the PC.

## Repo Layout

```text
firmware/robit_controller/    ESP/Arduino firmware for movement and head control
pc_brain/                     FastAPI service that talks to the robot and owns future AI features
web_control/                  Browser control panel for manual driving
docs/architecture.md          Current system architecture and initial build roadmap
DESIGN.md                     Long-term brain, memory, vision, tools, and autonomy roadmap
Maindesign.stl                Current printable model
```

## First Build Path

1. Flash `firmware/robit_controller/robit_controller.ino`.
2. Edit Wi-Fi credentials in `firmware/robit_controller/config.example.h`, save as `config.h`, and keep it private.
3. Confirm manual control works through the robot HTTP API.
4. Run the PC brain and point it at the robot IP.
5. Add camera streaming.
6. Add LLM tool calling against the PC brain API, not directly against the microcontroller.

## Windows Quick Start

From PowerShell:

```powershell
.\Scripts\setup.bat
.\Scripts\run.bat
```

By default, `run.bat` uses Robit's mDNS name, `http://robit.local`. Its bundled
mDNS discovery resolves the current numeric address internally, including after
Robit moves to another network; no IP argument is normally needed.
To override it with a direct IP printed by the robot Serial Monitor:

```powershell
.\Scripts\run.bat 172.22.1.126
```

## Robot HTTP API

The firmware exposes:

- `GET /status`
- `GET /cmd?move=forward|reverse|left|right|stop`
- `GET /speed?value=0..255`
- `GET /servo?pan=55..135`
- `GET /servo?tilt=35..115`

It also exposes first-pass JSON-style aliases for the PC brain and later LLM
tool layer:

- `GET /api/status`
- `POST /api/move` with `direction`, optional `speed`, and optional `duration_ms`
- `POST /api/head` with `pan`/`tilt` or `pan_delta`/`tilt_delta`
- `POST /api/emergency-stop`
- `GET /camera`
- `GET /camera/capture`
- `GET /camera/stream` redirects to the MJPEG stream on port `81`

All camera acquisition is serialized with a five-second minimum interval
(`ROBIT_CAMERA_FRAME_INTERVAL_MS=5000`) across still captures and MJPEG clients.
The browser uses the PC brain's shared frame broker rather than opening its own
raw stream.

The XIAO ESP32S3 Sense camera stream is served at:

```text
http://ROBOT_IP/camera
http://ROBOT_IP:81/stream
```

Current soldered pin assumptions:

- `D0` left forward, `D1` left reverse
- `D2` right forward, `D3` right reverse
- `D4` shared PWM for both motor-driver PWM inputs
- `D6`/`D7` PCA9685 servo I2C
- `D5`/`D8` reserved for the OLED eye I2C bus

## PC Brain

The PC service is intentionally a thin scaffold right now. It gives you a clean place to add:

- camera capture and streaming
- OpenAI/LLM tool calls
- speech input/output
- scripted behaviors
- telemetry logging

See [docs/architecture.md](docs/architecture.md) for the current architecture and
[DESIGN.md](DESIGN.md) for the long-term Robit design roadmap.

Text and the language step in realtime Voice use one local Gemma 4 E4B
`llama-server`; structured Vision currently reuses it as well. Parakeet STT and
Qwen TTS remain separate voice models, and the Vision adapter remains replaceable
by a dedicated detector or VLM later. Install the validated shared environment
with:

```powershell
.\Scripts\setup.bat
```

The current Gate 5 surface is `GET /perception/latest` plus
`POST /perception/query`. Visual results are descriptive only and cannot issue
movement or head commands in the same turn.

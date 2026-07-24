# HouseGremlin / Robit

Robit should be built as two cooperating codebases:

- **Robot firmware**: runs on the ESP board, drives motors/servos, exposes a small HTTP API, and stays responsive.
- **PC brain**: runs on your computer, handles camera streaming, LLM/tool calling, speech, logging, and higher-level autonomy.

Do not put vision or LLM work on the motor controller. Keep the robot firmware boring and real-time-ish; offload expensive work to the PC.

## Repo Layout

```text
firmware/robit_controller/    ESP/Arduino firmware for movement and head control
pc_brain/                     FastAPI service that talks to the robot and owns future AI features
pc_tracking/                  Isolated local RF-DETR Nano person detector
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

Install 64-bit Python 3.11 first, then run from PowerShell:

```powershell
py -3.11 --version
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
- `GET /camera/stream` is a legacy alias for the still-capture endpoint on port `81`

All camera acquisition is serialized through one mutex and capped globally at
5 FPS while tracking (`ROBIT_CAMERA_MAX_FPS=5`). The PC
brain's shared frame broker controls demand separately. Person tracking is on
by default and requests fresh shared frames without opening another camera
stream. The broker returns to its low idle rate only when tracking is explicitly
disabled.

The XIAO ESP32S3 Sense camera page and still endpoint are:

```text
http://ROBOT_IP/camera
http://ROBOT_IP:81/capture
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

RF-DETR Nano runs in its own `pc_tracking\.venv`, so its Transformers 5
dependency cannot alter the validated voice environment. It powers one simple
always-on person tracker: Robit turns its head toward the visible person and
uses one proportional in-place body turn based on the estimated target bearing
when the head alone cannot keep up. Tracking
starts with Robit, has no timeout, and remains off only after an explicit stop
or emergency stop. It does not replace semantic E4B scene descriptions and it
never retains camera frames. The tracking API is:

```text
GET  /tracking/status
POST /tracking/start
POST /tracking/stop
```

After `run.bat` is running, verify the default:

```powershell
Invoke-RestMethod http://localhost:8080/tracking/status
```

If tracking was explicitly stopped, turn it back on with:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:8080/tracking/start `
  -ContentType "application/json" `
  -Body '{}'
```

# HouseGremlin / Robit

Robit should be built as two cooperating codebases:

- **Robot firmware**: runs on the ESP board, drives motors/servos, exposes status/camera HTTP plus a persistent control socket, and stays responsive.
- **PC brain**: runs on your computer, handles camera streaming, LLM/tool calling, speech, logging, and higher-level autonomy.

Do not put vision or LLM work on the motor controller. Keep the robot firmware boring and real-time-ish; offload expensive work to the PC.

## Naming

- **HouseGremlin** = this repository / the overall project brand.
- **Robit** = the physical robot (firmware, mDNS `robit.local`, `ROBIT_*` env vars).

## Repo Layout

| Path | What it is | Runs standalone? |
|---|---|---|
| `brain/` | FastAPI PC service: talks to the robot, owns LLM/vision/speech/autonomy | needs robot + LLM |
| `tracking/` | RF-DETR Nano person-detector sidecar (port 8091) | needs GPU, no robot |
| `memory/` | SQLite knowledge graph + ingestion driver + web researcher (port 8092) | Python only |
| `web_control/` | Browser control panel (served by the brain) | — |
| `firmware/robit_controller/` | ESP32 firmware: motors, head, camera, control socket | on the robot |
| `Scripts/` | Cross-package glue: `supervisor.py`, `setup.bat`, `run.bat` | — |
| `cad/` | Printable model (Maindesign.stl) | — |
| `docs/architecture.md` | What talks to what, and why | — |

Port map: **8080** brain · **8081** llama-server (Gemma LLM) · **8091** tracking ·
**8092** memory · **8093** Bekko embeddings. Each package has its own `.venv`,
`requirements.txt`, and `.env.example`; start with the per-package README.

```text
                 ┌──────────────┐  TCP 82 (control)   ┌─────────────┐
 camera stream ─▶│    brain     │◀───────────────────▶│ Robit ESP32 │
   (from robot)  │    :8080     │                     │  firmware   │
                 └──┬───────┬───┘                     └─────────────┘
        HTTP 8091   │       │  OpenAI-compat 8081 / 8093
                    ▼       ▼
             ┌──────────┐  ┌──────────────────────────┐
             │ tracking │  │ llama-server (Gemma LLM) │ + Bekko embeds
             └──────────┘  └──────────────────────────┘
                 memory (:8092) is standalone — ingests brain telemetry,
                 runs the web researcher; nothing else depends on it.
```

`Scripts/supervisor.py` launches brain + tracking + both model servers together;
`memory/` runs on its own (`memory/.venv/Scripts/python -m memory.app.cli`).

## First Build Path

1. Flash `firmware/robit_controller/robit_controller.ino`.
2. Edit Wi-Fi credentials in `firmware/robit_controller/config.example.h`, save as `config.h`, and keep it private.
3. Deploy the matching Brain and confirm its `/robot/*` controls work.
4. Run the PC brain and point it at the robot IP.
5. Add camera streaming.
6. Add LLM tool calling against the PC brain API, not directly against the microcontroller.

## Windows Quick Start

Install 64-bit Python 3.13 first, then run from PowerShell:

```powershell
py -3.13 --version
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

## Robot interfaces

The firmware HTTP server exposes diagnostics and camera access only:

- `GET /status`
- `GET /api/status`
- `GET /camera`
- `GET /camera/capture`
- `GET /camera/stream` is a legacy alias for the still-capture endpoint on port `81`

Actuation uses protocol v1 on TCP port `82`. It is single-client, newline-delimited
JSON with a 512-byte limit, acknowledgements, TTL/sequence rejection, a one-second
PC heartbeat, and firmware telemetry. Direct ESP movement/head/eyes HTTP routes
were intentionally removed; use the stable Brain `/robot/*` API.

All camera acquisition is serialized and capped globally at 2 FPS
(`ROBIT_CAMERA_MAX_FPS=2`). The Brain shares one raw JPEG, rotated JPEG, and
preview for each frame. Idle acquisition is 0.2 FPS.

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

## Brain

The PC service owns everything above the robot's HTTP/sockets layer: camera
capture and streaming, OpenAI-compatible LLM tool calling, realtime voice
(Parakeet STT / Qwen TTS sidecar), scripted behaviors, telemetry logging, and
the person-tracking controller that consumes `tracking/` detections.

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

RF-DETR Nano runs in its own `tracking\.venv`, so its Transformers 5
dependency cannot alter the validated voice environment. It powers one simple
always-on person tracker: Robit turns its head toward the visible person and
uses a delayed, bounded in-place body turn only when the head cannot keep up.
RF-DETR runs at 2 FPS while searching, 1 FPS on a stable target, and 0.5 FPS
during voice activity. Tracking starts with Robit and remains off only after an
explicit tracking stop. It does not replace semantic E4B scene descriptions and
it never retains routine camera frames. The tracking API is:

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

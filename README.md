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
docs/architecture.md          System design and build roadmap
Maindesign.stl                Current printable model
```

## First Build Path

1. Flash `firmware/robit_controller/robit_controller.ino`.
2. Edit Wi-Fi credentials in `firmware/robit_controller/config.example.h`, save as `config.h`, and keep it private.
3. Confirm manual control works through the robot HTTP API.
4. Run the PC brain and point it at the robot IP.
5. Add camera streaming.
6. Add LLM tool calling against the PC brain API, not directly against the microcontroller.

## Robot HTTP API

The firmware exposes:

- `GET /status`
- `GET /cmd?move=forward|reverse|left|right|stop`
- `GET /speed?value=0..255`
- `GET /servo?pan=55..135`
- `GET /servo?tilt=35..115`

## PC Brain

The PC service is intentionally a thin scaffold right now. It gives you a clean place to add:

- camera capture and streaming
- OpenAI/LLM tool calls
- speech input/output
- scripted behaviors
- telemetry logging

See [docs/architecture.md](docs/architecture.md) for the design.

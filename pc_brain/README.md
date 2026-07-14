# PC Brain

This service runs on your computer and owns Robit's coordinator and safety layer: conversation state, event journaling, camera proxying, robot commands, typed and realtime LLM actions, arbitration, and logs.

The realtime speech server runs as a sidecar process, but it uses the same `pc_brain\.venv` as the rest of the project.

## Prerequisites

- Python 3.11 for the shared `pc_brain\.venv`
- [Ollama](https://ollama.com/) for typed Text mode with `gemma4:e4b`
- `llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) for realtime voice
- NVIDIA GPU recommended for local realtime speech

## Run

From the repository root, run setup once:

```powershell
cd C:\Users\z1sou\HouseGremlin
Scripts\setup.bat
```

Then start Robit:

```powershell
Scripts\run.bat
```

If omitted, `Scripts\run.bat` uses Robit's mDNS name: `robit.local`.

`setup.bat` creates one virtual environment at `pc_brain\.venv` and installs both PC brain and realtime voice dependencies there.
`run.bat` starts Ollama for silent typed chat, starts `llama-server` for realtime voice, starts the realtime voice sidecar from that same venv, starts the PC brain, and opens the browser UI.

To test realtime voice without the browser:

```powershell
cd C:\Users\z1sou\HouseGremlin
Scripts\voice_test.bat
```

This starts the realtime sidecar, connects a terminal mic/speaker client to `ROBIT_REALTIME_WS_URL`, and exits with `Ctrl+C`.

Manual PC brain setup, if you need to debug it directly:

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Manual realtime sidecar run, if you need to debug it directly:

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\Activate.ps1
$env:OPENAI_API_KEY="local"
$env:OPENAI_BASE_URL="http://127.0.0.1:8081/v1"
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF --host 127.0.0.1 --port 8081 -np 2 -c 65536 -fa on --swa-full
```

In another PowerShell window:

```powershell
cd C:\Users\z1sou\HouseGremlin\pc_brain
.\.venv\Scripts\Activate.ps1
$env:OPENAI_API_KEY="local"
$env:OPENAI_BASE_URL="http://127.0.0.1:8081/v1"
python -m speech_to_speech.s2s_pipeline --mode realtime --ws_host 0.0.0.0 --ws_port 7861 --stt parakeet-tdt --parakeet_tdt_model_name nvidia/parakeet-tdt-0.6b-v3 --parakeet_tdt_device cuda --parakeet_tdt_compute_type float16 --llm_backend responses-api --model_name ggml-org/gemma-4-E4B-it-GGUF --responses_api_api_key local --responses_api_base_url http://127.0.0.1:8081/v1 --responses_api_request_timeout_s 180 --tts qwen3 --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --qwen3_tts_device cuda --qwen3_tts_speaker serena
```

Edit `.env` only if `robit.local` does not resolve on your network.
The typed Text mode default is Ollama `gemma4:e4b`. The realtime voice default is the llama.cpp/Hugging Face model `ggml-org/gemma-4-E4B-it-GGUF`, served from `http://127.0.0.1:8081/v1`.
LLM-issued movement is clamped by `ROBIT_LLM_MAX_SPEED=180` and `ROBIT_LLM_MAX_DURATION_MS=1000`.

## Checks

From the repository root:

```powershell
python -m compileall pc_brain\app pc_brain\tests
python -m pytest pc_brain\tests
```

From inside `pc_brain`:

```powershell
python -m compileall app tests
python -m pytest tests
```

## Endpoints

- `GET /health`
- `GET /robot/status`
- `GET /robot/camera`
- `GET /robot/camera/capture`
- `POST /robot/drive`
- `POST /robot/head`
- `POST /robot/stop`
- `POST /robot/action`
- `POST /chat`
- `POST /chat/action`
- `GET /brain/state`
- `GET /brain/events`
- `WS /v1/realtime`
- `GET /tools`

The realtime voice sidecar remains separate at `ROBIT_REALTIME_WS_URL`, defaulting to `ws://localhost:7861/v1/realtime`, but it is private to `pc_brain`. The browser connects to the brain gateway at `WS /v1/realtime`.

## Operator Console

The console shows the live robot camera, robot telemetry, silent Text mode, realtime Voice mode, manual controls, and an action log. Text mode calls `/chat/action`; Voice mode streams mic audio to `pc_brain`, which owns the sidecar session and executes the declared `robot_action` tool server-side.

Quick endpoint checks:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod http://localhost:8080/robot/status
Invoke-RestMethod http://localhost:8080/robot/camera
Invoke-RestMethod http://localhost:8080/brain/state
Invoke-RestMethod "http://localhost:8080/brain/events?conversation_id=default&after_sequence=0&limit=100"
Invoke-WebRequest http://localhost:8080/robot/camera/capture -OutFile $env:TEMP\robit.jpg
Invoke-RestMethod http://localhost:8080/robot/action `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"movement":{"direction":"forward","speed":120,"duration_ms":300}}'
```

Test silent typed chat/action:

```powershell
Invoke-RestMethod http://localhost:8080/chat/action `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"Say hello in one short sentence.","conversation_id":"default"}'
```

Capture Gate 1 realtime latency and GPU usage after the stack is ready:

```powershell
cd C:\Users\z1sou\HouseGremlin
.\pc_brain\.venv\Scripts\python.exe .\Scripts\benchmark_gate1.py
```

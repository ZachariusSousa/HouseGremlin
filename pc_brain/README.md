# PC Brain

This service runs on your computer and owns the expensive work: camera streaming, LLM/tool calling, speech, autonomy, and logs.

## Prerequisites

- Python 3.11 recommended for the Chatterbox-Turbo voice stack.
- [Ollama](https://ollama.com/) running locally
- `ffmpeg` available on `PATH` for MP3/WAV normalization
- NVIDIA GPU recommended for interactive Chatterbox-Turbo speech

## Run

If you already have an existing voice-stack venv, recreate it so the Chatterbox dependency pins are clean:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull gemma4:e4b
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Run `ollama serve` in a separate terminal if Ollama is not already running.
Edit `.env` and replace `ROBIT_BASE_URL` with the IP printed by the robot firmware.
The default text model is `gemma4:e4b`, and chat requests send `think=false`.
The voice model is Chatterbox-Turbo, configured by `ROBIT_TTS_PROVIDER=chatterbox_turbo` and `ROBIT_TTS_MODEL=ResembleAI/chatterbox-turbo`.
The Chatterbox sampling settings are exposed for experiments:
`ROBIT_TTS_TEMPERATURE=0.8`, `ROBIT_TTS_TOP_P=0.95`, `ROBIT_TTS_TOP_K=1000`, and `ROBIT_TTS_REPETITION_PENALTY=1.2`.

`python --version` should print `Python 3.11.x`. If `py -3.11` is not available, install Python 3.11 first. The global Python on this machine appears to be 3.14, which is not the recommended runtime for this service.

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
- `POST /robot/drive`
- `POST /robot/head`
- `POST /robot/stop`
- `GET /tools`
- `GET /voices`
- `POST /voices`
- `POST /voice/transcribe`
- `POST /chat`
- `POST /chat/speak`
- `POST /voice/synthesize`
- `POST /voice/roundtrip`

The LLM/tool layer should call these PC endpoints, not the ESP firmware directly.

## Voice Pipeline

Upload a sample voice first:

```powershell
curl.exe -F "voice_id=default" -F "sample=@C:\path\to\voice.mp3" http://localhost:8080/voices
```

Upload more samples with the same `voice_id` to add more reference clips to that voice.

Then test the roundtrip:

```powershell
curl.exe -F "audio=@C:\path\to\question.wav" -F "voice_id=default" http://localhost:8080/voice/roundtrip
```

Generated speech is written under `data/audio` and served from `/audio/{file}.wav`.

The server logs timing lines for expensive stages with `perf operation=... elapsed_ms=...`.
Use these to compare LLM, STT, voice normalization, and Chatterbox-Turbo synthesis costs.

Check the active voice runtime:

```powershell
$health = Invoke-RestMethod http://localhost:8080/health
$health.tts_runtime
```

For best performance, `cuda_available` should be `True` and `cuda_device_name` should show the NVIDIA GPU.

Send text and have Robit answer with speech:

```powershell
$response = Invoke-RestMethod -Method Post "http://localhost:8080/chat/speak" `
  -ContentType "application/json" `
  -Body '{"text":"Say hello in one short sentence.","voice_id":"default"}'

$response
Start-Process "http://localhost:8080$($response.audio_url)"
```

Benchmark chat-to-speech latency:

```powershell
Measure-Command {
  Invoke-RestMethod http://localhost:8080/chat/speak `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"text":"How are you doing?","voice_id":"default"}'
}
```

Compare Chatterbox-only synthesis without LLM time:

```powershell
Measure-Command {
  Invoke-RestMethod http://localhost:8080/voice/synthesize `
    -Method Post `
    -ContentType "application/json" `
    -Body '{"text":"Ready to help.","voice_id":"default"}'
}
```

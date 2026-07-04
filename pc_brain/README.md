# PC Brain

This service runs on your computer and owns the expensive work: camera streaming, LLM/tool calling, speech, autonomy, and logs.

## Prerequisites

- Python 3.10 or 3.11. Python 3.12+ is not supported because Coqui `TTS==0.22.0` is required for XTTS voice cloning.
- [Ollama](https://ollama.com/) running locally
- `ffmpeg` available on `PATH` for MP3/WAV normalization
- NVIDIA GPU recommended for the local speech pipeline

## Run

```powershell
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

`python --version` should print `Python 3.11.x` or `Python 3.10.x`. If `py -3.11` is not available, install Python 3.11 first. The global Python on this machine appears to be 3.14, which is too new for Coqui XTTS.

If XTTS fails with `cannot import name 'BeamSearchScorer' from 'transformers'`, reinstall the pinned dependency stack:

```powershell
pip install --force-reinstall transformers==4.33.3 tokenizers==0.13.3
pip install -r requirements.txt
```

If XTTS fails with `Weights only load failed`, your venv has PyTorch 2.6+ installed. Reinstall the pinned XTTS runtime:

```powershell
pip install --force-reinstall numpy==1.26.4 torch==2.5.1 torchaudio==2.5.1 transformers==4.33.3 tokenizers==0.13.3
pip install -r requirements.txt
```

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

Send text and have Robit answer with speech:

```powershell
$response = Invoke-RestMethod -Method Post "http://localhost:8080/chat/speak" `
  -ContentType "application/json" `
  -Body '{"text":"Say hello in one short sentence.","voice_id":"default"}'

$response
Start-Process "http://localhost:8080$($response.audio_url)"
```

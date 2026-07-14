@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
set "DEFAULT_ROBOT_HOST=robit.local"
set "REALTIME_PORT=7861"
set "TEXT_MODEL=gemma4:e4b"
set "LLAMA_SERVER_PORT=8081"
set "LLAMA_SERVER_EXE=llama-server"
set "REALTIME_MODEL=ggml-org/gemma-4-E4B-it-GGUF"
set "REALTIME_LLM_BASE_URL=http://127.0.0.1:%LLAMA_SERVER_PORT%/v1"
set "REALTIME_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3"
set "REALTIME_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
if "%ROBIT_REALTIME_VOICE%"=="" (
  set "REALTIME_VOICE=serena"
) else (
  set "REALTIME_VOICE=%ROBIT_REALTIME_VOICE%"
)
set "RUN_ID=%RANDOM%%RANDOM%"
set "S2S_LOG=%TEMP%\robit-realtime-voice-%RUN_ID%.log"
set "S2S_RUNNER=%TEMP%\robit-realtime-voice-%RUN_ID%.bat"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=8080"

cd /d "%PC_BRAIN%" || exit /b 1

if not exist "%VENV%\Scripts\python.exe" (
  echo [run][error] pc_brain\.venv was not found.
  echo Run Scripts\setup.bat first.
  exit /b 1
)

"%VENV%\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
  echo [run][error] pc_brain\.venv exists but its Python executable is broken.
  echo Run Scripts\setup.bat after deleting or repairing pc_brain\.venv.
  exit /b 1
)

call "%VENV%\Scripts\activate.bat" || exit /b 1

if exist "C:\Tools\llama.cpp\llama-server.exe" set "LLAMA_SERVER_EXE=C:\Tools\llama.cpp\llama-server.exe"

echo [run] Stopping stale realtime voice processes
python "%ROOT%\Scripts\stop_voice_stack.py" --ports %REALTIME_PORT% %LLAMA_SERVER_PORT% || exit /b 1

python "%ROOT%\Scripts\patch_speech_to_speech_timeout.py" || exit /b 1

python -c "import pkg_resources" >nul 2>&1
if errorlevel 1 (
  echo [run] Installing setuptools with pkg_resources for Qwen/librosa
  python -m pip install "setuptools>=70,<81" || exit /b 1
)

python -c "import huggingface_hub, sys; from packaging.version import Version; v=Version(huggingface_hub.__version__); sys.exit(0 if Version('0.34.0') <= v < Version('1.0') else 1)" >nul 2>&1
if errorlevel 1 (
  echo [run] Installing transformers-compatible huggingface_hub
  python -m pip install "huggingface_hub>=0.34.0,<1.0" || exit /b 1
)

python -c "import pydantic, sys; sys.exit(0 if pydantic.__version__ == '2.13.4' else 1)" >nul 2>&1
if errorlevel 1 (
  echo [run] Installing OpenAI-compatible Pydantic
  python -m pip install "pydantic==2.13.4" || exit /b 1
)

if "%~1"=="" (
  set "ROBOT_ARG=%DEFAULT_ROBOT_HOST%"
) else (
  set "ROBOT_ARG=%~1"
)

if not "%ROBOT_ARG%"=="" (
  echo %ROBOT_ARG% | findstr /b /i "http:// https://" >nul
  if errorlevel 1 (
    set "ROBIT_BASE_URL=http://%ROBOT_ARG%"
  ) else (
    set "ROBIT_BASE_URL=%ROBOT_ARG%"
  )
)

echo [run] ROBIT_BASE_URL=%ROBIT_BASE_URL%

where ollama >nul 2>&1
if not errorlevel 1 (
  echo [run] Starting Ollama in a background window if it is not already running
  start "Ollama" /min cmd /c "ollama serve"
  timeout /t 3 /nobreak >nul
  echo [run] Prewarming typed Text mode model %TEXT_MODEL%
  ollama run %TEXT_MODEL% "Reply with the word ready." || exit /b 1
) else (
  echo [run] Ollama was not found on PATH. Chat will fail until Ollama is running.
)

if "%LLAMA_SERVER_EXE%"=="llama-server" (
  where llama-server >nul 2>&1
  if errorlevel 1 (
    echo [run][error] llama-server was not found on PATH.
    echo Install llama.cpp, make sure llama-server.exe is on PATH, then run this again.
    echo Download: https://github.com/ggml-org/llama.cpp/releases
    exit /b 1
  )
)

if not "%LLAMA_SERVER_EXE%"=="llama-server" if not exist "%LLAMA_SERVER_EXE%" (
  echo [run][error] llama-server was not found at %LLAMA_SERVER_EXE%.
  echo Install llama.cpp, make sure llama-server.exe is on PATH, then run this again.
  echo Download: https://github.com/ggml-org/llama.cpp/releases
  exit /b 1
)

echo [run] Starting llama-server for realtime voice on %REALTIME_LLM_BASE_URL%
echo [run] Model: %REALTIME_MODEL%
echo [run] llama-server: %LLAMA_SERVER_EXE%
start "Robit llama-server" /min cmd /c ""%LLAMA_SERVER_EXE%" -hf %REALTIME_MODEL% --host 127.0.0.1 --port %LLAMA_SERVER_PORT% -np 2 -c 65536 -fa on --swa-full"
echo [run] Waiting for llama-server /v1/responses
python "%ROOT%\Scripts\prewarm_responses.py" --base-url "%REALTIME_LLM_BASE_URL%" --model "%REALTIME_MODEL%" --attempts 90 --sleep 2 --target-seconds 180 || exit /b 1

echo [run] Downloading voice sidecar models if needed
python "%ROOT%\Scripts\download_voice_models.py" "%REALTIME_STT_MODEL%" "%REALTIME_TTS_MODEL%" || exit /b 1

echo [run] Starting realtime voice sidecar on ws://localhost:%REALTIME_PORT%/v1/realtime
echo [run] Pipeline: silero-vad to %REALTIME_STT_MODEL% to %REALTIME_MODEL% to %REALTIME_TTS_MODEL% (%REALTIME_VOICE%)
echo [run] Sidecar log: %S2S_LOG%
echo [run] Watch log: Get-Content "%S2S_LOG%" -Wait
(
  echo @echo off
  echo cd /d "%PC_BRAIN%"
  echo call "%VENV%\Scripts\activate.bat"
  echo set OPENAI_API_KEY=local
  echo set OPENAI_BASE_URL=%REALTIME_LLM_BASE_URL%
  echo echo [sidecar] verifying llama-server /v1/responses before speech-to-speech ^> "%S2S_LOG%"
  echo python "%ROOT%\Scripts\prewarm_responses.py" --base-url "%REALTIME_LLM_BASE_URL%" --model "%REALTIME_MODEL%" --attempts 4 --sleep 2 --target-seconds 180 ^>^> "%S2S_LOG%" 2^>^&1
  echo if errorlevel 1 exit /b 1
  echo echo [sidecar] starting HF-style local pipeline: silero-vad to %REALTIME_STT_MODEL% to %REALTIME_MODEL% to %REALTIME_TTS_MODEL% ^>^> "%S2S_LOG%"
  echo echo [sidecar] command: python -m speech_to_speech.s2s_pipeline --mode realtime --ws_host 0.0.0.0 --ws_port %REALTIME_PORT% --stt parakeet-tdt --parakeet_tdt_model_name %REALTIME_STT_MODEL% --parakeet_tdt_device cuda --parakeet_tdt_compute_type float16 --llm_backend responses-api --model_name %REALTIME_MODEL% --responses_api_base_url %REALTIME_LLM_BASE_URL% --tts qwen3 --qwen3_tts_model_name %REALTIME_TTS_MODEL% --qwen3_tts_device cuda --qwen3_tts_speaker %REALTIME_VOICE% ^>^> "%S2S_LOG%"
  echo python -m speech_to_speech.s2s_pipeline --mode realtime --ws_host 0.0.0.0 --ws_port %REALTIME_PORT% --stt parakeet-tdt --parakeet_tdt_model_name %REALTIME_STT_MODEL% --parakeet_tdt_device cuda --parakeet_tdt_compute_type float16 --llm_backend responses-api --model_name %REALTIME_MODEL% --responses_api_api_key local --responses_api_base_url %REALTIME_LLM_BASE_URL% --responses_api_request_timeout_s 180 --tts qwen3 --qwen3_tts_model_name %REALTIME_TTS_MODEL% --qwen3_tts_device cuda --qwen3_tts_speaker %REALTIME_VOICE% ^>^> "%S2S_LOG%" 2^>^&1
) > "%S2S_RUNNER%"
start "Robit Realtime Voice" /min cmd /k call "%S2S_RUNNER%"

echo [run] Opening http://localhost:%PORT%
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:%PORT%"

echo [run] Starting PC brain. Press Ctrl+C in this window to stop.
python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%

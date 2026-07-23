@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
set "DEFAULT_ROBOT_HOST=robit.local"
set "REALTIME_PORT=7861"
set "LLAMA_SERVER_PORT=8081"
set "LLAMA_SERVER_EXE=llama-server"
set "E4B_MODEL=ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
set "E4B_BASE_URL=http://127.0.0.1:%LLAMA_SERVER_PORT%/v1"
set "ROBIT_LLM_BASE_URL=%E4B_BASE_URL%"
set "ROBIT_LLM_MODEL=%E4B_MODEL%"
set "ROBIT_REALTIME_MODEL=%E4B_MODEL%"
set "ROBIT_VISION_BASE_URL=%E4B_BASE_URL%"
set "ROBIT_VISION_MODEL=%E4B_MODEL%"
set "ROBIT_VISION_IMAGE_TOKENS=140"
set "REALTIME_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3"
set "REALTIME_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
rem These voice models are public. Ignore a stale saved Hugging Face token so
rem metadata fallbacks do not fail with an unrelated 401 during sidecar startup.
set "HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
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

python -c "import huggingface_hub, numpy, pydantic, torch, torchaudio, transformers, sys; from packaging.version import Version; ok = transformers.__version__ == '4.57.3' and Version('0.34.0') <= Version(huggingface_hub.__version__) < Version('1.0') and numpy.__version__ == '1.26.4' and pydantic.__version__ == '2.13.4' and torch.__version__ == '2.6.0+cu124' and torchaudio.__version__ == '2.6.0+cu124'; sys.exit(0 if ok else 1)" >nul 2>&1
if errorlevel 1 (
  echo [run][error] The shared Python environment has incompatible package versions.
  echo Run Scripts\setup.bat to restore the validated voice and vision environment.
  exit /b 1
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

rem Resolve robit.local with true mDNS discovery. Windows DNS does not reliably
rem resolve .local names, so the PC brain uses the discovered IP internally.
set "ROBIT_REQUESTED_URL=%ROBIT_BASE_URL%"
set "ROBIT_RESOLVED_URL="
for /f "delims=" %%I in ('python "%ROOT%\Scripts\resolve_robot_host.py" "%ROBIT_BASE_URL%"') do set "ROBIT_RESOLVED_URL=%%I"
if "%ROBIT_RESOLVED_URL%"=="" (
  echo [run][error] Could not discover %ROBIT_BASE_URL% with mDNS.
  echo Confirm Robit is powered on, connected to this network, and running the latest firmware.
  exit /b 1
)
set "ROBIT_BASE_URL=%ROBIT_RESOLVED_URL%"

echo [run] Robot: %ROBIT_REQUESTED_URL% resolved to %ROBIT_BASE_URL%

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

echo [run] Starting the shared E4B llama-server for Text, Voice LLM, and current Vision on %E4B_BASE_URL%
echo [run] Model: %E4B_MODEL%
echo [run] llama-server: %LLAMA_SERVER_EXE%
start "Robit E4B llama-server" /min cmd /c ""%LLAMA_SERVER_EXE%" -hf %E4B_MODEL% --host 127.0.0.1 --port %LLAMA_SERVER_PORT% -np 2 -c 65536 -fa on --swa-full --reasoning off --image-max-tokens %ROBIT_VISION_IMAGE_TOKENS%"
echo [run] Waiting for llama-server /v1/responses
python "%ROOT%\Scripts\prewarm_responses.py" --base-url "%E4B_BASE_URL%" --model "%E4B_MODEL%" --attempts 90 --sleep 2 --target-seconds 180 || exit /b 1

echo [run] Downloading voice sidecar models if needed
python "%ROOT%\Scripts\download_voice_models.py" "%REALTIME_STT_MODEL%" "%REALTIME_TTS_MODEL%" || exit /b 1

echo [run] Starting realtime voice sidecar on ws://localhost:%REALTIME_PORT%/v1/realtime
echo [run] Pipeline: silero-vad to %REALTIME_STT_MODEL% to %E4B_MODEL% to %REALTIME_TTS_MODEL% (%REALTIME_VOICE%)
echo [run] Sidecar log: %S2S_LOG%
echo [run] Watch log: Get-Content "%S2S_LOG%" -Wait
(
  echo @echo off
  echo cd /d "%PC_BRAIN%"
  echo call "%VENV%\Scripts\activate.bat"
  echo set OPENAI_API_KEY=local
  echo set OPENAI_BASE_URL=%E4B_BASE_URL%
  echo echo [sidecar] verifying llama-server /v1/responses before speech-to-speech ^> "%S2S_LOG%"
  echo python "%ROOT%\Scripts\prewarm_responses.py" --base-url "%E4B_BASE_URL%" --model "%E4B_MODEL%" --attempts 4 --sleep 2 --target-seconds 180 ^>^> "%S2S_LOG%" 2^>^&1
  echo if errorlevel 1 exit /b 1
  echo echo [sidecar] starting HF-style local pipeline: silero-vad to %REALTIME_STT_MODEL% to %E4B_MODEL% to %REALTIME_TTS_MODEL% ^>^> "%S2S_LOG%"
  echo echo [sidecar] command: python -m speech_to_speech.s2s_pipeline --mode realtime --ws_host 0.0.0.0 --ws_port %REALTIME_PORT% --stt parakeet-tdt --parakeet_tdt_model_name %REALTIME_STT_MODEL% --parakeet_tdt_device cuda --parakeet_tdt_compute_type float16 --llm_backend responses-api --model_name %E4B_MODEL% --responses_api_base_url %E4B_BASE_URL% --tts qwen3 --qwen3_tts_model_name %REALTIME_TTS_MODEL% --qwen3_tts_device cuda --qwen3_tts_speaker %REALTIME_VOICE% ^>^> "%S2S_LOG%"
  echo python -m speech_to_speech.s2s_pipeline --mode realtime --ws_host 0.0.0.0 --ws_port %REALTIME_PORT% --stt parakeet-tdt --parakeet_tdt_model_name %REALTIME_STT_MODEL% --parakeet_tdt_device cuda --parakeet_tdt_compute_type float16 --llm_backend responses-api --model_name %E4B_MODEL% --responses_api_api_key local --responses_api_base_url %E4B_BASE_URL% --responses_api_request_timeout_s 180 --tts qwen3 --qwen3_tts_model_name %REALTIME_TTS_MODEL% --qwen3_tts_device cuda --qwen3_tts_speaker %REALTIME_VOICE% ^>^> "%S2S_LOG%" 2^>^&1
) > "%S2S_RUNNER%"
start "Robit Realtime Voice" /min cmd /k call "%S2S_RUNNER%"

echo [run] Opening http://localhost:%PORT%
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:%PORT%"

echo [run] Starting PC brain. Press Ctrl+C in this window to stop.
python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%

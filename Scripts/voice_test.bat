@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
set "REALTIME_PORT=7861"
set "LLAMA_SERVER_PORT=8081"
set "LLAMA_SERVER_EXE=llama-server"
set "REALTIME_MODEL=ggml-org/gemma-4-E4B-it-GGUF:Q4_0"
set "REALTIME_LLM_BASE_URL=http://127.0.0.1:%LLAMA_SERVER_PORT%/v1"
set "ROBIT_REALTIME_MODEL=%REALTIME_MODEL%"
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

cd /d "%PC_BRAIN%" || exit /b 1

if not exist "%VENV%\Scripts\python.exe" (
  echo [voice-test][error] pc_brain\.venv was not found.
  echo Run Scripts\setup.bat first.
  exit /b 1
)

"%VENV%\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
  echo [voice-test][error] pc_brain\.venv exists but its Python executable is broken.
  echo Run Scripts\setup.bat to recreate it.
  exit /b 1
)

call "%VENV%\Scripts\activate.bat" || exit /b 1

if exist "C:\Tools\llama.cpp\llama-server.exe" set "LLAMA_SERVER_EXE=C:\Tools\llama.cpp\llama-server.exe"

echo [voice-test] Stopping stale realtime voice processes
python "%ROOT%\Scripts\stop_voice_stack.py" --ports %REALTIME_PORT% %LLAMA_SERVER_PORT% || exit /b 1

python "%ROOT%\Scripts\patch_speech_to_speech_timeout.py" || exit /b 1

python -c "import huggingface_hub, numpy, pydantic, torch, torchaudio, transformers, sys; from packaging.version import Version; ok = transformers.__version__ == '4.57.3' and Version('0.34.0') <= Version(huggingface_hub.__version__) < Version('1.0') and numpy.__version__ == '1.26.4' and pydantic.__version__ == '2.13.4' and torch.__version__ == '2.6.0+cu124' and torchaudio.__version__ == '2.6.0+cu124'; sys.exit(0 if ok else 1)" >nul 2>&1
if errorlevel 1 (
  echo [voice-test][error] The shared Python environment has incompatible package versions.
  echo Run Scripts\setup.bat to restore the validated voice and vision environment.
  exit /b 1
)

if "%LLAMA_SERVER_EXE%"=="llama-server" (
  where llama-server >nul 2>&1
  if errorlevel 1 (
    echo [voice-test][error] llama-server was not found on PATH.
    echo Install llama.cpp, make sure llama-server.exe is on PATH, then run this again.
    echo Download: https://github.com/ggml-org/llama.cpp/releases
    exit /b 1
  )
)

if not "%LLAMA_SERVER_EXE%"=="llama-server" if not exist "%LLAMA_SERVER_EXE%" (
  echo [voice-test][error] llama-server was not found at %LLAMA_SERVER_EXE%.
  echo Install llama.cpp, make sure llama-server.exe is on PATH, then run this again.
  echo Download: https://github.com/ggml-org/llama.cpp/releases
  exit /b 1
)

echo [voice-test] Starting llama-server for realtime LLM on %REALTIME_LLM_BASE_URL%
echo [voice-test] Model: %REALTIME_MODEL%
echo [voice-test] llama-server: %LLAMA_SERVER_EXE%
start "Robit llama-server" /min cmd /c ""%LLAMA_SERVER_EXE%" -hf %REALTIME_MODEL% --host 127.0.0.1 --port %LLAMA_SERVER_PORT% -np 2 -c 65536 -fa on --swa-full --reasoning off --image-max-tokens %ROBIT_VISION_IMAGE_TOKENS%"
echo [voice-test] Waiting for llama-server /v1/responses
python "%ROOT%\Scripts\prewarm_responses.py" --base-url "%REALTIME_LLM_BASE_URL%" --model "%REALTIME_MODEL%" --attempts 90 --sleep 2 --target-seconds 180 || exit /b 1

echo [voice-test] Downloading voice sidecar models if needed
python "%ROOT%\Scripts\download_voice_models.py" "%REALTIME_STT_MODEL%" "%REALTIME_TTS_MODEL%" || exit /b 1

echo [voice-test] Starting realtime voice sidecar on ws://localhost:%REALTIME_PORT%/v1/realtime
echo [voice-test] Pipeline: silero-vad to %REALTIME_STT_MODEL% to %REALTIME_MODEL% to %REALTIME_TTS_MODEL% (%REALTIME_VOICE%)
echo [voice-test] Sidecar log: %S2S_LOG%
echo [voice-test] Watch log: Get-Content "%S2S_LOG%" -Wait
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

echo [voice-test] Starting standalone voice client. Press Ctrl+C to stop.
python "%ROOT%\Scripts\voice_test.py" --sidecar-log "%S2S_LOG%" %*

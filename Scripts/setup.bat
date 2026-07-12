@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
set "LLAMA_SERVER_EXE=llama-server"
set "PYTHON311="

echo [setup] HouseGremlin setup starting
cd /d "%PC_BRAIN%" || exit /b 1

py -3.11 --version >nul 2>&1
if not errorlevel 1 (
  set "PYTHON311=py -3.11"
) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
  set "PYTHON311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
  echo [setup][error] Python 3.11 was not found.
  echo Install Python 3.11, then run Scripts\setup.bat again.
  exit /b 1
)

if exist "%VENV%\Scripts\python.exe" (
  "%VENV%\Scripts\python.exe" --version >nul 2>&1
  if errorlevel 1 (
    echo [setup] Existing PC brain virtual environment is broken; recreating it
    rmdir /s /q "%VENV%" || exit /b 1
  )
)

if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] Creating virtual environment
  %PYTHON311% -m venv "%VENV%" || exit /b 1
) else (
  echo [setup] Reusing existing virtual environment
)

call "%VENV%\Scripts\activate.bat" || exit /b 1

if exist "C:\Tools\llama.cpp\llama-server.exe" set "LLAMA_SERVER_EXE=C:\Tools\llama.cpp\llama-server.exe"

echo [setup] Upgrading packaging tools
python -m pip install --upgrade pip "setuptools>=70,<81" wheel || exit /b 1

echo [setup] Installing PC brain requirements
python -m pip install -r requirements.txt || exit /b 1

if not exist ".env" (
  echo [setup] Creating pc_brain\.env from .env.example
  copy ".env.example" ".env" >nul || exit /b 1
) else (
  echo [setup] Keeping existing pc_brain\.env
)

echo [setup] Running endpoint tests
cd /d "%PC_BRAIN%" || exit /b 1
python -m pytest tests\test_main_endpoints.py || exit /b 1

where ollama >nul 2>&1
if not errorlevel 1 (
  echo [setup] Pulling default local model gemma4:e4b
  ollama pull gemma4:e4b
) else (
  echo [setup] Ollama was not found on PATH. Install it before using typed Text mode.
)

if "%LLAMA_SERVER_EXE%"=="llama-server" (
  where llama-server >nul 2>&1
  if errorlevel 1 (
    echo [setup] llama-server was not found on PATH. Install llama.cpp before using realtime voice.
    echo [setup] Download: https://github.com/ggml-org/llama.cpp/releases
    echo [setup] voice_test.bat will download the Gemma E4B GGUF through llama-server on first run.
  ) else (
    echo [setup] llama-server found for realtime voice.
  )
)

if not "%LLAMA_SERVER_EXE%"=="llama-server" (
  echo [setup] llama-server found at %LLAMA_SERVER_EXE%.
)

echo.
echo [setup] Done.
echo To start Robit later:
echo   Scripts\run.bat
echo.

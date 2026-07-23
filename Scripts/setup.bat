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

python -c "import torch, torchaudio, sys; sys.exit(0 if torch.__version__ == '2.6.0+cu124' and torchaudio.__version__ == '2.6.0+cu124' else 1)" >nul 2>&1
if errorlevel 1 (
  echo [setup] Installing validated CUDA Torch and Torchaudio pair
  python -m pip install --force-reinstall "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
)

echo [setup] Installing PC brain requirements
python -m pip install -r requirements.txt || exit /b 1

python -c "import huggingface_hub, numpy, pydantic, torch, torchaudio, transformers, sys; from packaging.version import Version; ok = transformers.__version__ == '4.57.3' and Version('0.34.0') <= Version(huggingface_hub.__version__) < Version('1.0') and numpy.__version__ == '1.26.4' and pydantic.__version__ == '2.13.4' and torch.__version__ == '2.6.0+cu124' and torchaudio.__version__ == '2.6.0+cu124'; sys.exit(0 if ok else 1)" >nul 2>&1
if errorlevel 1 (
  echo [setup][error] Package installation did not produce the validated voice and vision environment.
  exit /b 1
)

if not exist ".env" (
  echo [setup] Creating pc_brain\.env from .env.example
  copy ".env.example" ".env" >nul || exit /b 1
) else (
  echo [setup] Keeping existing pc_brain\.env
)

echo [setup] Running PC brain tests
cd /d "%PC_BRAIN%" || exit /b 1
python -m pytest tests || exit /b 1

if "%LLAMA_SERVER_EXE%"=="llama-server" (
  where llama-server >nul 2>&1
  if errorlevel 1 (
    echo [setup] llama-server was not found on PATH. Install llama.cpp for the shared E4B language backend.
    echo [setup] Download: https://github.com/ggml-org/llama.cpp/releases
    echo [setup] run.bat will download the Gemma 4 E4B GGUF through llama-server on first run.
  ) else (
    echo [setup] llama-server found for the shared E4B language backend.
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

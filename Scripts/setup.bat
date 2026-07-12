@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
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

if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] Creating virtual environment
  %PYTHON311% -m venv "%VENV%" || exit /b 1
) else (
  echo [setup] Reusing existing virtual environment
)

call "%VENV%\Scripts\activate.bat" || exit /b 1

echo [setup] Upgrading packaging tools
python -m pip install --upgrade pip setuptools wheel || exit /b 1

echo [setup] Installing ANTLR runtime workaround
python -m pip install --only-binary=:all: antlr4-python3-runtime==4.9.3
if errorlevel 1 (
  echo [setup] Binary ANTLR wheel was unavailable; trying no-build-isolation fallback
  python -m pip install antlr4-python3-runtime==4.9.3 --no-build-isolation || exit /b 1
)

echo [setup] Installing PC brain requirements
python -m pip install -r requirements.txt || exit /b 1

if not exist ".env" (
  echo [setup] Creating pc_brain\.env from .env.example
  copy ".env.example" ".env" >nul || exit /b 1
) else (
  echo [setup] Keeping existing pc_brain\.env
)

echo [setup] Running endpoint tests
python -m pytest tests\test_main_endpoints.py || exit /b 1

echo.
echo [setup] Done.
echo To start Robit later:
echo   Scripts\run.bat 172.22.1.176
echo.

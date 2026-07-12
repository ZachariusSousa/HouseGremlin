@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "PC_BRAIN=%ROOT%\pc_brain"
set "VENV=%PC_BRAIN%\.venv"
set "DEFAULT_ROBOT_HOST=robit.local"
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
) else (
  echo [run] Ollama was not found on PATH. Chat will fail until Ollama is running.
)

echo [run] Opening http://localhost:%PORT%
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:%PORT%"

echo [run] Starting PC brain. Press Ctrl+C in this window to stop.
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port %PORT%

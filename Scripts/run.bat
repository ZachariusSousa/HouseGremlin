@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "BRAIN_PYTHON=%ROOT%\pc_brain\.venv\Scripts\python.exe"

if not exist "%BRAIN_PYTHON%" (
  echo [run][error] pc_brain\.venv was not found.
  echo Run Scripts\setup.bat first.
  exit /b 1
)

"%BRAIN_PYTHON%" --version >nul 2>&1
if errorlevel 1 (
  echo [run][error] pc_brain\.venv points to a missing or incompatible Python.
  echo Install 64-bit Python 3.11, then run Scripts\setup.bat.
  exit /b 1
)

"%BRAIN_PYTHON%" "%ROOT%\Scripts\supervisor.py" %*
exit /b %ERRORLEVEL%

@echo off
py -3.13 "%~dp0Scripts\stop_stack.py" %*
exit /b %ERRORLEVEL%

@echo off
title AI Object Detection System
cd /d "%~dp0"

echo Starting AI Object Detection System...
echo.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)
echo.
echo Detection system has stopped.
pause

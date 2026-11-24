@echo off
title AI Object Detection Setup
cd /d "%~dp0"

echo Checking Python version (3.11 or 3.12 recommended)...
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)"
if errorlevel 1 (
    echo Please install Python 3.11 or 3.12 and run setup again.
    goto failed
)

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto failed

echo Installing project dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Installing optional hand-landmark support...
.venv\Scripts\python.exe -m pip install -r requirements-hands.txt
if errorlevel 1 (
    echo WARNING: MediaPipe hand support could not be installed.
    echo Hand detection will be unavailable. Check Python compatibility and internet.
) else (
    .venv\Scripts\python.exe -c "import mediapipe as mp; h=mp.solutions.hands.Hands(max_num_hands=2); h.close(); print('MediaPipe Hands API verified. Version:', mp.__version__)"
    if errorlevel 1 (
        echo WARNING: MediaPipe installed, but Hands API initialization failed.
        echo Hand detection may not work. See the error above.
    )
)

echo.
echo Setup completed. Run start_detection.bat
pause
exit /b 0

:failed
echo Setup failed. Check Python installation and network connection.
pause
exit /b 1

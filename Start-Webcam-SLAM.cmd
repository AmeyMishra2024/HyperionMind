@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. See README.md for installation instructions.
  pause
  exit /b 1
)
echo Open http://127.0.0.1:8765/slam and click Start mapping.
echo Keep this window open. Press Ctrl+C to stop the server.
".venv\Scripts\python.exe" tools\run_webcam_slam.py
pause

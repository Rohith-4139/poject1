@echo off
title Hypertension PPG - Server
cd /d "%~dp0backend"
echo Starting Flask server...
echo Open in browser: http://127.0.0.1:5000
echo.
start "" http://127.0.0.1:5000
"venv311\Scripts\python.exe" app.py
pause

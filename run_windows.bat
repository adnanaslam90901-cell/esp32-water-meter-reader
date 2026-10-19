@echo off
title ESP32 Water Meter Reader
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8000
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause

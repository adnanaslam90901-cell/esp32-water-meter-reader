@echo off
title Install ESP32 Water Meter Reader
where py >nul 2>nul || (echo Python is not installed. Install Python 3.11 or 3.12 from python.org and enable Add Python to PATH.& pause & exit /b 1)
py -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Installation complete. Double-click run_windows.bat.
pause

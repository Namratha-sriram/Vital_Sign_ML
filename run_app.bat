@echo off
title VitalSign Healthcare Machine Learning System
color 0B
echo =====================================================================
echo                 VitalSign Clinical ML System Launcher
echo =====================================================================
echo.

cd /d "%~dp0"

set "PY_CMD="
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=.venv\Scripts\python.exe"
)
if not defined PY_CMD (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -3"
    )
)
if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 (
        python --version >nul 2>&1
        if not errorlevel 1 set "PY_CMD=python"
    )
)
if not defined PY_CMD (
    echo [Error] Python 3 is not installed or could not be started.
    echo Install Python 3.11 or newer, then run this launcher again.
    pause
    exit /b 1
)

echo [1/2] Checking Python environment...
call %PY_CMD% --version

echo.
echo [2/2] Launching VitalSign Web Application on http://127.0.0.1:5000 ...
echo =====================================================================
echo Open your web browser and navigate to: http://127.0.0.1:5000
echo =====================================================================
call %PY_CMD% app.py

pause

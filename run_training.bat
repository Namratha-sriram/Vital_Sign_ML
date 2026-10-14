@echo off
title VitalSign Model Training & Comparison Pipeline
color 0A
echo =====================================================================
echo    VitalSign ML Model Training & Faculty Comparison (50k Dataset)
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

echo [1/2] Training 5 Supervised Classifiers on 50,000 Records...
call %PY_CMD% src/train_models.py

echo.
echo [2/2] Generating Faculty Before vs After Comparison Benchmark...
call %PY_CMD% src/compare_before_after_M3.py

echo.
echo =====================================================================
echo [OK] Pipeline Complete! All models, metrics, and Before vs After 
echo      comparison plots generated in models/, output/, and static/images/
echo =====================================================================
pause

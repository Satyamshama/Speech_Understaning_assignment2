@echo off
REM Setup script for Speech Assignment 2 (Windows)

echo ===================================
echo Speech Assignment 2 Setup Script
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.10+
    exit /b 1
)

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ===================================
    echo Setup Complete!
    echo ===================================
    echo.
    echo To run the pipeline:
    echo   python pipeline.py
    echo.
    echo Individual parts can be run with:
    echo   python Scripts\PA2_Part1_STT.py
    echo   python Scripts\PA2_Part2_Phonetic.py
    echo   python Scripts\PA2_Part3_TTS.py
    echo   python Scripts\PA2_Part4_Adversarial.py
) else (
    echo Installation failed. Please check the error messages above.
    exit /b 1
)

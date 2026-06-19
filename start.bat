@echo off
chcp 65001 >nul
title Exo AI Server

cd /d "%~dp0"

echo ========================================
echo   Starting Exo AI Server
echo ========================================
echo.

set VENV_DIR=%~dp0.venv
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    echo [INFO] Using virtual environment: %VENV_PYTHON%
    goto :run
)

echo [ERROR] Virtual environment not found
echo Please run: install.bat
echo.
pause
exit /b 1

:run
if not exist "config.json" (
    echo [INFO] Creating config file...
    copy config.example.json config.json >nul
    echo.
    echo Please edit config.json to configure the node
    notepad config.json
    pause
    exit /b 1
)

echo [INFO] Python: %VENV_PYTHON%
echo.

"%VENV_PYTHON%" exo_launcher.py %*

if errorlevel 1 (
    echo.
    echo ========================================
    echo   Server stopped with error
    echo ========================================
    pause
)

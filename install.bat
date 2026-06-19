@echo off
chcp 65001 >nul
title Exo Installer

cd /d "%~dp0"

echo ========================================
echo   Exo Distributed Inference Framework
echo   Installation Wizard
echo ========================================
echo.

set VENV_DIR=%~dp0.venv
set PYTHON_DIR=%~dp0python\windows
set PYTHON_EXE=%PYTHON_DIR%\python.exe

:: Check if Python already exists
if exist "%PYTHON_EXE%" (
    echo [INFO] Found embedded Python: %PYTHON_EXE%
    goto :create_venv
)

:: Try system Python first
where python >nul 2>&1
if %errorlevel%==0 (
    echo [INFO] Found system Python
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
    goto :create_venv
)

:: Download embedded Python
echo [1/3] Downloading Python 3.12...
echo.

if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"

powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip' -OutFile '%PYTHON_DIR%\python.zip' -UseBasicParsing"

if not exist "%PYTHON_DIR%\python.zip" (
    echo [ERROR] Failed to download Python
    pause
    exit /b 1
)

echo Extracting...
powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%PYTHON_DIR%\python.zip' -DestinationPath '%PYTHON_DIR%' -Force"
del "%PYTHON_DIR%\python.zip"

echo Configuring Python...
set PTH_FILE=%PYTHON_DIR%\python312._pth
(
echo python312.zip
echo .
echo Lib
echo Lib\site-packages
echo import site
) > "%PTH_FILE%"

echo Installing pip...
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%PYTHON_DIR%\get-pip.py' -UseBasicParsing"
"%PYTHON_EXE%" "%PYTHON_DIR%\get-pip.py"
del "%PYTHON_DIR%\get-pip.py" 2>nul

echo Done!

:create_venv
echo.
echo [2/3] Creating virtual environment...
echo.

if exist "%VENV_DIR%" (
    echo [INFO] Virtual environment already exists
) else (
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    echo [INFO] Virtual environment created
)

set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

echo.
echo [3/3] Installing dependencies...
echo This may take several minutes...
echo.

"%VENV_PIP%" install --upgrade pip --quiet 2>nul

echo Installing PyTorch...
"%VENV_PIP%" install "torch>=2.0.0,<2.7.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121

echo Installing transformers (compatible version)...
"%VENV_PIP%" install "transformers>=4.40.0,<4.56.0" "tokenizers" "safetensors" "huggingface-hub" "accelerate"

echo Installing other dependencies...
"%VENV_PIP%" install -r requirements.txt

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Python: %VENV_PYTHON%
echo Virtual Environment: %VENV_DIR%
echo.
echo Next steps:
echo   1. Edit config.json to configure the node
echo   2. Run start.bat to launch the server
echo.

if not exist "config.json" (
    echo [INFO] Creating config.json from template...
    copy config.example.json config.json >nul
)

pause

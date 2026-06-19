#!/bin/bash

cd "$(dirname "$0")"

echo "========================================"
echo "  Starting Exo AI Server"
echo "========================================"
echo ""

LIB_DIR="$(pwd)/lib"
VENV_DIR="$(pwd)/.venv"

PYTHON_DIR="$(pwd)/python/linux"
PYTHON_EXE="$PYTHON_DIR/bin/python3"

if [ ! -f "$PYTHON_EXE" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_EXE=$(which python3)
        echo "[INFO] Using system Python: $PYTHON_EXE"
    else
        echo "[ERROR] Python not found"
        echo "Please run: ./install.sh"
        echo ""
        exit 1
    fi
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON_EXE="$VENV_DIR/bin/python"
    echo "[INFO] Using virtual environment: $PYTHON_EXE"
elif [ -d "$VENV_DIR" ]; then
    PYTHON_EXE="$VENV_DIR/bin/python"
fi

if [ ! -f "$VENV_DIR/bin/python" ] && [ -d "$LIB_DIR" ]; then
    echo "[WARNING] Virtual environment not found, using lib directory"
    export PYTHONPATH="$LIB_DIR:$(pwd)"
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    export PYTHONPATH=""
fi

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ ! -f "config.json" ]; then
    echo "[INFO] Creating config file..."
    cp config.example.json config.json
    echo ""
    echo "Please edit config.json to configure the node"
    exit 1
fi

echo "[INFO] Python: $PYTHON_EXE"
echo ""

"$PYTHON_EXE" exo_launcher.py "$@"

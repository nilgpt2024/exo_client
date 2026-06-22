#!/bin/bash

cd "$(dirname "$0")"

VENV_DIR="$(pwd)/.venv"

if [ -z "$PYTHON_EXE" ]; then
    PYTHON_EXE="$(pwd)/python/linux/bin/python3"
fi

if [ ! -f "$PYTHON_EXE" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_EXE=$(which python3)
    else
        echo "[ERROR] Python not found"
        exit 1
    fi
fi

echo "[2/2] Installing dependencies..."
echo "This may take several minutes..."
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_EXE -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

$VENV_PIP install --upgrade pip --quiet 2>/dev/null || true

echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA 12.1 version..."
    $VENV_PIP install "torch>=2.0.0,<2.7.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected, installing CPU version..."
    $VENV_PIP install "torch>=2.0.0,<2.7.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing transformers (compatible version)..."
$VENV_PIP install "transformers>=4.40.0,<4.56.0" "tokenizers" "safetensors" "huggingface-hub" "accelerate"

echo "Installing other dependencies..."
$VENV_PIP install -r requirements.txt

echo ""
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo ""
echo "Python: $VENV_PYTHON"
echo "Virtual Environment: $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Edit config.json to configure the node"
echo "  2. Run ./start.sh to launch the server"
echo ""

if [ ! -f "config.json" ]; then
    echo "[INFO] Creating config.json from template..."
    cp config.example.json config.json
fi

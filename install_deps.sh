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

create_venv() {
    local py="$1"
    local venv_dir="$2"

    # 如果已存在但可能损坏（没有可用的 pip/python），先删除重建
    if [ -d "$venv_dir" ]; then
        if [ -x "$venv_dir/bin/python" ] && [ -x "$venv_dir/bin/pip" ]; then
            echo "[INFO] Virtual environment already exists."
            return 0
        fi
        echo "[WARN] Existing virtual environment appears incomplete, recreating..."
        rm -rf "$venv_dir"
    fi

    echo "Creating virtual environment..."

    # 首次尝试标准 venv
    if "$py" -m venv "$venv_dir" 2>/tmp/venv_error.log; then
        return 0
    fi

    echo "[WARN] Standard venv creation failed, attempting to fix..."
    cat /tmp/venv_error.log

    # Debian/Ubuntu/Colab 环境通常缺少 python3-venv
    if command -v apt-get &> /dev/null; then
        echo "[INFO] Installing python3-venv and python3-pip..."
        apt-get update -qq
        apt-get install -y -qq python3-venv python3-pip || true
    fi

    # 再次尝试
    if "$py" -m venv "$venv_dir" 2>/tmp/venv_error2.log; then
        return 0
    fi

    echo "[WARN] venv still failing, falling back to --without-pip + get-pip.py..."
    cat /tmp/venv_error2.log

    # 降级方案：不带 pip 创建 venv，再手动安装 pip
    "$py" -m venv "$venv_dir" --without-pip
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$venv_dir/bin/python" /tmp/get-pip.py
    rm -f /tmp/get-pip.py
}

create_venv "$PYTHON_EXE" "$VENV_DIR"

# 修复 venv 中脚本的 shebang 路径（解决虚拟环境移动/复制后路径失效的问题）
echo "Fixing virtual environment paths..."
VENV_PYTHON="$VENV_DIR/bin/python"
for script in "$VENV_DIR"/bin/*; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        # 替换 shebang 行为当前正确的 python 路径
        sed -i "1s|#!.*|#!$VENV_PYTHON|" "$script" 2>/dev/null || true
    fi
done

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

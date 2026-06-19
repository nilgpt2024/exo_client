#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "========================================"
echo "  Exo Distributed Inference Framework"
echo "  Installation Wizard"
echo "========================================"
echo ""

PYTHON_DIR="$(pwd)/python/linux"
LIB_DIR="$(pwd)/lib"
PYTHON_EXE="$PYTHON_DIR/bin/python3"

if [ -f "$PYTHON_EXE" ]; then
    echo "[INFO] Python already installed: $PYTHON_EXE"
    echo ""
    . ./install_deps.sh
    exit 0
fi

if command -v python3 &> /dev/null; then
    echo "[INFO] Found system Python: $(which python3)"
    PYTHON_EXE=$(which python3)
    export PYTHON_EXE
    . ./install_deps.sh
    exit 0
fi

echo "[1/2] Downloading Python 3.12..."
echo ""

OS_TYPE=$(uname -m)
if [ "$OS_TYPE" = "x86_64" ]; then
    PYTHON_URL="https://www.python.org/ftp/python/3.12.9/Python-3.12.9.tgz"
else
    PYTHON_URL="https://www.python.org/ftp/python/3.12.9/Python-3.12.9.tgz"
fi

mkdir -p "$PYTHON_DIR"

echo "Downloading Python..."
curl -fsSL "$PYTHON_URL" -o /tmp/python.tgz

echo "Extracting..."
cd /tmp
tar xzf python.tgz
cd Python-3.12.9

echo "Configuring..."
./configure --prefix="$PYTHON_DIR" --enable-optimizations --quiet

echo "Building (this may take a few minutes)..."
make -j$(nproc) --quiet
make install --quiet

cd "$(dirname "$0")"
rm -rf /tmp/python.tgz /tmp/Python-3.12.9

echo "Done!"
echo ""

. ./install_deps.sh

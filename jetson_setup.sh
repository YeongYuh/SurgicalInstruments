#!/bin/bash
# One-time setup script for Jetson Nano (JetPack 4.6.x, Ubuntu 18.04, aarch64).
# Run this once after copying the project to the Jetson.
# Requires internet access and sudo privileges.
set -e
cd "$(dirname "$0")"

echo "================================================================"
echo " Jetson Nano Setup — Surgical Instrument Counting System"
echo "================================================================"
echo ""

# ── Step 1: Python 3.8 ──────────────────────────────────────────────
echo "[1/6] Installing Python 3.8 (via deadsnakes PPA)..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -q
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev
echo "      Python 3.8 installed: $(python3.8 --version)"
echo ""

# ── Step 2: Remove old (x86-64) venv ───────────────────────────────
if [ -d venv ]; then
  echo "[2/6] Removing old virtual environment (likely x86-64, unusable on ARM64)..."
  rm -rf venv
else
  echo "[2/6] No existing venv found — skipping removal."
fi
echo ""

# ── Step 3: Create new ARM64 venv ──────────────────────────────────
echo "[3/6] Creating new Python 3.8 virtual environment..."
echo "      Using --system-site-packages so the venv inherits system OpenCV 4.1.1"
python3.8 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip --quiet
echo "      venv ready."
echo ""

# ── Step 4: PyTorch (manual) ────────────────────────────────────────
echo "[4/6] PyTorch for JetPack 4.6 (CUDA 10.2)..."
echo ""
echo "  PyTorch is NOT on PyPI for Jetson — you must download it from NVIDIA."
echo ""
echo "  Download URL (NVIDIA Developer Forums):"
echo "    https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo ""
echo "  Recommended file for JetPack 4.6 + Python 3.8:"
echo "    torch-1.10.0-cp38-cp38-linux_aarch64.whl  (~530 MB)"
echo ""
echo "  After downloading, run:"
echo "    source venv/bin/activate"
echo "    pip install /path/to/torch-1.10.0-cp38-cp38-linux_aarch64.whl"
echo ""
echo "  Then re-run this script (or skip to step 5 manually)."
echo ""

# Check if torch is already installed in the venv
if python3 -c "import torch" 2>/dev/null; then
  TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
  echo "  torch $TORCH_VER is already installed — skipping download prompt."
else
  echo "  torch is NOT yet installed. Install the wheel above, then run:"
  echo "    pip install -r requirements.txt"
  echo ""
  echo "  Continuing setup (remaining steps will still complete)..."
fi
echo ""

# ── Step 5: Python dependencies ────────────────────────────────────
echo "[5/6] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt
echo ""

# ── Step 6: Required directories ───────────────────────────────────
echo "[6/6] Creating required output directories..."
mkdir -p output/uploads output/annotated output/frames models input
echo ""

# ── Done ────────────────────────────────────────────────────────────
echo "================================================================"
echo " Setup complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Place your trained YOLO model at:"
echo "       $(pwd)/models/best.pt"
echo ""
echo "  2. If using the serial scale, add yourself to the dialout group:"
echo "       sudo usermod -aG dialout \$USER"
echo "     Then log out and back in."
echo ""
echo "  3. (Optional) Set Jetson to max performance mode:"
echo "       sudo nvpmodel -m 0"
echo "       sudo jetson_clocks"
echo ""
echo "  4. Start the web server:"
echo "       ./run_jetson.sh"
echo "     Or with a real serial scale:"
echo "       SCALE_READER_MODE=serial SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh"
echo ""

#!/bin/bash
# Jetson Nano startup script — prints diagnostics, then launches the web server.
#
# Default scale configuration (override by setting env vars before calling):
#   SCALE_READER_MODE=serial   # "serial" or "mock"
#   SERIAL_PORT=/dev/ttyUSB0   # USB scale/Arduino path
#   SERIAL_BAUDRATE=9600       # baud rate
#
# Examples:
#   ./run_jetson.sh                                      # serial scale on /dev/ttyUSB0
#   SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh             # Arduino on ACM0
#   SCALE_READER_MODE=mock ./run_jetson.sh               # no hardware, mock weight
#   WEBCAM_INDEX=1 BACKEND_PORT=8080 ./run_jetson.sh     # custom camera / port
set -e
cd "$(dirname "$0")"

if [ ! -f venv/bin/activate ]; then
  echo "[ERROR] Virtual environment not found."
  echo "  Run ./jetson_setup.sh first."
  exit 1
fi

source venv/bin/activate

# ── Scale defaults (only set if not already in environment) ──────────────
export SCALE_READER_MODE="${SCALE_READER_MODE:-serial}"
export SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"
export SERIAL_BAUDRATE="${SERIAL_BAUDRATE:-9600}"

# ── Startup diagnostics ──────────────────────────────────────────────────
python3 - <<'PYEOF'
import sys, platform, os

print("=" * 56)
print(" Jetson Startup Diagnostics")
print("=" * 56)
print(f"  Python     : {sys.version.split()[0]}")
print(f"  Arch       : {platform.machine()}")
print(f"  OS         : {platform.platform()}")

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  PyTorch    : {torch.__version__}  (CUDA={'yes' if cuda_ok else 'no'})")
except ImportError:
    print("  PyTorch    : NOT installed")
    print("               -> pip install <jetson-torch-wheel>")

try:
    import cv2
    print(f"  OpenCV     : {cv2.__version__}")
except ImportError:
    print("  OpenCV     : NOT installed")
    print("               -> create venv with --system-site-packages")

try:
    import ultralytics
    print(f"  ultralytics: {ultralytics.__version__}")
except ImportError:
    print("  ultralytics: NOT installed  -> pip install -r requirements.txt")

import flask
print(f"  Flask      : {flask.__version__}")

model = os.environ.get("MODEL_PATH", "models/best.pt")
print(f"  Model      : {model}  ({'OK' if os.path.exists(model) else 'MISSING'})")
if not os.path.exists(model):
    print("               -> place trained best.pt at the path above")

cam = os.environ.get("WEBCAM_INDEX", "0")
print(f"  Camera     : /dev/video{cam}  (index={cam})")

mode = os.environ.get("SCALE_READER_MODE", "serial")
port = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
baud = os.environ.get("SERIAL_BAUDRATE", "9600")
print(f"  Scale mode : {mode}")
print(f"  Scale port : {port}  (baud={baud})")

if mode == "serial":
    if os.path.exists(port):
        print(f"               -> port found OK")
    else:
        print(f"  [WARNING]  {port} not found!")
        print("               -> check USB cable:  ls /dev/ttyUSB*  ls /dev/ttyACM*")
        print("               -> set port:          SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh")
        print("               -> permission fix:    sudo usermod -aG dialout $USER  (re-login)")
        print("               -> no hardware:       SCALE_READER_MODE=mock ./run_jetson.sh")

print("=" * 56)
PYEOF

# ── Output directories ───────────────────────────────────────────────────
mkdir -p output/uploads output/annotated output/frames

export FLASK_APP=app.web
export PYTHONPATH="$(pwd)"

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
PORT="${BACKEND_PORT:-5000}"

echo ""
echo "Scale : ${SCALE_READER_MODE}  port=${SERIAL_PORT}  baud=${SERIAL_BAUDRATE}"
echo ""
echo "Web UI:"
echo "  Local  : http://localhost:${PORT}"
if [ -n "$LOCAL_IP" ]; then
  echo "  Network: http://${LOCAL_IP}:${PORT}"
fi
echo ""
echo "Press Ctrl+C to stop."
echo ""

python -m flask run \
  --host="${BACKEND_HOST:-0.0.0.0}" \
  --port="${PORT}" \
  --no-debugger \
  --no-reload

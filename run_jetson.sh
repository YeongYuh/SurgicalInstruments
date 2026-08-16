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

# ── Active model package ──────────────────────────────────────────────────────
# Which surgical instrument family to load.  The package manifest under
# model_packages/<id>/manifest.json is the source of truth for the adapter, the
# model file, the per-class weights, and the default standards.
#   ACTIVE_MODEL_PACKAGE=demo ./run_jetson.sh
export ACTIVE_MODEL_PACKAGE="${ACTIVE_MODEL_PACKAGE:-ortho_tka}"
export MODEL_PACKAGES_DIR="${MODEL_PACKAGES_DIR:-$(pwd)/model_packages}"

# ── Camera source (integer index or /dev/videoN path) ────────────────────────
# Default to path-based open — more reliable than integer index on Jetson OpenCV.
# Use CAMERA_SOURCE=/dev/video1 ./run_jetson.sh if /dev/video0 is wrong device.
export CAMERA_SOURCE="${CAMERA_SOURCE:-/dev/video0}"

# ── Camera capture parameters ─────────────────────────────────────────────────
# YOLO inference runs in a background thread every WEBCAM_DETECTION_INTERVAL s.
# CAMERA_FOURCC options:
#   MJPG  — camera sends native JPEG frames (fast, some cameras log libjpeg warnings)
#   YUYV  — raw YUV, no decode warnings but slightly more CPU to re-encode
#   AUTO  — let the V4L2 driver negotiate
# If you see "Corrupt JPEG data" warnings and unstable preview, switch to YUYV:
#   CAMERA_FOURCC=YUYV ./run_jetson.sh
export WEBCAM_WIDTH="${WEBCAM_WIDTH:-640}"
export WEBCAM_HEIGHT="${WEBCAM_HEIGHT:-480}"
export WEBCAM_FPS="${WEBCAM_FPS:-15}"
export CAMERA_FOURCC="${CAMERA_FOURCC:-YUYV}"
export WEBCAM_DETECTION_INTERVAL="${WEBCAM_DETECTION_INTERVAL:-5}"

# ── Safety gate for the web-UI shutdown button ────────────────────────────
# Keep false unless sudoers is configured (see docs/DEPLOY_JETSON.md §13).
export ENABLE_SYSTEM_SHUTDOWN="${ENABLE_SYSTEM_SHUTDOWN:-false}"

# ── Startup diagnostics ──────────────────────────────────────────────────
export PYTHONPATH="$(pwd)"
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

# ── Active model package ─────────────────────────────────────────────────
# Delegates to the application's own manifest loader — the shell must never
# grow a second, divergent parser.
active_pkg = os.environ.get("ACTIVE_MODEL_PACKAGE", "ortho_tka")
try:
    from pathlib import Path
    import app.config as _cfg
    from app.inference import discover_packages

    found = discover_packages(Path(_cfg.MODEL_PACKAGES_DIR), project_root=_cfg.PROJECT_ROOT)
    print(f"  Packages   : {', '.join(sorted(found)) or '(none found)'}")
    entry = found.get(active_pkg)
    if entry is None:
        print(f"  [ERROR]    Active package '{active_pkg}' not found in {_cfg.MODEL_PACKAGES_DIR}")
        print(f"             -> set ACTIVE_MODEL_PACKAGE to one of the packages above")
    elif entry.package is None:
        print(f"  [ERROR]    Active package '{active_pkg}' is invalid:")
        print(f"             {entry.error}")
    else:
        pkg = entry.package
        print(f"  Package    : {pkg.id}")
        print(f"  Name       : {pkg.display_name}  ({pkg.department or 'n/a'})")
        print(f"  Adapter    : {pkg.adapter}")
        if pkg.is_template:
            print(f"  [WARNING]  '{pkg.id}' is a TEMPLATE — fill in its manifest before use")
        model_file = str(pkg.model_file) if pkg.model_file else "(none)"
        print(f"  Model      : {model_file}  ({'OK' if pkg.model_file_exists() else 'MISSING'})")
        if not pkg.model_file_exists():
            if pkg.fallback_exists():
                print(f"               -> falling back to {pkg.fallback_model_file}")
            else:
                fb = pkg.fallback_model_file or "(none declared)"
                print(f"               -> fallback also missing: {fb}")
                print(f"               -> export best.onnx: python3 -c \"from ultralytics import "
                      f"YOLO; YOLO('models/best.pt').export(format='onnx', opset=12)\"")
        print(f"  Weights    : {pkg.class_weights_path}")
except Exception as exc:
    print(f"  [ERROR]    Cannot read model packages: {exc}")

import glob
cam_src      = os.environ.get("CAMERA_SOURCE", os.environ.get("WEBCAM_INDEX", "0"))
cam_w        = os.environ.get("WEBCAM_WIDTH", "640")
cam_h        = os.environ.get("WEBCAM_HEIGHT", "480")
cam_fps      = os.environ.get("WEBCAM_FPS", "15")
cam_fourcc   = os.environ.get("CAMERA_FOURCC", "MJPG")
det_interval = os.environ.get("WEBCAM_DETECTION_INTERVAL", "5")
video_devs   = sorted(glob.glob("/dev/video*"))
devs_str     = "  ".join(video_devs) if video_devs else "none found"
print(f"  Camera src : {cam_src}")
print(f"  Video devs : {devs_str}")
print(f"  Capture    : {cam_w}x{cam_h} @ {cam_fps} fps  fourcc={cam_fourcc}")
print(f"  Detect int : {det_interval} s  (preview ~15 FPS independent)")
if cam_fourcc == "MJPG":
    print("               (if 'Corrupt JPEG data' warnings appear, try CAMERA_FOURCC=YUYV)")
if cam_src.startswith("/") and not os.path.exists(cam_src):
    print(f"  [WARNING]  {cam_src} not found!")
    print(f"               -> available: {devs_str}")
    print(f"               -> try: CAMERA_SOURCE=/dev/video1 ./run_jetson.sh")
elif not cam_src.startswith("/"):
    expected = f"/dev/video{cam_src}"
    if not os.path.exists(expected):
        print(f"  [WARNING]  {expected} not found (index={cam_src})!")
        print(f"               -> available: {devs_str}")
        print(f"               -> try: CAMERA_SOURCE=/dev/video1 ./run_jetson.sh")

try:
    import onnxruntime as ort
    providers_str = ", ".join(ort.get_available_providers())
    print(f"  onnxruntime: {ort.__version__}  providers=[{providers_str}]")
except ImportError:
    print("  onnxruntime: NOT installed  -> pip install 'onnxruntime>=1.16,<1.20'")

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

shutdown_enabled = os.environ.get("ENABLE_SYSTEM_SHUTDOWN", "false").lower() == "true"
shutdown_label   = "ENABLED" if shutdown_enabled else "disabled  (set ENABLE_SYSTEM_SHUTDOWN=true to enable)"
print(f"  Shutdown   : {shutdown_label}")

print("=" * 56)
PYEOF

# ── Output directories ───────────────────────────────────────────────────
mkdir -p output/uploads output/annotated output/frames

export FLASK_APP=app.web
export PYTHONPATH="$(pwd)"

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
PORT="${BACKEND_PORT:-5000}"

echo ""
echo "Package  : ${ACTIVE_MODEL_PACKAGE}  (dir=${MODEL_PACKAGES_DIR})"
echo "Scale    : ${SCALE_READER_MODE}  port=${SERIAL_PORT}  baud=${SERIAL_BAUDRATE}"
echo "Camera   : source=${CAMERA_SOURCE}  detect_interval=${WEBCAM_DETECTION_INTERVAL}s"
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

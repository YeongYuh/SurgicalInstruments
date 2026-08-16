import os
from pathlib import Path

# Repository root — model package manifests resolve their relative paths
# against this, so the app behaves the same regardless of the caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_is_set(name: str) -> bool:
    """True only when the variable was explicitly provided by the environment."""
    return name in os.environ and os.environ[name] != ""


# Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
INPUT_PATH = os.environ.get("INPUT_PATH", "input/test_image.jpg")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# ── Model packages ───────────────────────────────────────────────────────────
# A model package bundles a manifest, an adapter name, a model file reference,
# per-class weights, and factory-default standards.  Switching instrument
# families (orthopaedics -> obstetrics -> ...) means switching packages, not
# editing platform code.
MODEL_PACKAGES_DIR   = os.environ.get("MODEL_PACKAGES_DIR", str(PROJECT_ROOT / "model_packages"))
ACTIVE_MODEL_PACKAGE = os.environ.get("ACTIVE_MODEL_PACKAGE", "ortho_tka")
# Per-package operator-editable configuration lives here.
PROFILES_DIR = os.environ.get("PROFILES_DIR", os.path.join(OUTPUT_DIR, "profiles"))
# The package that inherits the pre-package global output/standards.json and
# output/unit_weights.json on first start, so an in-service unit keeps its site
# configuration across the upgrade.
LEGACY_PACKAGE_ID = os.environ.get("LEGACY_PACKAGE_ID", "ortho_tka")

# Detection — manifest ``inference.confidence`` is the source of truth; this env
# var overrides it only when explicitly set.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
CONF_THRESHOLD_EXPLICIT = _env_is_set("CONF_THRESHOLD")

# ── Legacy model configuration (CLI only) ────────────────────────────────────
# app/main.py (the standalone image/webcam CLI) still uses these.  The web
# platform ignores them: the active package manifest decides the adapter, the
# model file, and the task.
CLASS_WEIGHT_PATH       = os.environ.get("CLASS_WEIGHT_PATH", "models/class_weight.json")
DETECTOR_BACKEND        = os.environ.get("DETECTOR_BACKEND", "pt")
ONNX_MODEL_PATH         = os.environ.get("ONNX_MODEL_PATH", "models/best.onnx")
ONNX_TASK               = os.environ.get("ONNX_TASK", "segment")
STRICT_DETECTOR_BACKEND = os.environ.get("STRICT_DETECTOR_BACKEND", "false").lower() == "true"

# Weight verification
EXPECTED_WEIGHT  = float(os.environ.get("EXPECTED_WEIGHT", "1520.35"))
WEIGHT_TOLERANCE = float(os.environ.get("WEIGHT_TOLERANCE", "0.5"))

# Scale reader — "mock" | "serial"
SCALE_READER_MODE   = os.environ.get("SCALE_READER_MODE", "mock")
MOCK_WEIGHT         = float(os.environ.get("MOCK_WEIGHT", "1520.35"))

# Serial settings
# Common Jetson Nano paths: /dev/ttyUSB0, /dev/ttyACM0
# Windows paths (COM3, COM4) must NOT be used on Jetson.
# Run: sudo usermod -aG dialout $USER  (then re-login) for serial access.
SERIAL_PORT         = os.environ.get("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUDRATE     = int(os.environ.get("SERIAL_BAUDRATE", "9600"))
# Timeout per readline() call.  Kept short (0.1 s) so the background scale
# poll thread never blocks longer than one interval when the serial buffer is
# momentarily empty.  Raise via env var if your Arduino outputs slower than 10 Hz.
SERIAL_TIMEOUT      = float(os.environ.get("SERIAL_TIMEOUT", "0.1"))
SERIAL_READ_RETRIES = int(os.environ.get("SERIAL_READ_RETRIES", "5"))

# How often the background scale poll thread drains the serial buffer (seconds).
# 0.1 s = 10 Hz — keeps the weight cache fresh without blocking the web threads.
SCALE_BG_POLL_INTERVAL = float(os.environ.get("SCALE_BG_POLL_INTERVAL", "0.1"))

# Scale signal quality — zero-rejection debounce + median filter
# Readings whose absolute value is <= threshold are treated as zero candidates.
SCALE_ZERO_THRESHOLD_GRAMS = float(os.environ.get("SCALE_ZERO_THRESHOLD_GRAMS", "2.0"))
# How many consecutive near-zero readings required before accepting zero.
SCALE_ZERO_CONFIRM_SAMPLES = int(os.environ.get("SCALE_ZERO_CONFIRM_SAMPLES", "3"))
# Rolling-median window size (last N accepted non-zero reads + confirmed zeros).
SCALE_FILTER_WINDOW        = int(os.environ.get("SCALE_FILTER_WINDOW", "3"))
# If the raw reading differs from current filtered value by >= this many grams,
# treat it as a real weight transition and flush the stale window.
SCALE_TRANSITION_THRESHOLD_GRAMS = float(os.environ.get("SCALE_TRANSITION_THRESHOLD_GRAMS", "5.0"))
# Log every raw/filtered sample when true.
SCALE_DEBUG                = os.environ.get("SCALE_DEBUG", "false").lower() == "true"

# ── Scale stability gate ─────────────────────────────────────────────────────
# A PASS/FAIL verdict is only issued for a reading that is both fresh and
# settled.  At the default 10 Hz background poll, a 1.5 s window holds ~15
# samples; stability needs at least SCALE_STABLE_MIN_SAMPLES of them spanning at
# least SCALE_STABLE_MIN_COVERAGE_RATIO of the window, with a spread no larger
# than SCALE_STABLE_RANGE_GRAMS.
SCALE_STABLE_WINDOW_SEC        = float(os.environ.get("SCALE_STABLE_WINDOW_SEC", "1.5"))
SCALE_STABLE_RANGE_GRAMS       = float(os.environ.get("SCALE_STABLE_RANGE_GRAMS", "1.0"))
SCALE_STABLE_MIN_SAMPLES       = int(os.environ.get("SCALE_STABLE_MIN_SAMPLES", "3"))
# Beyond this age the cached reading is reported fresh=false — a cached value
# must never masquerade as a live measurement after the scale is unplugged.
SCALE_MAX_SAMPLE_AGE_SEC       = float(os.environ.get("SCALE_MAX_SAMPLE_AGE_SEC", "2.0"))
SCALE_STABLE_MIN_COVERAGE_RATIO = float(os.environ.get("SCALE_STABLE_MIN_COVERAGE_RATIO", "0.5"))

# Source type — "image" | "webcam"
SOURCE_TYPE = os.environ.get("SOURCE_TYPE", "image")

# Webcam / camera settings
# CAMERA_SOURCE accepts:
#   integer index  : "0", "1"          → cv2.VideoCapture(0)
#   device path    : "/dev/video0"     → cv2.VideoCapture("/dev/video0")
#   GStreamer str  : "nvarguscamerasrc ..." → cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
# Falls back to WEBCAM_INDEX for backward compatibility.
def _parse_camera_source(raw: str):
    """Return int for numeric index, str otherwise."""
    try:
        return int(raw)
    except ValueError:
        return raw

_camera_source_raw = os.environ.get(
    "CAMERA_SOURCE", os.environ.get("WEBCAM_INDEX", "0")
)
CAMERA_SOURCE = _parse_camera_source(_camera_source_raw)
WEBCAM_INDEX  = CAMERA_SOURCE if isinstance(CAMERA_SOURCE, int) else 0  # backward compat

WEBCAM_WIDTH              = int(os.environ.get("WEBCAM_WIDTH", "1280"))
WEBCAM_HEIGHT             = int(os.environ.get("WEBCAM_HEIGHT", "720"))
WEBCAM_FPS                = int(os.environ.get("WEBCAM_FPS", "15"))
# CAMERA_FOURCC: "MJPG" | "YUYV" | "AUTO"
#   MJPG — camera sends native JPEG (fast, but some cameras produce libjpeg warnings)
#   YUYV — raw planar YUV (no decode warnings, slightly more CPU for JPEG re-encode)
#   AUTO — do not force fourcc, let the driver negotiate
CAMERA_FOURCC             = os.environ.get("CAMERA_FOURCC", "MJPG").upper()
CAMERA_DEBUG              = os.environ.get("CAMERA_DEBUG", "false").lower() == "true"
WEBCAM_DETECTION_INTERVAL = float(os.environ.get("WEBCAM_DETECTION_INTERVAL", "5.0"))
WEBCAM_SAVE_FRAMES        = os.environ.get("WEBCAM_SAVE_FRAMES", "false").lower() == "true"
# Inference image size for camera recognition (width=height).
# Smaller values reduce CPU cost: 640 (full, default), 416, 320.
# Upload inference always uses the package's own image size.
CAMERA_INFERENCE_IMGSZ    = int(os.environ.get("CAMERA_INFERENCE_IMGSZ", "640"))
CAMERA_INFERENCE_IMGSZ_EXPLICIT = _env_is_set("CAMERA_INFERENCE_IMGSZ")

# ── Camera runtime recovery ──────────────────────────────────────────────────
# A USB camera can stop delivering frames while the device node stays open —
# the capture thread lives on but nothing ever arrives.  After this many
# consecutive cap.read() failures the capture is released and reopened with a
# bounded exponential backoff (interruptible by stop()).
CAMERA_READ_FAIL_THRESHOLD   = int(os.environ.get("CAMERA_READ_FAIL_THRESHOLD", "60"))
CAMERA_REOPEN_BACKOFF_SEC    = float(os.environ.get("CAMERA_REOPEN_BACKOFF_SEC", "1.0"))
CAMERA_REOPEN_MAX_BACKOFF_SEC = float(os.environ.get("CAMERA_REOPEN_MAX_BACKOFF_SEC", "15.0"))

# Web server
BACKEND_HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "5000"))

# Safety gate for the 安全關機 web button.
# Set ENABLE_SYSTEM_SHUTDOWN=true AND configure sudoers (see docs/DEPLOY_JETSON.md)
# before enabling.  When false the route returns an error without touching hardware.
ENABLE_SYSTEM_SHUTDOWN = os.environ.get("ENABLE_SYSTEM_SHUTDOWN", "false").lower() == "true"

# Skip the background model load / scale warmup threads.  Set by the test suite
# so importing app.web never touches hardware or a real model.
DISABLE_BOOTSTRAP = os.environ.get("INSTRUMENT_DISABLE_BOOTSTRAP", "false").lower() == "true"

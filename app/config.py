import os

# Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
INPUT_PATH = os.environ.get("INPUT_PATH", "input/test_image.jpg")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# Detection
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

# Detector backend — "pt" (default) or "onnx"
# ONNX is ~3x faster on CPU (onnxruntime CPUExecutionProvider).
# Set STRICT_DETECTOR_BACKEND=true to crash instead of falling back to .pt.
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
SERIAL_TIMEOUT      = float(os.environ.get("SERIAL_TIMEOUT", "2.0"))
SERIAL_READ_RETRIES = int(os.environ.get("SERIAL_READ_RETRIES", "5"))

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
WEBCAM_DETECTION_INTERVAL = float(os.environ.get("WEBCAM_DETECTION_INTERVAL", "5.0"))
WEBCAM_SAVE_FRAMES        = os.environ.get("WEBCAM_SAVE_FRAMES", "false").lower() == "true"
CAMERA_PREVIEW_FPS        = int(os.environ.get("CAMERA_PREVIEW_FPS", "10"))

# Web server
BACKEND_HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "5000"))

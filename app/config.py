import os

# Paths
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
INPUT_PATH = os.environ.get("INPUT_PATH", "input/test_image.jpg")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

# Detection
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

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

# Webcam settings
# WEBCAM_INDEX=0 for /dev/video0 (USB webcam).
# For Jetson CSI camera use a GStreamer string instead of an integer index.
WEBCAM_INDEX              = int(os.environ.get("WEBCAM_INDEX", "0"))
WEBCAM_WIDTH              = int(os.environ.get("WEBCAM_WIDTH", "1280"))
WEBCAM_HEIGHT             = int(os.environ.get("WEBCAM_HEIGHT", "720"))
WEBCAM_DETECTION_INTERVAL = float(os.environ.get("WEBCAM_DETECTION_INTERVAL", "1.0"))
WEBCAM_SAVE_FRAMES        = os.environ.get("WEBCAM_SAVE_FRAMES", "false").lower() == "true"

# Web server
BACKEND_HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "5000"))

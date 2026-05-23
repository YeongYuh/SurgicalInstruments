from __future__ import annotations

import atexit
import json
import threading
from pathlib import Path

from flask import Flask

import app.config as config
from app.detector import SurgicalInstrumentDetector
from app.scale_reader import create_scale_reader

# ── Shared singletons ────────────────────────────────────────────────────────
detector = SurgicalInstrumentDetector(
    model_path=config.MODEL_PATH,
    backend=config.DETECTOR_BACKEND,
    onnx_path=config.ONNX_MODEL_PATH,
    onnx_task=config.ONNX_TASK,
)

scale_reader = create_scale_reader(
    mode=config.SCALE_READER_MODE,
    mock_weight=config.MOCK_WEIGHT,
    port=config.SERIAL_PORT,
    baudrate=config.SERIAL_BAUDRATE,
    timeout=config.SERIAL_TIMEOUT,
    retries=config.SERIAL_READ_RETRIES,
    zero_threshold=config.SCALE_ZERO_THRESHOLD_GRAMS,
    zero_confirm_samples=config.SCALE_ZERO_CONFIRM_SAMPLES,
    filter_window=config.SCALE_FILTER_WINDOW,
    transition_threshold=config.SCALE_TRANSITION_THRESHOLD_GRAMS,
    debug=config.SCALE_DEBUG,
)
atexit.register(scale_reader.close)

# For serial mode: open the port now in a background thread so the Arduino
# reset-on-DTR initialization (~3 s) completes during app startup rather than
# blocking the user's first upload request.
if config.SCALE_READER_MODE == "serial":
    def _warmup_scale() -> None:
        scale_reader.read_weight()  # triggers lazy connect + drains startup buffer

    threading.Thread(target=_warmup_scale, name="scale-warmup", daemon=True).start()

# Load the detector model (PT or ONNX) in the background at startup so the
# first user upload is not stalled by a cold-load delay (~1-10 s).
import logging as _logging
import numpy as _np
_det_logger = _logging.getLogger(__name__)

def _warmup_detector() -> None:
    import time as _time
    t0 = _time.perf_counter()
    dummy = _np.zeros((64, 64, 3), dtype=_np.uint8)
    try:
        detector.predict(dummy, conf=0.25)
        ms = (_time.perf_counter() - t0) * 1000
        _det_logger.info("[Detector] Warmup done — backend=%s  %.0f ms", detector._effective_backend, ms)
    except Exception as exc:
        _det_logger.warning("[Detector] Warmup failed: %s", exc)

threading.Thread(target=_warmup_detector, name="detector-warmup", daemon=True).start()

# Per-class expected counts — loaded from disk, editable at runtime
_STANDARDS_FILE = Path(config.OUTPUT_DIR) / "standards.json"
standards: dict[str, int] = {}
if _STANDARDS_FILE.exists():
    try:
        standards = json.loads(_STANDARDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        standards = {}

# Per-class unit weight (g/unit) — loaded from disk, editable at runtime
_UNIT_WEIGHTS_FILE = Path(config.OUTPUT_DIR) / "unit_weights.json"
unit_weights: dict[str, float] = {}
if _UNIT_WEIGHTS_FILE.exists():
    try:
        unit_weights = json.loads(_UNIT_WEIGHTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        unit_weights = {}

# Per-class instrument weight from model definition — read-only source of truth
# for standard weight calculation (标准重量).
# Loaded once at startup from CLASS_WEIGHT_PATH (default: models/class_weight.json).
_CLASS_WEIGHT_PATH = Path(config.CLASS_WEIGHT_PATH)
_cw_logger = _logging.getLogger(__name__)
class_weights: dict[str, float] = {}
if _CLASS_WEIGHT_PATH.exists():
    try:
        _raw_cw = json.loads(_CLASS_WEIGHT_PATH.read_text(encoding="utf-8"))
        class_weights = {str(k): float(v) for k, v in _raw_cw.items()}
        _cw_logger.info(
            "[class_weights] loaded %d classes from %s",
            len(class_weights), _CLASS_WEIGHT_PATH,
        )
    except Exception as _exc:
        _cw_logger.warning(
            "[class_weights] failed to load %s: %s — standard weight will be 0",
            _CLASS_WEIGHT_PATH, _exc,
        )
else:
    _cw_logger.warning(
        "[class_weights] %s not found — standard weight will be 0", _CLASS_WEIGHT_PATH,
    )

state_lock = threading.Lock()

# Serialises camera start and stop operations so that a new open can never race
# with an in-progress cap.release().  Both /camera/start and /camera/stop acquire
# this lock for the duration of their device-level work (but NOT during
# wait_until_ready, so the UI is not frozen while the camera warms up).
_camera_lifecycle_lock = threading.Lock()

# Camera thread — set by routes.py after start
camera_thread = None  # type: ignore[assignment]
camera_session_id: int = 0  # incremented on each successful start

# Latest inference result shared between camera thread and /status route
latest_state: dict = {
    "timestamp": "",
    "counts": {},
    "weight": None,
    "annotated_b64": None,
    "weight_verification": None,
}


# ── Shared utility ───────────────────────────────────────────────────────────

def compute_weight_verification(
    standards: dict,
    unit_weights: dict,
    actual_weight: float | None,
    tolerance: float,
) -> dict:
    """Calculate expected weight from standards × unit weights and compare with actual scale reading.

    Callers should pass class_weights (from class_weight.json) as unit_weights so
    that standard weight is derived from the model definition rather than the
    user-editable output/unit_weights.json file.

    Classes present in standards but absent from unit_weights are logged as
    warnings and contribute 0 g (no crash).
    """
    _wv_log = _logging.getLogger(__name__)
    expected = 0.0
    for cls, std_qty in standards.items():
        qty = int(std_qty or 0)
        if qty == 0:
            continue
        if cls not in unit_weights:
            _wv_log.warning(
                "[weight_verification] class '%s' (std=%d) not in class_weight.json — 0 g",
                cls, qty,
            )
            continue
        expected += qty * float(unit_weights[cls])
    if actual_weight is None:
        return {
            "passed": False,
            "expected": round(expected, 4),
            "actual": None,
            "difference": None,
            "tolerance": tolerance,
            "message": "無法讀取重量",
        }
    diff = abs(actual_weight - expected)
    passed = diff <= tolerance
    return {
        "passed": passed,
        "expected": round(expected, 4),
        "actual": round(actual_weight, 4),
        "difference": round(diff, 4),
        "tolerance": tolerance,
        "message": "重量在容許範圍內" if passed else "重量超出容許範圍",
    }


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # re-read templates from disk on every request

from app.web import routes  # noqa: E402, F401  (registers blueprints)

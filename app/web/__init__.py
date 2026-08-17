"""Shared singletons for the Flask platform.

The platform owns hardware and workflow: camera, scale, weight verification,
inventory comparison, history, reports.  It does NOT own a model — that comes
from whichever Model Package is active, through ModelManager.

Nothing in this package imports ultralytics, torch, or onnxruntime.
"""

from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask

import app.config as config
from app.inference import ModelManager, ModelManagerError, PackageMismatchError
from app.scale_reader import create_scale_reader
from app.weight_verification import compute_weight_verification  # re-exported

logger = logging.getLogger(__name__)

# ── Model packages ───────────────────────────────────────────────────────────
_OUTPUT_DIR = Path(config.OUTPUT_DIR)

model_manager = ModelManager(
    packages_dir=Path(config.MODEL_PACKAGES_DIR),
    profiles_dir=Path(config.PROFILES_DIR),
    project_root=config.PROJECT_ROOT,
    legacy_package_id=config.LEGACY_PACKAGE_ID,
    legacy_standards_path=_OUTPUT_DIR / "standards.json",
    legacy_unit_weights_path=_OUTPUT_DIR / "unit_weights.json",
)
model_manager.discover(force=True)

if not config.DISABLE_BOOTSTRAP:
    # Publishes the manifest + profile synchronously (JSON only, so /standards
    # answers immediately) and loads the model in the background, exactly like
    # the previous detector warmup thread.
    if model_manager.bootstrap(
        config.ACTIVE_MODEL_PACKAGE,
        startup_preset=config.STARTUP_INVENTORY_PRESET,
    ) is None:
        logger.error(
            "[startup] no active model package — set ACTIVE_MODEL_PACKAGE to one of: %s",
            ", ".join(sorted(model_manager.discover(force=False))) or "(none discovered)",
        )

# ── Scale ────────────────────────────────────────────────────────────────────
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
    stable_window_sec=config.SCALE_STABLE_WINDOW_SEC,
    stable_range_grams=config.SCALE_STABLE_RANGE_GRAMS,
    stable_min_samples=config.SCALE_STABLE_MIN_SAMPLES,
    max_sample_age_sec=config.SCALE_MAX_SAMPLE_AGE_SEC,
    stable_min_coverage_ratio=config.SCALE_STABLE_MIN_COVERAGE_RATIO,
    reconnect_backoff_initial=config.SCALE_RECONNECT_BACKOFF_INITIAL,
    reconnect_backoff_max=config.SCALE_RECONNECT_BACKOFF_MAX,
    reconnect_log_interval_sec=config.SCALE_RECONNECT_LOG_INTERVAL_SEC,
    max_plausible_grams=config.SCALE_MAX_PLAUSIBLE_GRAMS,
)
atexit.register(scale_reader.close)

if config.SCALE_READER_MODE == "serial" and not config.DISABLE_BOOTSTRAP:
    # Open the port now so the Arduino reset-on-DTR delay (~3 s) is paid during
    # startup rather than on the operator's first upload.
    def _warmup_scale() -> None:
        scale_reader.read_weight()

    threading.Thread(target=_warmup_scale, name="scale-warmup", daemon=True).start()

# ── Shared runtime state ─────────────────────────────────────────────────────
state_lock = threading.Lock()

# Serialises camera start and stop so a new open can never race an in-progress
# cap.release().  Both /camera/start and /camera/stop hold it for their
# device-level work, but NOT during wait_until_ready, so the UI is not frozen
# while the camera warms up.
_camera_lifecycle_lock = threading.Lock()

camera_thread = None  # type: ignore[assignment]
camera_session_id: int = 0

latest_state: Dict[str, Any] = {
    "timestamp": "",
    "counts": {},
    "weight": None,
    "annotated_b64": None,
    "weight_verification": None,
    "package_id": None,
    "package_display_name": None,
    "model_generation": 0,
}


def reset_latest_state() -> None:
    """Clear inference state left over from another model package.

    Called on every package switch: counts produced by the orthopaedic model
    are meaningless under obstetric standards, so they must not survive.
    """
    with state_lock:
        latest_state.update({
            "timestamp": "",
            "counts": {},
            "weight": None,
            "annotated_b64": None,
            "weight_verification": None,
            "package_id": None,
            "package_display_name": None,
            "model_generation": model_manager.generation,
        })


def _on_package_switched(_state) -> None:
    """Clear application state INSIDE the switch transaction.

    Registered with ModelManager so it runs while the lifecycle gate is still
    held.  Doing it afterwards, from the HTTP route, leaves a window in which
    the manager already reports package B while latest_state and the camera
    still hold package A's counts — and any request landing in that window
    would combine them.
    """
    with state_lock:
        latest_state.update({
            "timestamp": "",
            "counts": {},
            "weight": None,
            "annotated_b64": None,
            "weight_verification": None,
            "package_id": None,
            "package_display_name": None,
            "model_generation": model_manager.generation,
        })
    cam = camera_thread
    if cam is not None:
        cam.invalidate_results()


model_manager.add_switch_listener(_on_package_switched)


# ── Package-scoped inventory accessors ───────────────────────────────────────
# Always fetch through these.  Holding on to a dict returned by the profile is
# fine (it is a copy); holding a reference to the profile itself across a
# package switch is not, which is why no caller is given one.

def active_profile():
    return model_manager.active_profile


def get_standards() -> Dict[str, int]:
    profile = model_manager.active_profile
    return profile.standards if profile is not None else {}


def get_unit_weights() -> Dict[str, float]:
    profile = model_manager.active_profile
    return profile.unit_weights if profile is not None else {}


def get_class_weights() -> Dict[str, float]:
    profile = model_manager.active_profile
    return profile.class_weights if profile is not None else {}


def update_standards(values: Dict[str, Any],
                     package_id: Optional[str] = None) -> Dict[str, int]:
    """Edit standards, refusing the write if the caller meant another package."""
    return model_manager.update_active_profile("standards", values, package_id=package_id)


def update_unit_weights(values: Dict[str, Any],
                        package_id: Optional[str] = None) -> Dict[str, float]:
    return model_manager.update_active_profile("unit_weights", values, package_id=package_id)


# NOTE: there is deliberately no module-level register_classes().  Registering a
# model's output classes must target the package that produced them, which is
# only knowable from the inference session that ran — asking "which profile is
# active now?" after inference is exactly how one package's classes end up in
# another's profile.  Use InferenceSession.register_classes().


def package_identity() -> Dict[str, Optional[str]]:
    state = model_manager.state()
    return {
        "package_id": state.package_id or None,
        "package_display_name": state.display_name or None,
    }


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # re-read templates from disk on every request

from app.web import routes  # noqa: E402, F401  (registers routes)

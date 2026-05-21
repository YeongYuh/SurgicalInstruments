from __future__ import annotations

import base64
import copy
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import cv2
import numpy as np
from flask import Response, jsonify, render_template, request, stream_with_context

import app.config as config
import app.web as web_pkg
from app.visualizer import draw_detections
from app.web import history as hist
from app.web import report as rpt
from app.web.camera import CameraThread
from app.web import compute_weight_verification

_STANDARDS_FILE = Path(config.OUTPUT_DIR) / "standards.json"
_UNIT_WEIGHTS_FILE = Path(config.OUTPUT_DIR) / "unit_weights.json"


# ── Persist helpers ──────────────────────────────────────────────────────────

def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ── Auto-discovery: add new classes to both dicts ────────────────────────────

def _register_new_classes(counts: dict) -> None:
    changed_std = changed_uw = False
    with web_pkg.state_lock:
        for name in counts:
            if name not in web_pkg.standards:
                web_pkg.standards[name] = 0
                changed_std = True
            if name not in web_pkg.unit_weights:
                web_pkg.unit_weights[name] = 0.0
                changed_uw = True
        if changed_std:
            _save_json(_STANDARDS_FILE, web_pkg.standards)
        if changed_uw:
            _save_json(_UNIT_WEIGHTS_FILE, web_pkg.unit_weights)


# ── Core inference helper ────────────────────────────────────────────────────

def _run_image_inference(image_bgr: np.ndarray, source_label: str) -> dict:
    _, detections = web_pkg.detector.predict(image_bgr, conf=config.CONF_THRESHOLD)
    counts = web_pkg.detector.count_instruments(detections)
    weight = web_pkg.scale_reader.read_weight()

    annotated = draw_detections(image_bgr, detections)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with web_pkg.state_lock:
        std_snap = dict(web_pkg.standards)
        uw_snap = dict(web_pkg.unit_weights)

    wv = compute_weight_verification(std_snap, uw_snap, weight, config.WEIGHT_TOLERANCE)

    with web_pkg.state_lock:
        web_pkg.latest_state.update({
            "timestamp": ts,
            "counts": copy.deepcopy(counts),
            "weight": weight,
            "annotated_b64": b64,
            "weight_verification": wv,
        })

    _register_new_classes(counts)

    record = hist.make_record(
        source=source_label,
        counts=counts,
        weight=weight,
        standards_snapshot=std_snap,
    )
    hist.append_record(record)

    return {
        "ok": True,
        "timestamp": ts,
        "counts": counts,
        "annotated_image": b64,
        "weight": weight,
        "weight_verification": wv,
    }


# ── Routes ───────────────────────────────────────────────────────────────────

app = web_pkg.app


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if f is None:
        return jsonify(ok=False, error="No image file provided"), 400
    data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(ok=False, error="Cannot decode image"), 400
    return jsonify(_run_image_inference(image, "upload"))


@app.route("/recognize", methods=["POST"])
def recognize():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is None or not cam.is_running():
        return jsonify(ok=False, error="Camera is not running"), 400
    frame_bytes = cam.get_mjpeg_frame()
    if not frame_bytes:
        return jsonify(ok=False, error="No frame available yet"), 400
    data = np.frombuffer(frame_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify(ok=False, error="Cannot decode camera frame"), 400
    return jsonify(_run_image_inference(image, "webcam_snapshot"))


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with web_pkg.state_lock:
                cam = web_pkg.camera_thread
            if cam is None or not cam.is_running():
                break
            frame = cam.get_mjpeg_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/camera/frame")
def camera_frame():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is None or not cam.is_running():
        return Response(status=503)
    frame = cam.get_mjpeg_frame()
    if not frame or len(frame) < 500:   # guard against empty / malformed buffer
        return Response(status=204)
    return Response(
        frame,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.route("/camera/status")
def camera_status():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is None:
        return jsonify(running=False, source=None, last_detection_ts=None,
                       inference_running=False, error=None)
    return jsonify(**cam.get_status())


@app.route("/camera/result")
def camera_result():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is None:
        return jsonify(ok=False, error="Camera not started")
    result = cam.get_result()
    result["ok"] = True
    with web_pkg.state_lock:
        std = dict(web_pkg.standards)
        uw = dict(web_pkg.unit_weights)
    result["weight_verification"] = compute_weight_verification(
        std, uw, result.get("weight"), config.WEIGHT_TOLERANCE
    )
    return jsonify(result)


@app.route("/camera/start", methods=["POST"])
def camera_start():
    with web_pkg.state_lock:
        if web_pkg.camera_thread is not None and web_pkg.camera_thread.is_running():
            return jsonify(ok=True, status="already_running")
        cam = CameraThread(camera_source=config.CAMERA_SOURCE)
        web_pkg.camera_thread = cam
    cam.start()
    # Wait briefly so a camera-open failure is detectable before returning
    time.sleep(0.6)
    if not cam.is_running():
        err = cam.get_error() or "Camera failed to open"
        with web_pkg.state_lock:
            web_pkg.camera_thread = None
        return jsonify(ok=False, error=err), 500
    return jsonify(ok=True, status="streaming")


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
        web_pkg.camera_thread = None  # immediately makes /camera/frame return 503
    if cam is not None:
        cam.stop()
        cam.join(timeout=2.0)         # wait for cap.release() to complete
        if cam.is_alive():
            logger.warning("[camera_stop] CameraThread did not stop within 2 s")
    return jsonify(ok=True, status="stopped")


@app.route("/api/weight")
def api_weight():
    """
    Live scale weight endpoint — polled by the frontend every second.

    Reads the latest value from the shared scale_reader (same singleton used
    by /upload and the camera thread).  The serial port is kept open between
    calls so this is fast (returns the cached _latest value after draining any
    new buffered lines).  Never triggers YOLO inference.

    Returns:
        {ok: true,  weight: 123.4, unit: "g", source: "serial", weight_verification: {...}}
        {ok: false, weight: null,  unit: "g", source: "serial",
         error: "No valid scale reading yet", weight_verification: {...}}
    """
    weight = web_pkg.scale_reader.read_weight()
    logger.debug("[/api/weight] weight=%s source=%s", weight, config.SCALE_READER_MODE)

    with web_pkg.state_lock:
        std = dict(web_pkg.standards)
        uw  = dict(web_pkg.unit_weights)

    wv = compute_weight_verification(std, uw, weight, config.WEIGHT_TOLERANCE)

    resp = {
        "ok":                  weight is not None,
        "weight":              weight,
        "unit":                "g",
        "source":              config.SCALE_READER_MODE,
        "weight_verification": wv,
    }
    if weight is None:
        resp["error"] = "No valid scale reading yet"
    return jsonify(resp)


@app.route("/status")
def status():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
        state = copy.deepcopy(web_pkg.latest_state)
    return jsonify(
        camera_active=(cam is not None and cam.is_running()),
        **state,
    )


@app.route("/history")
def get_history():
    return jsonify(hist.load_history())


@app.route("/report")
def get_report():
    records = hist.load_history()
    with web_pkg.state_lock:
        standards = dict(web_pkg.standards)
    csv_str = rpt.generate_csv(records, standards)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        csv_str.encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{ts}.csv"},
    )


@app.route("/bom_report")
def bom_report():
    with web_pkg.state_lock:
        standards = dict(web_pkg.standards)
        unit_weights = dict(web_pkg.unit_weights)
        state = copy.deepcopy(web_pkg.latest_state)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_str = rpt.generate_bom_csv(
        counts=state.get("counts", {}),
        standards=standards,
        unit_weights=unit_weights,
        actual_weight=state.get("weight"),
        tolerance=config.WEIGHT_TOLERANCE,
        timestamp=ts,
    )
    file_ts = ts.replace(" ", "_").replace(":", "-")
    return Response(
        csv_str.encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bom_{file_ts}.csv"},
    )


@app.route("/standards", methods=["GET"])
def get_standards():
    with web_pkg.state_lock:
        return jsonify(dict(web_pkg.standards))


@app.route("/standards", methods=["POST"])
def post_standards():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    parsed = {}
    for k, v in data.items():
        try:
            parsed[str(k)] = max(0, int(v))
        except (ValueError, TypeError):
            return jsonify(ok=False, error=f"Invalid value for '{k}': {v}"), 400
    with web_pkg.state_lock:
        web_pkg.standards.update(parsed)
        _save_json(_STANDARDS_FILE, web_pkg.standards)
    return jsonify(ok=True)


@app.route("/unit_weights", methods=["GET"])
def get_unit_weights():
    with web_pkg.state_lock:
        return jsonify(dict(web_pkg.unit_weights))


@app.route("/unit_weights", methods=["POST"])
def post_unit_weights():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    parsed = {}
    for k, v in data.items():
        try:
            parsed[str(k)] = max(0.0, float(v))
        except (ValueError, TypeError):
            return jsonify(ok=False, error=f"Invalid value for '{k}': {v}"), 400
    with web_pkg.state_lock:
        web_pkg.unit_weights.update(parsed)
        _save_json(_UNIT_WEIGHTS_FILE, web_pkg.unit_weights)
    return jsonify(ok=True)

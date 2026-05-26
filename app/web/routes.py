from __future__ import annotations

import base64
import copy
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Background scale poll thread ──────────────────────────────────────────────
# Drains the serial buffer at SCALE_BG_POLL_INTERVAL Hz during recognition so
# that /api/weight can return get_latest_weight() (non-blocking) instead of
# calling read_weight() (which can block up to SERIAL_TIMEOUT seconds).
_scale_bg_stop   = threading.Event()
_scale_bg_thread: Optional[threading.Thread] = None
_scale_bg_lock   = threading.Lock()


def _scale_bg_poll_loop(stop: threading.Event) -> None:
    """Continuously drain the serial scale buffer while recognition is active."""
    while not stop.is_set():
        try:
            web_pkg.scale_reader.read_weight()
        except Exception as exc:
            logger.debug("[scale-bg] read_weight error: %s", exc)
        # wait returns early (True) when stop is set, saving up to one interval on shutdown
        stop.wait(timeout=config.SCALE_BG_POLL_INTERVAL)


def _start_scale_bg_poll() -> None:
    """Start the background serial drain thread (idempotent)."""
    global _scale_bg_thread
    with _scale_bg_lock:
        if _scale_bg_thread is not None and _scale_bg_thread.is_alive():
            return
        _scale_bg_stop.clear()
        _scale_bg_thread = threading.Thread(
            target=_scale_bg_poll_loop,
            args=(_scale_bg_stop,),
            name="scale-bg-poll",
            daemon=True,
        )
        _scale_bg_thread.start()
    logger.debug("[scale] bg poll started interval=%.0f ms",
                 config.SCALE_BG_POLL_INTERVAL * 1000)


def _stop_scale_bg_poll() -> None:
    """Signal the background drain thread to stop and wait for it."""
    global _scale_bg_thread
    with _scale_bg_lock:
        _scale_bg_stop.set()
        t = _scale_bg_thread
        _scale_bg_thread = None
    if t is not None and t.is_alive():
        t.join(timeout=1.0)   # 1 s >> SERIAL_TIMEOUT (0.1 s) — should return quickly
    logger.debug("[scale] bg poll stopped")

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

    wv = compute_weight_verification(std_snap, web_pkg.class_weights, weight, config.WEIGHT_TOLERANCE)

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


_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma":        "no-cache",
    "Expires":       "0",
}


@app.route("/camera/frame")
def camera_frame():
    req_session = request.args.get("session", type=int)
    with web_pkg.state_lock:
        cam        = web_pkg.camera_thread
        active_sid = web_pkg.camera_session_id

    # Reject stale preview loops — any request whose session doesn't match the
    # current active session gets 204 so the old frontend loop exits cleanly.
    if req_session is not None and req_session != active_sid:
        if config.CAMERA_DEBUG:
            logger.debug(
                "[/camera/frame] stale session req=%s active=%s → 204",
                req_session, active_sid,
            )
        return Response(status=204, headers=_NO_CACHE)

    # 204 for both "not running" and "running but no frame yet" — frontend
    # keeps the last good image on 204 rather than going blank.
    if cam is None or not cam.is_running():
        return Response(status=204, headers=_NO_CACHE)
    frame, seq = cam.get_mjpeg_frame_and_seq()
    if not frame or len(frame) < 500:
        return Response(status=204, headers=_NO_CACHE)
    headers = {**_NO_CACHE, "X-Frame-Seq": str(seq)}
    return Response(frame, mimetype="image/jpeg", headers=headers)


@app.route("/camera/stream")
def camera_stream():
    req_session = request.args.get("session", type=int)
    # overlay=1 reserved for future detection-box overlay without re-running YOLO

    with web_pkg.state_lock:
        cam        = web_pkg.camera_thread
        active_sid = web_pkg.camera_session_id

    if req_session is not None and req_session != active_sid:
        return Response(status=204, headers=_NO_CACHE)
    if cam is None or not cam.is_running():
        return Response(status=204, headers=_NO_CACHE)

    def generate():
        last_seq = -1
        frames_yielded = 0
        while True:
            with web_pkg.state_lock:
                cur_cam = web_pkg.camera_thread
                cur_sid = web_pkg.camera_session_id
            if cur_cam is None or not cur_cam.is_running():
                break
            if req_session is not None and cur_sid != req_session:
                if config.CAMERA_DEBUG:
                    logger.debug(
                        "[/camera/stream] session expired req=%s cur=%s after %d frames",
                        req_session, cur_sid, frames_yielded,
                    )
                break
            frame, seq = cur_cam.get_mjpeg_frame_and_seq()
            if frame and len(frame) >= 500 and seq != last_seq:
                last_seq = seq
                frames_yielded += 1
                if config.CAMERA_DEBUG and frames_yielded % 30 == 1:
                    logger.debug(
                        "[/camera/stream] session=%s frame=%d seq=%d size=%dB",
                        req_session, frames_yielded, seq, len(frame),
                    )
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n" + frame + b"\r\n"
                )
            else:
                time.sleep(0.04)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=_NO_CACHE,
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
    result["weight_verification"] = compute_weight_verification(
        std, web_pkg.class_weights, result.get("weight"), config.WEIGHT_TOLERANCE
    )
    return jsonify(result)


@app.route("/camera/start", methods=["POST"])
def camera_start():
    # Acquire the lifecycle lock for the device-setup phase only (state check,
    # thread creation, cam.start()).  This ensures we never open /dev/video0
    # while a concurrent /camera/stop is still in its join (cap.release pending).
    # The lock is released before wait_until_ready so a /camera/stop arriving
    # during warmup is not blocked for the full 8 s timeout.
    with web_pkg._camera_lifecycle_lock:
        with web_pkg.state_lock:
            existing = web_pkg.camera_thread
            if existing is not None and existing.is_running():
                return jsonify(ok=True, status="already_running",
                               session_id=web_pkg.camera_session_id)
            if existing is not None and not existing.is_running():
                # Stale reference: thread exited without going through /camera/stop
                # (e.g. open failure that set _ready_event then returned, or stop() was
                # called externally).  Clear it before creating the new session.
                if config.CAMERA_DEBUG:
                    logger.debug(
                        "[camera_start] clearing stale session=%s (not running)",
                        existing.session_id,
                    )
                web_pkg.camera_thread = None
            web_pkg.camera_session_id += 1
            session_id = web_pkg.camera_session_id
            # Clear any annotated frame from the previous session so a refresh after
            # stop never re-displays the old camera image.
            web_pkg.latest_state["annotated_b64"] = None
            cam = CameraThread(camera_source=config.CAMERA_SOURCE, session_id=session_id)
            web_pkg.camera_thread = cam
        cam.start()
        if config.CAMERA_DEBUG:
            logger.debug(
                "[camera_start] session=%s thread launched — waiting for first frame",
                session_id,
            )
    # Lifecycle lock released — /camera/stop can now acquire it if needed.
    # Block until the camera thread has produced its first valid frame, or until
    # it exits (camera open error), or until 8 s have elapsed (hung device).
    event_fired = cam.wait_until_ready(timeout=8.0)
    if not event_fired:
        err = "Camera timed out waiting for first frame (device hung?)"
        cam.stop()
        with web_pkg.state_lock:
            web_pkg.camera_thread = None
        return jsonify(ok=False, error=err), 500
    if not cam.is_running():
        err = cam.get_error() or "Camera failed to open"
        with web_pkg.state_lock:
            web_pkg.camera_thread = None
        return jsonify(ok=False, error=err), 500
    if config.CAMERA_DEBUG:
        logger.debug("[camera_start] session=%s ready  ok=True", session_id)
    return jsonify(ok=True, status="streaming", session_id=session_id)


@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    # Stop the background scale poll thread first — it has no dependency on the
    # camera and stopping it here covers the case where the user closes the camera
    # while recognition (and therefore the bg thread) is still active.
    _stop_scale_bg_poll()
    # Hold the lifecycle lock for the entire stop-and-join sequence so that a
    # concurrent /camera/start cannot open /dev/video0 before cap.release() has
    # been called.  state_lock is acquired briefly inside, then released; the
    # camera capture thread's join runs while the lifecycle lock is still held.
    inf_running_at_stop = False
    with web_pkg._camera_lifecycle_lock:
        with web_pkg.state_lock:
            cam = web_pkg.camera_thread
            # Do NOT clear camera_thread here — cam.stop() sets _stopping=True which
            # makes is_running()=False atomically, so all callers that use is_running()
            # (/camera/frame, /camera/stream, /status) already see the camera as
            # inactive.  Clearing camera_thread before join risks an ABA race where
            # a concurrent start replaces the reference and our finally nulls it out.
        if cam is not None:
            if config.CAMERA_DEBUG:
                logger.debug(
                    "[camera_stop] shutdown requested  session=%s  rec=%s  inf=%s",
                    cam.session_id,
                    cam.get_status().get("recognition_running"),
                    cam.get_status().get("inference_running"),
                )
            # stop_recognition() increments generation + clears recognition_running so
            # any in-flight inference worker discards its result immediately.
            cam.stop_recognition()
            # stop() sets _stopping=True + _stop_event → is_running()=False instantly,
            # blocking new inference and making all preview/status routes return "stopped".
            inf_running_at_stop = cam.get_status().get("inference_running", False)
            cam.stop()
            if config.CAMERA_DEBUG:
                logger.debug(
                    "[camera_stop] _stop_event set  inf_running_at_stop=%s  joining…",
                    inf_running_at_stop,
                )
            # Wait for the capture loop to exit and cap.release() to complete.
            # 5 s is far more than the ~100 ms a USB camera cap.read() can block,
            # but covers pathological cases (driver stall, slow USB enumeration).
            # Inference workers are daemon threads and do NOT block this join.
            cam.join(timeout=5.0)
            if cam.is_alive():
                logger.warning(
                    "[camera_stop] CameraThread still alive after 5 s — /dev/video0 "
                    "may not be released. New camera start may fail."
                )
            elif config.CAMERA_DEBUG:
                logger.debug("[camera_stop] CameraThread joined cleanly — /dev/video0 released")
        # Set camera_thread=None AFTER join so cap.release() is guaranteed before any
        # future start can open the device.  Guard with identity check: if a concurrent
        # start already replaced camera_thread, do not clobber the new thread reference.
        with web_pkg.state_lock:
            if web_pkg.camera_thread is cam:
                web_pkg.camera_thread = None
    # Clear camera-sourced annotated frame so a browser refresh after stop
    # never re-displays the last camera image via /status → pollStatus().
    with web_pkg.state_lock:
        web_pkg.latest_state["annotated_b64"] = None
    if config.CAMERA_DEBUG:
        logger.debug("[camera_stop] done  camera_running=False  recognition_running=False")
    return jsonify(
        ok=True,
        status="stopped",
        camera_running=False,
        recognition_running=False,
        inference_running=inf_running_at_stop,
    )


@app.route("/camera/recognition/start", methods=["POST"])
def camera_recognition_start():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    cam_running = cam is not None and cam.is_running()
    if config.CAMERA_DEBUG:
        logger.debug("[/camera/recognition/start] cam=%s running=%s", cam, cam_running)
    if not cam_running:
        return jsonify(ok=False, error="Camera is not running"), 400
    cam.start_recognition()
    _start_scale_bg_poll()   # keep weight cache fresh during recognition
    st = cam.get_status()
    if config.CAMERA_DEBUG:
        logger.debug("[/camera/recognition/start] started: gen=%s rec=%s",
                     st["recognition_generation"], st["recognition_running"])
    return jsonify(ok=True, status="recognition_started",
                   recognition_running=st["recognition_running"],
                   recognition_generation=st["recognition_generation"])


@app.route("/camera/recognition/stop", methods=["POST"])
def camera_recognition_stop():
    _stop_scale_bg_poll()    # no longer need the cache refresh
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is not None:
        cam.stop_recognition()
        st = cam.get_status()
        if config.CAMERA_DEBUG:
            logger.debug(
                "[/camera/recognition/stop] gen=%s rec=%s inf=%s",
                st["recognition_generation"], st["recognition_running"],
                st["inference_running"],
            )
        return jsonify(
            ok=True,
            camera_running=st["running"],
            recognition_running=st["recognition_running"],
            inference_running=st["inference_running"],
            recognition_generation=st["recognition_generation"],
        )
    return jsonify(ok=True, camera_running=False,
                   recognition_running=False, inference_running=False,
                   recognition_generation=0)


@app.route("/api/weight")
def api_weight():
    """
    Live scale weight endpoint — polled by the frontend during recognition.

    During recognition the background scale poll thread (started by
    /camera/recognition/start) continuously drains the serial buffer at
    SCALE_BG_POLL_INTERVAL Hz, keeping _latest fresh.  This endpoint calls
    get_latest_weight() (non-blocking, no serial I/O) so it returns in <1 ms.

    Falls back to read_weight() if no background thread is active (e.g. upload
    mode or first call before recognition starts).

    Returns:
        {ok: true,  weight: 123.4, unit: "g", source: "serial", weight_verification: {...}}
        {ok: false, weight: null,  unit: "g", source: "serial",
         error: "No valid scale reading yet", weight_verification: {...}}
    """
    client = request.args.get("client", "")
    reason = request.args.get("reason", "")
    t0 = time.monotonic()
    # Prefer non-blocking cached read; background thread keeps cache fresh.
    weight = web_pkg.scale_reader.get_latest_weight()
    source = "cache"
    if weight is None:
        weight = web_pkg.scale_reader.read_weight()
        source = "read"
    ms = (time.monotonic() - t0) * 1000
    logger.debug("[/api/weight] weight=%s source=%s(%s) client=%s reason=%s latency=%.1fms",
                 weight, config.SCALE_READER_MODE, source, client, reason, ms)

    with web_pkg.state_lock:
        std = dict(web_pkg.standards)

    wv = compute_weight_verification(std, web_pkg.class_weights, weight, config.WEIGHT_TOLERANCE)

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
    client = request.args.get("client", "")
    with web_pkg.state_lock:
        cam        = web_pkg.camera_thread
        active_sid = web_pkg.camera_session_id
        state      = copy.deepcopy(web_pkg.latest_state)
    # Read recognition flag outside state_lock to avoid lock-order deadlock
    # with CameraThread._run_inference (which holds self._lock then acquires state_lock).
    cam_running = cam is not None and cam.is_running()
    if cam is not None:
        cam_st       = cam.get_status()
        rec_running  = cam_st["recognition_running"]
        inf_running  = cam_st["inference_running"]
        cam_stopping = cam_st.get("stopping", False)
    else:
        rec_running  = False
        inf_running  = False
        cam_stopping = False
    if config.CAMERA_DEBUG:
        logger.debug(
            "[/status] cam=%s stopping=%s rec=%s inf=%s sid=%s client=%s",
            cam_running, cam_stopping, rec_running, inf_running, active_sid, client,
        )
    return jsonify(
        camera_active=cam_running,
        camera_stopping=cam_stopping,
        recognition_running=rec_running,
        session_id=active_sid,
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


@app.route("/class_weights")
def get_class_weights():
    """Return the class_weight.json data (read-only model-defined weights)."""
    return jsonify(web_pkg.class_weights)


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

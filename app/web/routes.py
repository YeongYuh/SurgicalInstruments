from __future__ import annotations

import base64
import copy
import logging
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── Background scale poll thread ──────────────────────────────────────────────
# Drains the serial buffer at SCALE_BG_POLL_INTERVAL Hz so that /api/weight can
# return get_latest_sample() (non-blocking) instead of calling read_weight()
# (which can block up to SERIAL_TIMEOUT seconds).
#
# It runs for the whole lifetime of the process in serial mode, not just during
# recognition.  Freshness and stability are defined over a *time window*: a
# single burst of reads taken at the moment of an upload all carry the same
# timestamp and could never satisfy the window, so a reading could never settle.
# Continuous sampling also keeps the serial buffer drained, so a read never
# returns a value that has been sitting in the kernel buffer for seconds.
_scale_bg_stop   = threading.Event()
_scale_bg_thread: Optional[threading.Thread] = None
_scale_bg_lock   = threading.Lock()


def _scale_bg_poll_loop(stop: threading.Event) -> None:
    """Continuously drain the serial scale buffer."""
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
    if config.SCALE_READER_MODE != "serial":
        return  # mock mode needs no sampling — a constant is always stable
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
    logger.info("[scale] bg poll started interval=%.0f ms",
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
from app.inference import (
    AdapterError,
    ModelBusyError,
    ModelManagerError,
    PackageMismatchError,
    available_adapters,
)
from app.visualizer import draw_detections
from app.web import history as hist
from app.web import report as rpt
from app.web.camera import CameraThread
from app.weight_verification import compute_weight_verification


# ── Scale helpers ────────────────────────────────────────────────────────────

def _current_sample():
    """Latest scale sample, without blocking a web thread on serial I/O.

    The background poll thread keeps the cache fresh.  If it is not running for
    some reason (mock mode has no thread; a startup race), fall back to one
    direct read so the endpoint still answers with real data.
    """
    sample = web_pkg.scale_reader.get_latest_sample()
    if sample.value is None:
        web_pkg.scale_reader.read_weight()
        sample = web_pkg.scale_reader.get_latest_sample()
    return sample


def _inference_overrides():
    """Env overrides for confidence / image size, else the package's own values."""
    conf = config.CONF_THRESHOLD if config.CONF_THRESHOLD_EXPLICIT else None
    imgsz = None
    return conf, imgsz


# ── Core inference helper ────────────────────────────────────────────────────

def _run_image_inference(image_bgr: np.ndarray, source_label: str) -> dict:
    """Single-shot inference on one image, published to the shared state.

    The whole transaction — model call, class registration, shared state, and
    the history record — runs inside one inference session, so the active
    package cannot change between producing the counts and publishing them.
    """
    manager = web_pkg.model_manager
    try:
        with manager.inference_session() as session:
            conf, imgsz = _inference_overrides()
            result = session.infer(image_bgr, conf=conf, imgsz=imgsz)

            counts = result.counts
            sample = _current_sample()
            weight = sample.value

            annotated = draw_detections(image_bgr, result.detections)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode()

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            std_snap = session.standards
            class_weights = session.class_weights
            wv = compute_weight_verification(
                std_snap, class_weights, sample, config.WEIGHT_TOLERANCE)

            with web_pkg.state_lock:
                web_pkg.latest_state.update({
                    "timestamp": ts,
                    "counts": copy.deepcopy(counts),
                    "weight": weight,
                    "annotated_b64": b64,
                    "weight_verification": wv,
                    "package_id": session.package_id,
                    "package_display_name": session.display_name,
                    "model_generation": session.generation,
                })

            # Registers on the profile this inference actually ran against.
            session.register_classes(counts.keys())

            record = hist.make_record(
                source=source_label,
                counts=counts,
                weight=weight,
                standards_snapshot=std_snap,
                package_id=session.package_id,
                package_display_name=session.display_name,
                model_identity=(result.model_info.identity() if result.model_info else None),
                class_weights_snapshot=class_weights,
                weight_verification=wv,
            )
            hist.append_record(record)

            return {
                "ok": True,
                "timestamp": ts,
                "counts": counts,
                "annotated_image": b64,
                "weight": weight,
                "weight_sample": sample.to_dict(),
                "weight_verification": wv,
                "package_id": session.package_id,
                "package_display_name": session.display_name,
                "model_generation": session.generation,
                "inference_ms": round(result.inference_ms, 1),
            }
    except ModelManagerError as exc:
        return {"ok": False, "error": str(exc)}
    except AdapterError as exc:
        return {"ok": False, "error": "模型推理失敗：%s" % exc}


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
    # overlay=1 reserved for future detection-box overlay without re-running inference

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
    model_state = web_pkg.model_manager.state()
    sample = _current_sample()
    result["weight_sample"] = sample.to_dict()
    result["weight_verification"] = compute_weight_verification(
        model_state.standards, model_state.class_weights, sample, config.WEIGHT_TOLERANCE
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
            # It also cancels any camera recovery backoff in progress.
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
    # Wait for the startup load rather than starting recognition against a
    # model that is not there yet — otherwise the failure only surfaces at the
    # first inference, long after the UI said recognition had started.
    manager = web_pkg.model_manager
    if not manager.wait_until_ready(timeout=30.0):
        detail = manager.fatal_error or manager.last_error
        if manager.loading:
            return jsonify(ok=False, error="器械模型載入中，請稍候再試",
                           model_loading=True), 503
        return jsonify(ok=False,
                       error="器械模型無法使用：%s" % (detail or "尚未載入"),
                       model_loading=False), 503
    cam.start_recognition()
    _start_scale_bg_poll()   # idempotent — the thread normally runs already
    st = cam.get_status()
    if config.CAMERA_DEBUG:
        logger.debug("[/camera/recognition/start] started: gen=%s rec=%s",
                     st["recognition_generation"], st["recognition_running"])
    return jsonify(ok=True, status="recognition_started",
                   recognition_running=st["recognition_running"],
                   recognition_generation=st["recognition_generation"])


@app.route("/camera/recognition/stop", methods=["POST"])
def camera_recognition_stop():
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
    """Live scale reading with freshness and stability metadata.

    The background poll thread keeps the sample cache fresh, so this returns in
    well under a millisecond without touching the serial port.

    ``ready_for_verification`` is the gate: while it is false the frontend must
    show a measuring state, NOT a red FAIL — an unsettled scale is an unfinished
    measurement, not a failed inventory.
    """
    client = request.args.get("client", "")
    reason = request.args.get("reason", "")
    t0 = time.monotonic()
    sample = _current_sample()
    ms = (time.monotonic() - t0) * 1000
    logger.debug("[/api/weight] weight=%s stable=%s fresh=%s age=%s client=%s "
                 "reason=%s latency=%.1fms",
                 sample.value, sample.stable, sample.fresh, sample.age_sec,
                 client, reason, ms)

    model_state = web_pkg.model_manager.state()
    wv = compute_weight_verification(
        model_state.standards, model_state.class_weights, sample, config.WEIGHT_TOLERANCE
    )

    resp = {
        "ok":                    sample.value is not None,
        "weight":                sample.value,
        "unit":                  "g",
        "source":                config.SCALE_READER_MODE,
        "stable":                sample.stable,
        "fresh":                 sample.fresh,
        "sample_age":            wv["sample_age"],
        "sample":                sample.to_dict(),
        "ready_for_verification": wv["ready"],
        "weight_verification":   wv,
    }
    if sample.value is None:
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
        cam_recovering = cam_st.get("recovering", False)
    else:
        rec_running  = False
        inf_running  = False
        cam_stopping = False
        cam_recovering = False
    manager = web_pkg.model_manager
    model_state = manager.state()
    if config.CAMERA_DEBUG:
        logger.debug(
            "[/status] cam=%s stopping=%s rec=%s inf=%s sid=%s client=%s pkg=%s",
            cam_running, cam_stopping, rec_running, inf_running, active_sid,
            client, model_state.package_id,
        )
    # Built explicitly rather than splatted: latest_state carries its own
    # model_generation (the one that produced the counts on screen), which is a
    # different thing from the currently active generation.
    payload = dict(state)
    payload.update({
        "camera_active": cam_running,
        "camera_stopping": cam_stopping,
        "camera_recovering": cam_recovering,
        "recognition_running": rec_running,
        "session_id": active_sid,
        "active_package": model_state.package_id or None,
        "active_package_name": model_state.display_name or None,
        # generation of the active package right now
        "model_generation": model_state.generation,
        # generation under which the displayed counts were produced
        "result_model_generation": state.get("model_generation"),
        # ready means the model is actually loaded and can run — a package
        # being selected is not the same thing as a model being usable.
        "model_ready": model_state.ready,
        "model_configured": model_state.configured,
        "model_loading": manager.loading,
        "model_warmup_error": (model_state.adapter.warmup_error
                               if model_state.adapter is not None else None),
        "model_error": manager.fatal_error or manager.last_error,
    })
    return jsonify(payload)


# ── Model package API ────────────────────────────────────────────────────────

@app.route("/api/model-packages")
def api_model_packages():
    """All discovered packages, whether valid, and which one is active."""
    manager = web_pkg.model_manager
    return jsonify(
        ok=True,
        active=manager.state().package_id or None,
        packages=manager.list_packages(),
        adapters=available_adapters(),
    )


@app.route("/api/model-package", methods=["GET"])
def api_model_package_get():
    """Identity and load state of the active package."""
    return jsonify(ok=True, **web_pkg.model_manager.model_info())


def _camera_snapshot():
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is None or not cam.is_running():
        return cam, False, False
    status = cam.get_status()
    return cam, True, bool(status.get("recognition_running"))


def _resume_recognition(cam, was_recognizing: bool) -> bool:
    """Put recognition back the way the operator left it.

    Recognition is the operator's stated intent, not a side effect of the
    switch.  Stopping it to change models and then leaving it stopped means a
    failed switch silently downgraded a working system, and a successful one
    makes the operator press 開始辨識 again for no reason.
    """
    if not was_recognizing or cam is None or not cam.is_running():
        return False
    if not web_pkg.model_manager.state().ready:
        logger.warning("[model-switch] not resuming recognition — model is not ready")
        return False
    cam.start_recognition()
    _start_scale_bg_poll()
    logger.info("[model-switch] recognition resumed")
    return True


@app.route("/api/model-package", methods=["POST"])
def api_model_package_post():
    """Switch the active model package.

    ``ok`` means the new model is loaded, warmed, and compatible with the
    package's expected instruments — activation is fully synchronous, and its
    warmup is a verification gate rather than a best-effort nicety.

    Order matters: recognition is paused first so no worker starts against the
    old package mid-switch; ModelManager.activate() releases the old model
    before loading the new one (never two resident) and rolls the old one back
    on any failure; stale results are then cleared, because counts from the old
    instrument family mean nothing under the new one; finally the operator's
    recognition intent is restored either way.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data.get("id"):
        return jsonify(ok=False, error="Expected JSON object with an 'id' field"), 400
    package_id = str(data["id"])

    manager = web_pkg.model_manager
    cam, camera_running, was_recognizing = _camera_snapshot()

    current = manager.state()
    if current.ready and current.package_id == package_id:
        return jsonify(ok=True, status="already_active",
                       camera_running=camera_running,
                       recognition_was_running=was_recognizing,
                       recognition_resumed=was_recognizing,
                       **manager.model_info())

    if was_recognizing:
        logger.info("[model-switch] pausing recognition for switch to '%s'", package_id)
        cam.stop_recognition()

    try:
        manager.activate(package_id)
    except (ModelManagerError, ModelBusyError) as exc:
        logger.error("[model-switch] failed: %s", exc)
        # activate() has already rolled the previous model back into service.
        resumed = _resume_recognition(cam, was_recognizing)
        state = manager.state()
        return jsonify(
            ok=False,
            error=str(exc),
            active=state.package_id or None,
            model_ready=state.ready,
            fatal_error=manager.fatal_error,
            camera_running=camera_running,
            recognition_was_running=was_recognizing,
            recognition_resumed=resumed,
        ), 400

    # Invalidate everything produced by the previous package.
    web_pkg.reset_latest_state()
    if cam is not None:
        cam.invalidate_results()

    resumed = _resume_recognition(cam, was_recognizing)
    logger.info("[model-switch] now active: '%s' (generation=%d)",
                package_id, manager.generation)
    return jsonify(
        ok=True,
        status="switched",
        camera_running=camera_running,
        recognition_was_running=was_recognizing,
        recognition_stopped=was_recognizing,
        recognition_resumed=resumed,
        **manager.model_info(),
    )


@app.route("/history")
def get_history():
    return jsonify(hist.load_history())


@app.route("/report")
def get_report():
    records = hist.load_history()
    # Only a fallback for pre-package records: every record written since
    # carries the standards that were in force when it was taken.
    csv_str = rpt.generate_csv(records, web_pkg.get_standards())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Response(
        csv_str.encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=report_{ts}.csv"},
    )


@app.route("/bom_report")
def bom_report():
    model_state = web_pkg.model_manager.state()
    with web_pkg.state_lock:
        state = copy.deepcopy(web_pkg.latest_state)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_str = rpt.generate_bom_csv(
        counts=state.get("counts", {}),
        standards=model_state.standards,
        unit_weights=model_state.unit_weights,
        actual_weight=state.get("weight"),
        tolerance=config.WEIGHT_TOLERANCE,
        timestamp=ts,
        package_id=model_state.package_id,
        package_display_name=model_state.display_name,
        weight_verification=state.get("weight_verification"),
    )
    file_ts = ts.replace(" ", "_").replace(":", "-")
    return Response(
        csv_str.encode("utf-8-sig"),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=bom_{file_ts}.csv"},
    )


def _parse_profile_payload(data):
    """Split a profile edit into (target package id, values).

    Accepts the wrapped form ``{"package_id": …, "values": {…}}`` and the
    ``X-Model-Package`` header; a bare ``{class: n}`` body still works for
    older clients, but then carries no package identity and is accepted
    against whatever is active.
    """
    header_pkg = request.headers.get("X-Model-Package") or None
    if isinstance(data, dict) and isinstance(data.get("values"), dict):
        return (data.get("package_id") or header_pkg), data["values"]
    return header_pkg, data


def _mismatch_response(exc: PackageMismatchError):
    """409: the edit belonged to a package that is no longer active.

    Writing it anyway would stamp one department's expected quantities onto
    another's tray, which is precisely what a debounced edit racing a package
    switch would otherwise do.
    """
    return jsonify(
        ok=False,
        error="設定已套用到其他器械套件，請重新編輯",
        detail=str(exc),
        requested_package=exc.requested,
        active_package=exc.active,
    ), 409


@app.route("/standards", methods=["GET"])
def get_standards():
    return jsonify(web_pkg.get_standards())


@app.route("/standards", methods=["POST"])
def post_standards():
    package_id, values = _parse_profile_payload(request.get_json(silent=True))
    if not isinstance(values, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    parsed = {}
    for k, v in values.items():
        try:
            parsed[str(k)] = max(0, int(v))
        except (ValueError, TypeError):
            return jsonify(ok=False, error=f"Invalid value for '{k}': {v}"), 400
    try:
        web_pkg.update_standards(parsed, package_id=package_id)
    except PackageMismatchError as exc:
        return _mismatch_response(exc)
    except ModelManagerError as exc:
        return jsonify(ok=False, error=str(exc)), 409
    return jsonify(ok=True, package_id=web_pkg.model_manager.state().package_id or None)


@app.route("/class_weights")
def get_class_weights():
    """Per-class unit weights defined by the ACTIVE package (read-only)."""
    return jsonify(web_pkg.get_class_weights())


@app.route("/unit_weights", methods=["GET"])
def get_unit_weights():
    return jsonify(web_pkg.get_unit_weights())


@app.route("/unit_weights", methods=["POST"])
def post_unit_weights():
    package_id, values = _parse_profile_payload(request.get_json(silent=True))
    if not isinstance(values, dict):
        return jsonify(ok=False, error="Expected JSON object"), 400
    parsed = {}
    for k, v in values.items():
        try:
            parsed[str(k)] = max(0.0, float(v))
        except (ValueError, TypeError):
            return jsonify(ok=False, error=f"Invalid value for '{k}': {v}"), 400
    try:
        web_pkg.update_unit_weights(parsed, package_id=package_id)
    except PackageMismatchError as exc:
        return _mismatch_response(exc)
    except ModelManagerError as exc:
        return jsonify(ok=False, error=str(exc)), 409
    return jsonify(ok=True, package_id=web_pkg.model_manager.state().package_id or None)


@app.route("/system/shutdown", methods=["POST"])
def system_shutdown():
    """Safe shutdown: stop recognition/camera/scale, then trigger OS poweroff.

    Protected by ENABLE_SYSTEM_SHUTDOWN env var so the button is harmless by
    default.  Requires sudoers entry (see docs/DEPLOY_JETSON.md).
    """
    if not config.ENABLE_SYSTEM_SHUTDOWN:
        return jsonify(
            ok=False,
            error="System shutdown disabled. Set ENABLE_SYSTEM_SHUTDOWN=true to enable.",
        ), 403

    # Stop recognition and camera so /dev/video0 is released before OS halt.
    with web_pkg.state_lock:
        cam = web_pkg.camera_thread
    if cam is not None:
        cam.stop_recognition()
        cam.stop()

    # Stop scale background polling thread.
    _stop_scale_bg_poll()

    # Fire-and-forget: give the HTTP response ~0.5 s to reach the browser, then
    # issue the shutdown command.  subprocess.Popen does not block Flask.
    import subprocess

    def _do_shutdown() -> None:
        import time as _time
        _time.sleep(0.5)
        try:
            subprocess.Popen(["sudo", "shutdown", "-h", "now"])
            logger.info("[shutdown] shutdown command issued")
        except Exception as exc:
            logger.error("[shutdown] Failed to invoke shutdown: %s", exc)

    threading.Thread(target=_do_shutdown, name="shutdown-trigger", daemon=True).start()
    logger.info("[shutdown] shutdown scheduled by web UI")
    return jsonify(ok=True, message="Shutdown scheduled")


# Start sampling immediately so weight readings have a populated time window
# before the operator's first action.
if not config.DISABLE_BOOTSTRAP:
    _start_scale_bg_poll()

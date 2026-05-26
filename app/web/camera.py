from __future__ import annotations

import base64
import copy
import glob
import threading
import time
from datetime import datetime
from typing import Optional, Union

import cv2

import app.config as config
from app.web import compute_weight_verification, history as hist


def _available_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


def _open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    """Try to open the camera; if integer-index fails, retry with /dev/videoN path."""
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap

    # Integer-index failed — MUST release before retrying; the failed cap can hold
    # the V4L2 file descriptor and block the second open attempt.
    cap.release()

    if isinstance(source, int):
        path = f"/dev/video{source}"
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            print(f"[CameraThread] Integer index {source} failed; opened via path {path}")
            return cap
        cap.release()

    return cv2.VideoCapture(source)  # return fresh failed cap so caller sees isOpened()=False


_debug = config.CAMERA_DEBUG


class CameraThread(threading.Thread):
    def __init__(self, camera_source: Union[int, str] = config.CAMERA_SOURCE,
                 session_id: int = 0):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.session_id = session_id
        # Keep legacy attribute for any external code that reads it
        self.webcam_index = camera_source if isinstance(camera_source, int) else 0
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()  # set after first valid frame hits _mjpeg_buffer
        self._stopping: bool = False            # set by stop(); blocks new inference from spawning
        self._lock = threading.Lock()

        self._mjpeg_buffer: bytes = b""
        self._latest_annotated: bytes = b""
        self._counts: dict = {}
        self._weight: float | None = None
        self._timestamp: str = ""
        self._last_detection_ts: str = ""
        self._inference_running: bool = False
        self._recognition_running: bool = False
        self._recognition_generation: int = 0
        self._last_inference_end: float = 0.0  # time.monotonic() when last inference finished
        self._frame_seq: int = 0
        self._error: Optional[str] = None

    # ── public read API (called from Flask routes) ────────────────────────

    def get_mjpeg_frame(self) -> bytes:
        with self._lock:
            return self._mjpeg_buffer

    def get_mjpeg_frame_and_seq(self) -> tuple:
        with self._lock:
            return self._mjpeg_buffer, self._frame_seq

    def get_latest_state(self) -> dict:
        with self._lock:
            return {
                "timestamp": self._timestamp,
                "counts": copy.deepcopy(self._counts),
                "weight": self._weight,
                "annotated_b64": (
                    base64.b64encode(self._latest_annotated).decode()
                    if self._latest_annotated
                    else None
                ),
            }

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self.is_alive() and not self._stopping and not self._stop_event.is_set(),
                "stopping": self._stopping or self._stop_event.is_set(),
                "source": str(self.camera_source),
                "session_id": self.session_id,
                "frame_seq": self._frame_seq,
                "last_detection_ts": self._last_detection_ts,
                "inference_running": self._inference_running,
                "recognition_running": self._recognition_running,
                "recognition_generation": self._recognition_generation,
                "error": self._error,
            }

    def get_result(self) -> dict:
        with self._lock:
            return {
                "timestamp": self._timestamp,
                "counts": copy.deepcopy(self._counts),
                "weight": self._weight,
                "annotated_b64": (
                    base64.b64encode(self._latest_annotated).decode()
                    if self._latest_annotated
                    else None
                ),
            }

    def get_error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            self._recognition_generation += 1  # invalidate any running inference worker
            self._recognition_running = False
        self._stop_event.set()
        if _debug:
            print(f"[CameraThread] session={self.session_id} stop() — _stopping=True "
                  f"_stop_event set")

    def start_recognition(self) -> None:
        with self._lock:
            self._recognition_generation += 1
            self._recognition_running = True
            gen = self._recognition_generation
        if _debug:
            print(f"[CameraThread] session={self.session_id} recognition START gen={gen}")

    def stop_recognition(self) -> None:
        with self._lock:
            self._recognition_generation += 1
            self._recognition_running = False
            gen = self._recognition_generation
        if _debug:
            print(f"[CameraThread] session={self.session_id} recognition STOP gen={gen}")

    def is_running(self) -> bool:
        return self.is_alive() and not self._stopping and not self._stop_event.is_set()

    def wait_until_ready(self, timeout: float = 8.0) -> bool:
        """Block until first frame is in _mjpeg_buffer, the thread exits, or timeout.
        Returns True if the event fired (caller must still check is_running()),
        False on timeout (camera hung without producing a frame)."""
        return self._ready_event.wait(timeout=timeout)

    # ── thread body ───────────────────────────────────────────────────────

    def run(self) -> None:
        # Import here to avoid circular import at module load time
        from app.web import detector, scale_reader, standards, unit_weights, state_lock

        cap = _open_capture(self.camera_source)
        if not cap.isOpened():
            available = _available_video_devices()
            msg = (
                f"Cannot open camera source={self.camera_source!r} — "
                f"available: {available or ['none found']}. "
                f"Try: CAMERA_SOURCE=/dev/video0 ./run_jetson.sh"
            )
            print(f"[CameraThread] {msg}")
            with self._lock:
                self._error = msg
            self._ready_event.set()  # unblock camera_start immediately
            return

        # Set FOURCC before resolution: some V4L2 drivers (Jetson 4.9 kernel) reset
        # the pixel format when width/height is applied, so FOURCC must come first.
        _fourcc = config.CAMERA_FOURCC
        if _fourcc in ("MJPG", "YUYV"):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*_fourcc))
        # AUTO: leave fourcc at whatever the driver negotiates
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.WEBCAM_FPS)
        # Note: CAP_PROP_BUFFERSIZE intentionally NOT set — forcing it to 1 triggers
        # VIDIOC_QBUF errors during cap.release() on the Jetson 4.9 kernel.
        # The driver default buffer count handles teardown correctly.

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cc_int     = int(cap.get(cv2.CAP_PROP_FOURCC))
        # Some V4L2 drivers (Jetson 4.9 kernel) return 0 for CAP_PROP_FOURCC even
        # when the requested format is in use.  This is a driver quirk, not an error.
        if cc_int != 0:
            cc_actual = "".join(chr((cc_int >> i) & 0xFF) for i in (0, 8, 16, 24))
        else:
            cc_actual = f"0 (driver did not report; requested {_fourcc!r} likely applied)"
        print(
            f"[CameraThread] session={self.session_id} started  source={self.camera_source!r}\n"
            f"  requested : {config.WEBCAM_WIDTH}x{config.WEBCAM_HEIGHT}"
            f" @ {config.WEBCAM_FPS} fps  fourcc={_fourcc!r}\n"
            f"  actual    : {actual_w}x{actual_h}"
            f" @ {actual_fps:.0f} fps  fourcc={cc_actual!r}"
        )
        if _debug:
            print(f"[CameraThread] DEBUG session={self.session_id} capture loop entering")

        # Discard initial black frames — USB cameras often send black frames on startup.
        # Deadline prevents an infinite loop if the device opens but never delivers frames.
        warmup = 0
        warmup_deadline = time.monotonic() + 4.0  # 4 s absolute limit
        while warmup < 30 and time.monotonic() < warmup_deadline and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            warmup += 1
            if frame is not None and float(frame.mean()) > 5.0:
                break
        if warmup:
            print(f"[CameraThread] Warmup: {warmup} frame(s) discarded")

        read_ok = 0
        read_fail = 0
        _ready_signaled = False

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    read_fail += 1
                    if read_fail % 30 == 1:
                        print(f"[CameraThread] cap.read() failures: {read_fail} "
                              f"(ok={read_ok})")
                    time.sleep(0.05)
                    continue
                read_ok += 1

                # Encode preview JPEG — only overwrite buffer when encode succeeds
                # so a transient bad frame never replaces a good one with black data.
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok and buf.nbytes > 500:
                    with self._lock:
                        self._mjpeg_buffer = buf.tobytes()
                        self._frame_seq += 1
                        _seq = self._frame_seq
                    if not _ready_signaled:
                        _ready_signaled = True
                        self._ready_event.set()
                        print(f"[CameraThread] session={self.session_id} "
                              f"ready — first frame seq={_seq} size={buf.nbytes:,}B")
                    if _debug and _seq % 30 == 1:
                        print(f"[CameraThread] DEBUG session={self.session_id} "
                              f"seq={_seq} jpeg={buf.nbytes:,}B "
                              f"mean={float(frame.mean()):.1f}")

                # Spawn inference in background — gated on recognition_running, stopping,
                # and cooldown.  Cooldown is measured from inference END (last_inference_end)
                # not start, so a slow inference never causes immediate back-to-back runs.
                now = time.monotonic()
                with self._lock:
                    inf_running = self._inference_running
                    rec_running = self._recognition_running
                    stopping    = self._stopping
                    gen         = self._recognition_generation
                    last_end    = self._last_inference_end
                if (not stopping          # never start inference after shutdown initiated
                        and rec_running
                        and not inf_running
                        and now - last_end >= config.WEBCAM_DETECTION_INTERVAL):
                    with self._lock:
                        self._inference_running = True
                    if _debug:
                        print(f"[CameraThread] session={self.session_id} "
                              f"inference START gen={gen} seq={self._frame_seq}")
                    t = threading.Thread(
                        target=self._run_inference_bg,
                        args=(frame.copy(), gen, detector, scale_reader,
                              standards, unit_weights, state_lock),
                        daemon=True,
                    )
                    t.start()
        finally:
            # Unblock camera_start if the thread exits before producing the first frame
            # (camera open failure, device error, etc.).  camera_start will then see
            # is_running()=False and report the error correctly without waiting 8 s.
            self._ready_event.set()
            # cap.release() may print "ioctl(VIDIOC_QBUF): Bad file descriptor"
            # on the Jetson 4.9.337-tegra kernel — this is a harmless V4L2 driver
            # noise during buffer teardown.  The camera restarts correctly on the
            # next session.  Do NOT call cap.grab() before release — that makes
            # the error worse by queuing into a half-closed device.
            cap.release()
            if _debug:
                print(f"[CameraThread] DEBUG session={self.session_id} released")
            print(f"[CameraThread] session={self.session_id} stopped.")

    def _run_inference_bg(self, frame, generation, detector, scale_reader,
                          standards, unit_weights, state_lock) -> None:
        try:
            self._run_inference(frame, generation, detector, scale_reader,
                                standards, unit_weights, state_lock)
        finally:
            # Stamp end-time BEFORE clearing the flag so the capture loop
            # uses the correct cooldown baseline on the very next frame tick.
            t_end = time.monotonic()
            with self._lock:
                self._inference_running = False
                self._last_inference_end = t_end

    def _run_inference(self, frame, generation, detector, scale_reader,
                       standards, unit_weights, state_lock) -> None:
        t_worker_start = time.perf_counter()
        try:
            backend = getattr(detector, "_effective_backend", "unknown")
            h, w = frame.shape[:2]
            if _debug:
                print(f"[CameraThread] session={self.session_id} "
                      f"inference worker gen={generation} "
                      f"backend={backend} frame={w}x{h} "
                      f"imgsz={config.CAMERA_INFERENCE_IMGSZ}")

            # ── detector.predict() — ONNX RT releases the GIL; preview keeps running ──
            t0 = time.perf_counter()
            _, detections = detector.predict(
                frame,
                conf=config.CONF_THRESHOLD,
                imgsz=config.CAMERA_INFERENCE_IMGSZ,
            )
            t_predict = time.perf_counter() - t0

            t0 = time.perf_counter()
            counts = detector.count_instruments(detections)
            t_count = time.perf_counter() - t0

            # ── scale_reader — non-blocking cached read; serial I/O stays in /api/weight ─
            # read_weight() blocks for up to timeout*retries seconds; get_latest_weight()
            # returns the last value cached by the /api/weight polling thread instantly.
            t0 = time.perf_counter()
            weight = scale_reader.get_latest_weight()
            t_scale = time.perf_counter() - t0

            # ── annotated image is intentionally skipped for camera mode ─────────────
            # The live preview is already the raw camera stream; generating a
            # segmentation-annotated JPEG here wastes CPU and is never shown.

            t0 = time.perf_counter()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Import here (not at module top) to avoid circular import at load time.
            # class_weights is read-only after startup — no lock needed.
            import app.web as web_pkg
            with state_lock:
                std_snap = dict(standards)
            wv = compute_weight_verification(
                std_snap, web_pkg.class_weights, weight, config.WEIGHT_TOLERANCE
            )
            t_prep = time.perf_counter() - t0

            total_ms = (time.perf_counter() - t_worker_start) * 1000

            # Always print the timing summary so slowness is visible without CAMERA_DEBUG.
            weight_str = f"{weight:.1f}g" if weight is not None else "None"
            print(
                f"[CameraThread] session={self.session_id} "
                f"inference gen={generation} "
                f"backend={backend} "
                f"predict={t_predict*1000:.0f}ms  "
                f"scale_cached={t_scale*1000:.1f}ms({weight_str})  "
                f"other={( t_count + t_prep)*1000:.0f}ms  "
                f"total={total_ms:.0f}ms  "
                f"dets={len(detections)}"
            )

            # ── discard if camera is stopping, recognition stopped, or generation advanced ──
            with self._lock:
                stale = (self._stopping
                         or not self._recognition_running
                         or self._recognition_generation != generation)
                if not stale:
                    self._counts = counts
                    self._weight = weight
                    self._timestamp = ts
                    self._last_detection_ts = ts
                    # _latest_annotated intentionally not set in camera mode

            if stale:
                print(f"[CameraThread] session={self.session_id} "
                      f"inference DISCARDED (stale) gen={generation}")
            elif _debug:
                print(f"[CameraThread] session={self.session_id} "
                      f"inference PUBLISHED gen={generation}")

            if stale:
                return

            with state_lock:
                web_pkg.latest_state.update({
                    "timestamp": ts,
                    "counts": copy.deepcopy(counts),
                    "weight": weight,
                    "annotated_b64": None,  # camera mode: live preview is the raw stream
                    "weight_verification": wv,
                })

            with state_lock:
                std_snapshot = dict(standards)
            record = hist.make_record(
                source="webcam",
                counts=counts,
                weight=weight,
                standards_snapshot=std_snapshot,
            )
            hist.append_record(record)

        except Exception as e:
            print(f"[CameraThread] Inference error: {e}")

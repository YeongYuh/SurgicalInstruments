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
from app.visualizer import draw_detections
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


class CameraThread(threading.Thread):
    def __init__(self, camera_source: Union[int, str] = config.CAMERA_SOURCE):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        # Keep legacy attribute for any external code that reads it
        self.webcam_index = camera_source if isinstance(camera_source, int) else 0
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._mjpeg_buffer: bytes = b""
        self._latest_annotated: bytes = b""
        self._counts: dict = {}
        self._weight: float | None = None
        self._timestamp: str = ""
        self._last_detection_ts: str = ""
        self._inference_running: bool = False
        self._error: Optional[str] = None

    # ── public read API (called from Flask routes) ────────────────────────

    def get_mjpeg_frame(self) -> bytes:
        with self._lock:
            return self._mjpeg_buffer

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
                "running": self.is_alive() and not self._stop_event.is_set(),
                "source": str(self.camera_source),
                "last_detection_ts": self._last_detection_ts,
                "inference_running": self._inference_running,
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
        self._stop_event.set()

    def is_running(self) -> bool:
        return self.is_alive() and not self._stop_event.is_set()

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
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        # MJPG fourcc: USB cameras send MJPEG natively — avoids expensive YUYV
        # decode + re-encode and roughly doubles throughput on Jetson.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS)
        # One internal V4L2 buffer keeps the queued-frame count minimal at release.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cc_int     = int(cap.get(cv2.CAP_PROP_FOURCC))
        cc_str     = "".join(chr((cc_int >> i) & 0xFF) for i in (0, 8, 16, 24))
        print(
            f"[CameraThread] Started (source={self.camera_source!r})  "
            f"{actual_w}x{actual_h} @ {actual_fps:.0f} fps  fourcc={cc_str!r}"
        )

        # Discard initial frames — USB cameras often send black frames on startup.
        warmup = 0
        while warmup < 30 and not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            warmup += 1
            if frame is not None and float(frame.mean()) > 5.0:
                break
        if warmup:
            print(f"[CameraThread] Warmup: {warmup} frame(s) discarded")

        last_inference = 0.0
        read_ok = 0
        read_fail = 0

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

                # Spawn inference in background thread — never blocks preview
                now = time.time()
                with self._lock:
                    inf_running = self._inference_running
                if (now - last_inference >= config.WEBCAM_DETECTION_INTERVAL
                        and not inf_running):
                    last_inference = now
                    with self._lock:
                        self._inference_running = True
                    t = threading.Thread(
                        target=self._run_inference_bg,
                        args=(frame.copy(), detector, scale_reader,
                              standards, unit_weights, state_lock),
                        daemon=True,
                    )
                    t.start()
        finally:
            # Do NOT call cap.grab() here — on the Jetson 4.9 kernel it triggers
            # VIDIOC_QBUF on a half-closed device.  Let cap.release() clean up
            # the single buffered frame itself.
            cap.release()
            print("[CameraThread] Stopped.")

    def _run_inference_bg(self, frame, detector, scale_reader,
                          standards, unit_weights, state_lock) -> None:
        try:
            self._run_inference(frame, detector, scale_reader,
                                standards, unit_weights, state_lock)
        finally:
            with self._lock:
                self._inference_running = False

    def _run_inference(self, frame, detector, scale_reader,
                       standards, unit_weights, state_lock) -> None:
        try:
            _, detections = detector.predict(frame, conf=config.CONF_THRESHOLD)
            counts = detector.count_instruments(detections)
            weight = scale_reader.read_weight()

            annotated = draw_detections(frame, detections)
            ok, abuf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with state_lock:
                std_snap = dict(standards)
                uw_snap = dict(unit_weights)

            wv = compute_weight_verification(std_snap, uw_snap, weight, config.WEIGHT_TOLERANCE)

            with self._lock:
                self._counts = counts
                self._weight = weight
                self._timestamp = ts
                self._last_detection_ts = ts
                if ok:
                    self._latest_annotated = abuf.tobytes()

            import app.web as web_pkg
            with state_lock:
                web_pkg.latest_state.update({
                    "timestamp": ts,
                    "counts": copy.deepcopy(counts),
                    "weight": weight,
                    "annotated_b64": (
                        base64.b64encode(abuf.tobytes()).decode() if ok else None
                    ),
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

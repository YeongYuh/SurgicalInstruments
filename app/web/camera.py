from __future__ import annotations

import base64
import copy
import glob
import threading
import time
from datetime import datetime
from typing import Union

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
            print(
                f"[CameraThread] Cannot open camera source={self.camera_source!r}\n"
                f"  Available video devices: {available or ['none found']}\n"
                f"  Try: CAMERA_SOURCE=/dev/video0 ./run_jetson.sh\n"
                f"       CAMERA_SOURCE=/dev/video1 ./run_jetson.sh\n"
                f"       CAMERA_SOURCE=1 ./run_jetson.sh"
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        print(f"[CameraThread] Started (source={self.camera_source!r})")

        last_inference = 0.0

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                # Always update MJPEG buffer for smooth video
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    with self._lock:
                        self._mjpeg_buffer = buf.tobytes()

                # Inference on interval
                now = time.time()
                if now - last_inference >= config.WEBCAM_DETECTION_INTERVAL:
                    last_inference = now
                    self._run_inference(frame, detector, scale_reader, standards, unit_weights, state_lock)
        finally:
            cap.release()
            print("[CameraThread] Stopped.")

    def _run_inference(self, frame, detector, scale_reader, standards, unit_weights, state_lock) -> None:
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

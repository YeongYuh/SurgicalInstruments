from __future__ import annotations

import base64
import copy
import threading
import time
from datetime import datetime

import cv2

import app.config as config
from app.visualizer import draw_detections
from app.web import compute_weight_verification, history as hist


class CameraThread(threading.Thread):
    def __init__(self, webcam_index: int = config.WEBCAM_INDEX):
        super().__init__(daemon=True)
        self.webcam_index = webcam_index
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

        cap = cv2.VideoCapture(self.webcam_index)
        if not cap.isOpened():
            print(f"[CameraThread] Cannot open webcam index {self.webcam_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        print(f"[CameraThread] Started (index={self.webcam_index})")

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

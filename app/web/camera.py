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
from app.web import history as hist


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


def _next_backoff(current: float) -> float:
    return min(max(current, 0.1) * 2.0, config.CAMERA_REOPEN_MAX_BACKOFF_SEC)


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
        self._package_id: Optional[str] = None
        self._package_display_name: Optional[str] = None
        self._model_generation: int = 0

        # Runtime read-failure recovery bookkeeping
        self._consecutive_read_failures: int = 0
        self._reopen_count: int = 0
        self._recovering: bool = False

    # ── public read API (called from Flask routes) ────────────────────────

    def get_mjpeg_frame(self) -> bytes:
        with self._lock:
            return self._mjpeg_buffer

    def get_mjpeg_frame_and_seq(self) -> tuple:
        with self._lock:
            return self._mjpeg_buffer, self._frame_seq

    # NOTE: there is deliberately no get_latest_state() variant that omits the
    # package identity.  Every accessor that returns counts must carry
    # package_id and model_generation, or a caller can reach a result without
    # the means to tell whether it still belongs to the active package.
    # get_result() is the one accessor; run it through
    # app.runtime_result.sanitize_runtime_result before serving it.

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
                "model_generation": self._model_generation,
                "package_id": self._package_id,
                "read_failures": self._consecutive_read_failures,
                "reopen_count": self._reopen_count,
                "recovering": self._recovering,
                "error": self._error,
            }

    def get_result(self) -> dict:
        with self._lock:
            return {
                "timestamp": self._timestamp,
                "counts": copy.deepcopy(self._counts),
                "weight": self._weight,
                "package_id": self._package_id,
                "package_display_name": self._package_display_name,
                # Carried so readers can tell whether this result still belongs
                # to the active package: the id alone is not enough, because a
                # reload of the same package produces a new generation.
                "model_generation": self._model_generation,
                "annotated_b64": (
                    base64.b64encode(self._latest_annotated).decode()
                    if self._latest_annotated
                    else None
                ),
            }

    def commit_inference_result(self, generation: int, payload: dict) -> bool:
        """Publish an inference result, or refuse it — one linearization point.

        The staleness check and BOTH writes (camera-local result and the shared
        application state) happen in a single critical section, ordered against
        ``stop_recognition()`` which bumps the generation under the same lock.

        Whichever gets the lock first wins: a result that commits before the
        stop is a valid inventory taken while recognition was running, and one
        that arrives after it is discarded entirely.  Checking staleness and
        then publishing in two separate steps let a late worker write the UI
        and the history *after* the operator had already been told recognition
        had stopped.

        Returns True when the result was published (and may be recorded in
        history), False when it was refused.
        """
        import app.web as web_pkg

        with self._lock:
            if (self._stopping
                    or not self._recognition_running
                    or self._recognition_generation != generation):
                return False

            self._counts = payload["counts"]
            self._weight = payload["weight"]
            self._timestamp = payload["timestamp"]
            self._last_detection_ts = payload["timestamp"]
            self._package_id = payload["package_id"]
            self._package_display_name = payload["package_display_name"]
            self._model_generation = payload["model_generation"]
            # _latest_annotated intentionally not set in camera mode

            with web_pkg.state_lock:
                web_pkg.latest_state.update({
                    "timestamp": payload["timestamp"],
                    "counts": copy.deepcopy(payload["counts"]),
                    "weight": payload["weight"],
                    "annotated_b64": None,  # live preview is the raw stream
                    "weight_verification": payload["weight_verification"],
                    "package_id": payload["package_id"],
                    "package_display_name": payload["package_display_name"],
                    "model_generation": payload["model_generation"],
                })
        return True

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

    def invalidate_results(self) -> None:
        """Drop published results and invalidate in-flight inference workers.

        Called when the active model package changes: counts produced by the
        previous model describe a different instrument set entirely.
        """
        with self._lock:
            self._recognition_generation += 1
            self._counts = {}
            self._weight = None
            self._timestamp = ""
            self._last_detection_ts = ""
            self._latest_annotated = b""
            self._package_id = None
            self._package_display_name = None

    def is_running(self) -> bool:
        return self.is_alive() and not self._stopping and not self._stop_event.is_set()

    def wait_until_ready(self, timeout: float = 8.0) -> bool:
        """Block until first frame is in _mjpeg_buffer, the thread exits, or timeout.
        Returns True if the event fired (caller must still check is_running()),
        False on timeout (camera hung without producing a frame)."""
        return self._ready_event.wait(timeout=timeout)

    # ── capture device setup ──────────────────────────────────────────────

    def _open_configured(self) -> Optional[cv2.VideoCapture]:
        """Open /dev/videoN and apply FOURCC / resolution / FPS.  None on failure."""
        cap = _open_capture(self.camera_source)
        if not cap.isOpened():
            cap.release()
            return None

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
        return cap

    def _log_capture_settings(self, cap: cv2.VideoCapture) -> None:
        _fourcc = config.CAMERA_FOURCC
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

    def _discard_warmup_frames(self, cap: cv2.VideoCapture) -> None:
        """Discard initial black frames — USB cameras often send them on startup.

        The deadline prevents an infinite loop if the device opens but never
        delivers frames.
        """
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

    def _recover_capture(self, cap: Optional[cv2.VideoCapture]) -> Optional[cv2.VideoCapture]:
        """Release a dead capture and reopen it with bounded backoff.

        A USB camera can keep its device node open while silently delivering
        nothing — the thread stays alive but no frame ever arrives again.  This
        releases the handle and retries until it works or stop() is called.

        Returns the new capture, or None if stop() was requested.  Never resets
        the USB bus, never calls sudo, never reboots.
        """
        with self._lock:
            self._recovering = True
            self._reopen_count += 1
            attempt = self._reopen_count
        msg = (f"no frames after {config.CAMERA_READ_FAIL_THRESHOLD} consecutive read "
               f"failures — reopening {self.camera_source!r} (attempt {attempt})")
        print(f"[CameraThread] session={self.session_id} RECOVERY: {msg}")
        with self._lock:
            self._error = f"camera recovery in progress: {msg}"

        if cap is not None:
            try:
                cap.release()
            except Exception as exc:  # noqa: BLE001 - the handle is already broken
                print(f"[CameraThread] recovery: release error ignored: {exc}")

        delay = config.CAMERA_REOPEN_BACKOFF_SEC
        while not self._stop_event.is_set():
            # Interruptible sleep — stop() cancels recovery immediately.
            if self._stop_event.wait(timeout=delay):
                return None
            new_cap = self._open_configured()
            if new_cap is not None:
                self._discard_warmup_frames(new_cap)
                with self._lock:
                    self._recovering = False
                    self._consecutive_read_failures = 0
                    self._error = None
                print(f"[CameraThread] session={self.session_id} RECOVERY: camera reopened "
                      f"after {attempt} attempt(s)")
                return new_cap
            delay = _next_backoff(delay)
            print(f"[CameraThread] session={self.session_id} RECOVERY: reopen failed — "
                  f"retrying in {delay:.1f}s  (available: "
                  f"{_available_video_devices() or ['none found']})")
            with self._lock:
                self._error = (f"camera unavailable — retrying every {delay:.0f}s "
                               f"(source={self.camera_source!r})")
        return None

    # ── thread body ───────────────────────────────────────────────────────

    def run(self) -> None:
        cap = self._open_configured()
        if cap is None:
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

        self._log_capture_settings(cap)
        if _debug:
            print(f"[CameraThread] DEBUG session={self.session_id} capture loop entering")
        self._discard_warmup_frames(cap)

        read_ok = 0
        read_fail_total = 0
        consecutive_fail = 0
        _ready_signaled = False

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    read_fail_total += 1
                    consecutive_fail += 1
                    with self._lock:
                        self._consecutive_read_failures = consecutive_fail
                    if consecutive_fail % 30 == 1:
                        print(f"[CameraThread] cap.read() failures: {read_fail_total} "
                              f"(consecutive={consecutive_fail} ok={read_ok})")
                    if consecutive_fail >= config.CAMERA_READ_FAIL_THRESHOLD:
                        cap = self._recover_capture(cap)
                        if cap is None:
                            break  # stop() requested during recovery
                        consecutive_fail = 0
                        continue
                    if self._stop_event.wait(timeout=0.05):
                        break
                    continue

                if consecutive_fail:
                    consecutive_fail = 0
                    with self._lock:
                        self._consecutive_read_failures = 0
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
                        args=(frame.copy(), gen),
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
            if cap is not None:
                cap.release()
            if _debug:
                print(f"[CameraThread] DEBUG session={self.session_id} released")
            print(f"[CameraThread] session={self.session_id} stopped.")

    def _run_inference_bg(self, frame, generation) -> None:
        try:
            self._run_inference(frame, generation)
        finally:
            # Stamp end-time BEFORE clearing the flag so the capture loop
            # uses the correct cooldown baseline on the very next frame tick.
            t_end = time.monotonic()
            with self._lock:
                self._inference_running = False
                self._last_inference_end = t_end

    def _run_inference(self, frame, generation) -> None:
        """Run one background inference and publish it, as a single transaction.

        Everything model-dependent happens inside one ModelManager inference
        session.  Holding the session across the publish is what closes the
        window where a package switch could land between "my generation is
        still current" and the actual write, leaving package B's UI showing
        package A's counts.

        The session serialises against the model, not against the capture loop,
        so live preview keeps running at full rate while this executes.
        """
        # Imported here (not at module top) to avoid a circular import at load time.
        import app.web as web_pkg
        from app.weight_verification import compute_weight_verification

        t_worker_start = time.perf_counter()
        try:
            manager = web_pkg.model_manager
            with manager.inference_session() as session:
                h, w = frame.shape[:2]
                imgsz = (config.CAMERA_INFERENCE_IMGSZ
                         if config.CAMERA_INFERENCE_IMGSZ_EXPLICIT else None)
                conf = config.CONF_THRESHOLD if config.CONF_THRESHOLD_EXPLICIT else None
                if _debug:
                    print(f"[CameraThread] session={self.session_id} "
                          f"inference worker gen={generation} "
                          f"package={session.package_id} frame={w}x{h} imgsz={imgsz}")

                # ── inference — ONNX RT releases the GIL; preview keeps running ──
                t0 = time.perf_counter()
                result = session.infer(frame, conf=conf, imgsz=imgsz)
                t_predict = time.perf_counter() - t0
                counts = result.counts

                # ── scale — non-blocking cached sample; serial I/O stays in the bg poll ──
                t0 = time.perf_counter()
                sample = web_pkg.scale_reader.get_latest_sample()
                weight = sample.value
                t_scale = time.perf_counter() - t0

                # ── annotated image is intentionally skipped for camera mode ────
                # The live preview is already the raw camera stream; generating an
                # annotated JPEG here wastes CPU and is never shown.

                t0 = time.perf_counter()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                std_snap = session.standards
                class_weights = session.class_weights
                wv = compute_weight_verification(
                    std_snap, class_weights, sample, config.WEIGHT_TOLERANCE
                )
                t_prep = time.perf_counter() - t0

                total_ms = (time.perf_counter() - t_worker_start) * 1000

                # Always print the timing summary so slowness is visible without
                # CAMERA_DEBUG.
                weight_str = f"{weight:.1f}g" if weight is not None else "None"
                print(
                    f"[CameraThread] session={self.session_id} "
                    f"inference gen={generation} model_gen={session.generation} "
                    f"package={session.package_id} "
                    f"predict={t_predict*1000:.0f}ms  "
                    f"scale_cached={t_scale*1000:.1f}ms({weight_str},"
                    f"{'stable' if sample.stable else sample.reason})  "
                    f"other={(t_prep)*1000:.0f}ms  "
                    f"total={total_ms:.0f}ms  "
                    f"dets={len(result.detections)}"
                )

                # The model package cannot have changed — the session pins it —
                # so the only staleness left is the operator stopping camera or
                # recognition while we ran.  Commit is a single linearization
                # point against stop_recognition(): either this result lands
                # while recognition was still running, or it is dropped whole.
                committed = self.commit_inference_result(generation, {
                    "counts": counts,
                    "weight": weight,
                    "timestamp": ts,
                    "package_id": session.package_id,
                    "package_display_name": session.display_name,
                    "model_generation": session.generation,
                    "weight_verification": wv,
                })

                if not committed:
                    print(f"[CameraThread] session={self.session_id} "
                          f"inference DISCARDED (stale) gen={generation} "
                          f"model_gen={session.generation}")
                    return
                if _debug:
                    print(f"[CameraThread] session={self.session_id} "
                          f"inference PUBLISHED gen={generation}")

                # Registers on the profile this inference actually ran against.
                session.register_classes(counts.keys())

                record = hist.make_record(
                    source="webcam",
                    counts=counts,
                    weight=weight,
                    standards_snapshot=std_snap,
                    package_id=session.package_id,
                    package_display_name=session.display_name,
                    preset_id=session.preset_id,
                    model_identity=(result.model_info.identity()
                                    if result.model_info else None),
                    class_weights_snapshot=class_weights,
                    weight_verification=wv,
                )
                hist.append_record(record)

        except Exception as e:
            print(f"[CameraThread] Inference error: {e}")

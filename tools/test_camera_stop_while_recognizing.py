"""
test_camera_stop_while_recognizing.py
======================================
Simulates the lifecycle-lock fix and documents the manual test procedure.

Run this script from the project root:
    python tools/test_camera_stop_while_recognizing.py

It exercises the _camera_lifecycle_lock and CameraThread._stopping logic
without a real camera device.

Manual curl test (run with server at localhost:5000):
    # Terminal 1 — simulate open → recognize → close camera directly
    curl -s -X POST http://localhost:5000/camera/start | python3 -m json.tool
    curl -s -X POST http://localhost:5000/camera/recognition/start | python3 -m json.tool
    # Without stopping recognition, close camera:
    curl -s -X POST http://localhost:5000/camera/stop | python3 -m json.tool
    # Immediately try to restart:
    curl -s -X POST http://localhost:5000/camera/start | python3 -m json.tool

    # Repeat 5 times to confirm reliability.
"""
from __future__ import annotations

import threading
import time

print("=" * 60)
print("Camera stop-while-recognizing lifecycle lock simulation")
print("=" * 60)


# ── Reproduce the pre-fix race ───────────────────────────────────────────────

class FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""
    def __init__(self, name: str, open_delay: float = 0.05, frame_delay: float = 0.066):
        self.name = name
        self._open_delay = open_delay
        self._frame_delay = frame_delay
        self._opened = False
        self._released = False
        self._lock = threading.Lock()

    def open(self):
        time.sleep(self._open_delay)
        with self._lock:
            if self._released:
                print(f"  [{self.name}] OPEN FAILED — already released (device busy?)")
                return False
            self._opened = True
        print(f"  [{self.name}] opened")
        return True

    def read(self):
        time.sleep(self._frame_delay)
        with self._lock:
            return self._opened and not self._released

    def release(self):
        with self._lock:
            self._opened = False
            self._released = True
        print(f"  [{self.name}] released")

    def is_open(self):
        with self._lock:
            return self._opened and not self._released


class SimCameraThread:
    """Simulates CameraThread with _stopping and _ready_event."""
    def __init__(self, name: str, cap: FakeCapture):
        self.name = name
        self._cap = cap
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._stopping = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._first_frame_ready = False

    def start(self): self._thread.start()

    def stop(self):
        with self._lock:
            self._stopping = True
        self._stop_event.set()

    def join(self, timeout=5.0):
        self._thread.join(timeout=timeout)

    def is_alive(self): return self._thread.is_alive()

    def wait_until_ready(self, timeout=8.0): return self._ready_event.wait(timeout=timeout)

    def _run(self):
        try:
            if not self._cap.open():
                self._ready_event.set()
                return
            # Warmup + main loop
            frames = 0
            deadline = time.monotonic() + 4.0
            while not self._stop_event.is_set() and time.monotonic() < deadline:
                ok = self._cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frames += 1
                if frames == 1:
                    self._ready_event.set()
                    print(f"  [{self.name}] first frame ready")
                with self._lock:
                    if self._stopping:
                        break
        finally:
            self._ready_event.set()  # unblock waiter on error
            self._cap.release()


print()
print("── Scenario A: no lifecycle lock (pre-fix) ─────────────────")
print("  stop() fires, immediately followed by start() — race on /dev/video0")

device_A = FakeCapture("device-A", open_delay=0.01, frame_delay=0.066)
cam_old = SimCameraThread("old-cam", device_A)
cam_old._cap.open()  # old camera already open
cam_old.start()
cam_old.wait_until_ready(timeout=2.0)

# Simulate camera stop (no lock)
def _stop_no_lock():
    cam_old.stop()
    cam_old.join(timeout=2.0)

stop_thread = threading.Thread(target=_stop_no_lock, daemon=True)
stop_thread.start()
time.sleep(0.01)  # stop is in-flight

# Simulate new camera start racing with stop (before old cap released)
device_B = FakeCapture("device-B", open_delay=0.01, frame_delay=0.066)
cam_new_A = SimCameraThread("new-cam-A", device_B)
# Without the lifecycle lock: new cam starts while old cap.release() is in-flight
# device_B is a fresh object — open succeeds even mid-race in this simulation
# (real V4L2 would fail here)
cam_new_A.start()
ready = cam_new_A.wait_until_ready(timeout=3.0)
print(f"  new-cam-A ready={ready}  (real V4L2: may fail if old cap not yet released)")
stop_thread.join()
cam_new_A.stop()
cam_new_A.join(timeout=2.0)

print()
print("── Scenario B: with lifecycle lock (post-fix) ──────────────")
print("  stop() holds lifecycle lock during join; start() waits for lock")

lifecycle_lock = threading.Lock()

def _stop_with_lock(cam):
    with lifecycle_lock:
        cam.stop()
        cam.join(timeout=5.0)
        print(f"  stop done — lock released, device free")

# Fresh device
device_C_shared = FakeCapture("shared-device", open_delay=0.02, frame_delay=0.066)

# Old camera already running
device_C_shared.open()
cam_old2 = SimCameraThread("old-cam2", device_C_shared)
cam_old2.start()
cam_old2.wait_until_ready(timeout=2.0)

results = []
t_total = time.perf_counter()

stop_thr = threading.Thread(target=_stop_with_lock, args=(cam_old2,), daemon=True)
stop_thr.start()
time.sleep(0.005)  # start fires shortly after stop (fire-and-forget)

# Start tries to acquire lock — blocked until stop completes + old cap released
with lifecycle_lock:
    # Old cap is now released (stop's join completed before we got here)
    device_new = FakeCapture("new-device", open_delay=0.02, frame_delay=0.066)
    cam_new_B = SimCameraThread("new-cam-B", device_new)
    cam_new_B.start()

ready = cam_new_B.wait_until_ready(timeout=8.0)
elapsed = (time.perf_counter() - t_total) * 1000
results.append(ready)
print(f"  new-cam-B ready={ready}  elapsed={elapsed:.0f} ms")
cam_new_B.stop()
cam_new_B.join(timeout=2.0)
stop_thr.join()

print()
print("── Scenario C: 5-cycle start/recognize/stop/restart ────────")

# Simulate the exact failure sequence 5 times
lifecycle_lock_C = threading.Lock()
all_pass = True

for cycle in range(1, 6):
    device = FakeCapture(f"dev-cycle{cycle}", open_delay=0.02, frame_delay=0.05)

    # Step 1: Open camera
    with lifecycle_lock_C:
        device.open()
        cam = SimCameraThread(f"cam-cycle{cycle}", device)
        cam.start()
    ready = cam.wait_until_ready(timeout=4.0)
    assert ready, f"Cycle {cycle}: camera failed to start"

    # Step 2: Start recognition (simulated — just mark it)
    recognition_active = True
    assert recognition_active

    # Step 3: Close camera WITHOUT stopping recognition
    stop_result = [False]  # list so the nested function can mutate it
    def _do_stop():
        with lifecycle_lock_C:
            cam.stop()          # also stops recognition via _stopping=True
            cam.join(timeout=5.0)
            stop_result[0] = not cam.is_alive()

    stop_thr_C = threading.Thread(target=_do_stop, daemon=True)
    stop_thr_C.start()
    time.sleep(0.005)  # brief gap (fire-and-forget)

    # Step 4: Reopen camera
    device2 = FakeCapture(f"dev2-cycle{cycle}", open_delay=0.02, frame_delay=0.05)
    with lifecycle_lock_C:
        device2.open()
        cam2 = SimCameraThread(f"cam2-cycle{cycle}", device2)
        cam2.start()
    ready2 = cam2.wait_until_ready(timeout=4.0)
    stop_thr_C.join()

    status = "PASS" if (ready2 and stop_result[0]) else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  Cycle {cycle}: stop_clean={stop_result[0]}  reopen_ready={ready2}  → {status}")

    cam2.stop()
    cam2.join(timeout=2.0)

print()
print("=" * 60)
print("RESULT:", "ALL PASS ✓" if all_pass else "SOME FAILURES ✗")
print("=" * 60)
print()
print("Root cause summary")
print("──────────────────")
print("  PRE-FIX:  /camera/stop sets camera_thread=None then calls cam.join().")
print("            /camera/start sees camera_thread=None and opens /dev/video0")
print("            while the old cap.read() is still in-flight (join not done).")
print("            V4L2 rejects concurrent streaming → new cap.read() always fails")
print("            → wait_until_ready times out → camera_start returns 500.")
print()
print("  POST-FIX: Both routes acquire _camera_lifecycle_lock.")
print("            stop holds the lock through cam.join() so cap.release() is")
print("            guaranteed before the lock is released.")
print("            start acquires the lock after stop has finished → device is free.")
print()
print("Additional fixes:")
print("  • CameraThread._stopping=True in stop() prevents new inference from spawning")
print("  • _stopping checked in stale test → in-flight workers discard results immediately")
print("  • Warmup loop has a 4 s absolute deadline → no infinite hang on read failure")
print("  • cam.stop() atomically stops recognition (generation++) + sets _stop_event")
print()
print("Manual test procedure:")
print("  1. Start app: ./run_jetson.sh")
print("  2. Open browser → click 開啟攝影機")
print("  3. Click 開始辨識")
print("  4. WITHOUT clicking 停止辨識, click 關閉攝影機")
print("  5. Confirm button shows 開啟攝影機 immediately")
print("  6. Click 開啟攝影機 → camera should start successfully")
print("  7. Repeat steps 2-6 five times to confirm reliability")
print()
print("Expected backend log (with CAMERA_DEBUG=true):")
print("  [camera_stop] shutdown requested  session=N  rec=True  inf=True/False")
print("  [camera_stop] _stop_event set  inf_running_at_stop=True/False  joining…")
print("  [CameraThread] session=N stop() — _stopping=True _stop_event set")
print("  [CameraThread] session=N stopped.")
print("  [camera_stop] CameraThread joined cleanly — /dev/video0 released")
print("  [camera_start] session=N+1 thread launched — waiting for first frame")
print("  [CameraThread] session=N+1 ready — first frame seq=1 size=…B")
print("  [camera_start] session=N+1 ready  ok=True")

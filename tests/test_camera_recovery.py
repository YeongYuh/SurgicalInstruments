"""Camera runtime recovery: backoff shape and stop-interruptibility.

Deliberately does NOT fake a V4L2 device — those tests are brittle and prove
little.  What matters and is testable here is the contract: backoff is bounded,
and stop() cancels recovery instead of leaving a thread sleeping.
"""

from __future__ import annotations

import threading

import app.config as config
from app.web.camera import CameraThread, _next_backoff


def test_backoff_doubles_and_is_capped():
    delay = config.CAMERA_REOPEN_BACKOFF_SEC
    seen = []
    for _ in range(12):
        delay = _next_backoff(delay)
        seen.append(delay)

    assert seen[0] > config.CAMERA_REOPEN_BACKOFF_SEC        # it grows
    assert max(seen) <= config.CAMERA_REOPEN_MAX_BACKOFF_SEC  # ...but is bounded
    assert seen[-1] == config.CAMERA_REOPEN_MAX_BACKOFF_SEC   # and saturates


def test_backoff_never_returns_zero():
    """A zero delay would turn recovery into a tight reopen loop."""
    assert _next_backoff(0.0) > 0.0
    assert _next_backoff(-5.0) > 0.0


def test_recovery_is_cancelled_by_stop():
    """stop() must abort recovery immediately, not after a full backoff."""
    cam = CameraThread(camera_source="/dev/null", session_id=1)
    cam.stop()

    finished = threading.Event()
    result = []

    def run():
        result.append(cam._recover_capture(None))
        finished.set()

    threading.Thread(target=run, daemon=True).start()

    assert finished.wait(timeout=2.0), "recovery ignored stop() and kept sleeping"
    assert result == [None]


def test_status_exposes_recovery_diagnostics():
    """A camera stuck in recovery must stay diagnosable from /camera/status."""
    cam = CameraThread(camera_source="/dev/video0", session_id=1)
    status = cam.get_status()

    for key in ("read_failures", "reopen_count", "recovering", "error",
                "model_generation", "package_id"):
        assert key in status
    assert status["recovering"] is False
    assert status["reopen_count"] == 0


def test_invalidate_results_clears_published_state_and_bumps_generation():
    """A package switch must not leave the previous model's counts on screen."""
    cam = CameraThread(camera_source="/dev/video0", session_id=1)
    cam.start_recognition()
    with cam._lock:
        cam._counts = {"widget": 3}
        cam._weight = 30.0
        cam._package_id = "ortho_tka"
    generation_before = cam.get_status()["recognition_generation"]

    cam.invalidate_results()

    result = cam.get_result()
    assert result["counts"] == {}
    assert result["weight"] is None
    assert result["package_id"] is None
    # In-flight workers holding the old generation now fail their staleness check.
    assert cam.get_status()["recognition_generation"] > generation_before


def test_recognition_generation_advances_on_every_transition():
    cam = CameraThread(camera_source="/dev/video0", session_id=1)
    generations = [cam.get_status()["recognition_generation"]]
    cam.start_recognition()
    generations.append(cam.get_status()["recognition_generation"])
    cam.stop_recognition()
    generations.append(cam.get_status()["recognition_generation"])
    cam.stop()
    generations.append(cam.get_status()["recognition_generation"])

    assert generations == sorted(generations)
    assert len(set(generations)) == len(generations)

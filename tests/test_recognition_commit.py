"""Stopping recognition and publishing a result share one linearization point.

Without it, a worker whose staleness check passed could still write the UI and
the history *after* the operator had been told recognition had stopped — the
inventory would grow a record nobody asked for.
"""

from __future__ import annotations

import threading

import pytest

import app.web as web_pkg
from app.web.camera import CameraThread


def _payload(counts=None, generation=1):
    return {
        "counts": counts if counts is not None else {"widget": 2},
        "weight": 20.0,
        "timestamp": "2026-05-21 10:00:00",
        "package_id": "ortho",
        "package_display_name": "骨科",
        "model_generation": generation,
        "weight_verification": {"ready": True, "passed": True},
    }


@pytest.fixture
def cam():
    web_pkg.reset_latest_state()
    camera = CameraThread(camera_source="/dev/video0", session_id=1)
    yield camera
    web_pkg.reset_latest_state()


# ── 14. stop wins ───────────────────────────────────────────────────────────

def test_result_is_refused_when_recognition_stopped_first(cam):
    cam.start_recognition()
    generation = cam.get_status()["recognition_generation"]

    cam.stop_recognition()        # operator pressed 停止辨識
    committed = cam.commit_inference_result(generation, _payload())

    assert committed is False
    assert cam.get_result()["counts"] == {}
    with web_pkg.state_lock:
        assert web_pkg.latest_state["counts"] == {}
        assert web_pkg.latest_state["timestamp"] == ""


def test_result_is_refused_after_the_camera_stops(cam):
    cam.start_recognition()
    generation = cam.get_status()["recognition_generation"]

    cam.stop()
    assert cam.commit_inference_result(generation, _payload()) is False
    with web_pkg.state_lock:
        assert web_pkg.latest_state["counts"] == {}


def test_result_from_an_older_recognition_run_is_refused(cam):
    cam.start_recognition()
    stale_generation = cam.get_status()["recognition_generation"]
    cam.stop_recognition()
    cam.start_recognition()       # a new run began

    assert cam.commit_inference_result(stale_generation, _payload()) is False


# ── 15. publication wins ────────────────────────────────────────────────────

def test_result_committed_before_the_stop_is_kept(cam):
    cam.start_recognition()
    generation = cam.get_status()["recognition_generation"]

    committed = cam.commit_inference_result(generation, _payload())
    cam.stop_recognition()        # stop arrives after the commit

    assert committed is True
    result = cam.get_result()
    assert result["counts"] == {"widget": 2}
    assert result["package_id"] == "ortho"
    assert result["model_generation"] == 1
    with web_pkg.state_lock:
        assert web_pkg.latest_state["counts"] == {"widget": 2}
        assert web_pkg.latest_state["package_id"] == "ortho"


# ── 16. camera-local and global state move together ─────────────────────────

def test_commit_updates_camera_and_application_state_atomically(cam):
    """Either both are written, or neither is — never half a result."""
    cam.start_recognition()
    generation = cam.get_status()["recognition_generation"]

    assert cam.commit_inference_result(generation, _payload()) is True
    assert cam.get_result()["counts"] == {"widget": 2}
    with web_pkg.state_lock:
        assert web_pkg.latest_state["counts"] == {"widget": 2}


def test_no_partial_state_under_concurrent_stop_and_commit(cam):
    """Race stop against commit repeatedly; the invariant must always hold.

    The assertion is on consistency, not on which side wins — either outcome is
    correct, a mixture is not.
    """
    for _ in range(60):
        web_pkg.reset_latest_state()
        cam.start_recognition()
        generation = cam.get_status()["recognition_generation"]
        start = threading.Barrier(2, timeout=10)
        outcome = {}

        def committer():
            start.wait()
            outcome["committed"] = cam.commit_inference_result(
                generation, _payload(generation=generation))

        def stopper():
            start.wait()
            cam.stop_recognition()

        threads = [threading.Thread(target=committer), threading.Thread(target=stopper)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        committed = outcome.get("committed")
        camera_counts = cam.get_result()["counts"]
        with web_pkg.state_lock:
            global_counts = dict(web_pkg.latest_state["counts"])

        if committed:
            assert camera_counts == {"widget": 2}
            assert global_counts == {"widget": 2}, "camera published but UI did not"
        else:
            assert camera_counts == {}
            assert global_counts == {}, "UI shows a result the camera refused"

        cam.invalidate_results()


def test_commit_returns_false_when_recognition_was_never_started(cam):
    assert cam.commit_inference_result(0, _payload()) is False
    with web_pkg.state_lock:
        assert web_pkg.latest_state["counts"] == {}

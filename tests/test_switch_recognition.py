"""Recognition is the operator's intent — a package switch must preserve it."""

from __future__ import annotations

import json
import threading

import pytest

import app.web as web_pkg
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import make_manager, write_package


class FakeCamera:
    """Just enough CameraThread surface for the switch route."""

    def __init__(self, running: bool = True, recognizing: bool = False) -> None:
        self._running = running
        self._recognizing = recognizing
        self.generation = 0
        self.invalidated = 0
        self.start_calls = 0
        self.stop_calls = 0
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self._running

    def stop_camera(self) -> None:
        self._running = False
        self._recognizing = False

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "stopping": not self._running,
            "recognition_running": self._recognizing,
            "inference_running": False,
            "recognition_generation": self.generation,
            "session_id": 1,
        }

    def start_recognition(self) -> None:
        self.start_calls += 1
        self.generation += 1
        self._recognizing = True

    def stop_recognition(self) -> None:
        self.stop_calls += 1
        self.generation += 1
        self._recognizing = False

    def invalidate_results(self) -> None:
        self.invalidated += 1
        self.generation += 1


@pytest.fixture
def api(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(manager.packages_dir, "b",
                  class_weights={"forceps": 7.5}, standards={"forceps": 4},
                  adapter_options={"counts": {"forceps": 4}})
    write_package(manager.packages_dir, "bad", adapter_options={"fail_load": True})
    manager.activate("a")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, manager


def _switch(client, package_id):
    response = client.post("/api/model-package", json={"id": package_id})
    return response, json.loads(response.data.decode("utf-8"))


def _set_camera(monkeypatch, cam):
    monkeypatch.setattr(web_pkg, "camera_thread", cam, raising=False)


# ── 12. recognition ON + successful switch -> recognition ON ────────────────

def test_successful_switch_resumes_recognition(api, monkeypatch):
    """Changing models should not make the operator press 開始辨識 again."""
    client, manager = api
    cam = FakeCamera(running=True, recognizing=True)
    _set_camera(monkeypatch, cam)

    response, payload = _switch(client, "b")

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["recognition_was_running"] is True
    assert payload["recognition_resumed"] is True
    assert cam.stop_calls == 1          # paused for the switch
    assert cam.start_calls == 1         # ...and put back
    assert cam.get_status()["recognition_running"] is True
    assert cam.invalidated == 1         # A's counts were dropped
    assert manager.state().package_id == "b"


# ── 13. recognition ON + failed switch -> recognition ON ────────────────────

def test_failed_switch_restores_recognition(api, monkeypatch):
    """A failed switch must leave the old package genuinely serving.

    Rolling the model back but leaving recognition stopped would be a silent
    downgrade of a working system.
    """
    client, manager = api
    cam = FakeCamera(running=True, recognizing=True)
    _set_camera(monkeypatch, cam)

    response, payload = _switch(client, "bad")

    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["active"] == "a"
    assert payload["model_ready"] is True
    assert payload["recognition_was_running"] is True
    assert payload["recognition_resumed"] is True
    assert cam.get_status()["recognition_running"] is True
    assert manager.state().package_id == "a"
    assert manager.state().ready is True


# ── 14. recognition OFF + switch -> stays OFF ───────────────────────────────

def test_switch_does_not_start_recognition_that_was_not_running(api, monkeypatch):
    client, _ = api
    cam = FakeCamera(running=True, recognizing=False)
    _set_camera(monkeypatch, cam)

    response, payload = _switch(client, "b")

    assert payload["ok"] is True
    assert payload["recognition_was_running"] is False
    assert payload["recognition_resumed"] is False
    assert cam.start_calls == 0
    assert cam.get_status()["recognition_running"] is False


def test_failed_switch_does_not_start_recognition_that_was_not_running(api, monkeypatch):
    client, _ = api
    cam = FakeCamera(running=True, recognizing=False)
    _set_camera(monkeypatch, cam)

    _, payload = _switch(client, "bad")

    assert payload["ok"] is False
    assert payload["recognition_resumed"] is False
    assert cam.start_calls == 0


# ── 15. camera stopped during the switch ────────────────────────────────────

def test_recognition_is_not_restarted_if_the_camera_stopped(api, monkeypatch):
    """The operator closed the camera mid-switch — do not reopen the workflow."""
    client, manager = api
    cam = FakeCamera(running=True, recognizing=True)
    _set_camera(monkeypatch, cam)

    original_activate = manager.activate

    def activate_then_camera_stops(*args, **kwargs):
        result = original_activate(*args, **kwargs)
        cam.stop_camera()          # camera goes away while the model is loading
        return result

    monkeypatch.setattr(manager, "activate", activate_then_camera_stops)

    response, payload = _switch(client, "b")

    assert payload["ok"] is True
    assert payload["recognition_resumed"] is False
    assert cam.start_calls == 0
    assert cam.get_status()["recognition_running"] is False


def test_switch_with_no_camera_at_all(api, monkeypatch):
    client, _ = api
    _set_camera(monkeypatch, None)

    response, payload = _switch(client, "b")

    assert payload["ok"] is True
    assert payload["camera_running"] is False
    assert payload["recognition_was_running"] is False
    assert payload["recognition_resumed"] is False


def test_already_active_switch_reports_current_recognition(api, monkeypatch):
    client, _ = api
    cam = FakeCamera(running=True, recognizing=True)
    _set_camera(monkeypatch, cam)

    _, payload = _switch(client, "a")

    assert payload["status"] == "already_active"
    assert payload["recognition_was_running"] is True
    assert payload["recognition_resumed"] is True
    assert cam.stop_calls == 0        # nothing was disturbed
    assert cam.start_calls == 0

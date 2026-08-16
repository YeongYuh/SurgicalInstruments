"""No API may ever combine one package's result with another's configuration.

The dangerous window is between the manager publishing package B and the
application state (latest_state, camera result) being cleared.  Two defences
are tested here:

* the switch listener clears that state *inside* the switch transaction, so the
  window does not exist for a manager wired up the way the app wires it;
* every reader sanitizes what it finds anyway, so even a manager without the
  listener cannot serve B's standards against A's counts.
"""

from __future__ import annotations

import io
import json
import threading

import cv2
import numpy as np
import pytest

import app.web as web_pkg
from app.runtime_result import result_matches_active, sanitize_runtime_result
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import make_manager, write_package


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A manager deliberately WITHOUT the app's switch listener.

    That reproduces the raw window: activate() returns with B active while the
    application state still holds A's counts.
    """
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "ortho",
                  display_name="骨科", class_weights={"widget": 10.0},
                  standards={"widget": 2}, adapter_options={"counts": {"widget": 2}})
    write_package(manager.packages_dir, "obgyn",
                  display_name="婦產科", class_weights={"forceps": 7.5},
                  standards={"forceps": 4}, adapter_options={"counts": {"forceps": 4}})
    manager.activate("ortho")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, manager


def _json(response):
    return json.loads(response.data.decode("utf-8"))


def _produce_result(client):
    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    payload = _json(client.post(
        "/upload",
        data={"image": (io.BytesIO(buf.tobytes()), "f.jpg")},
        content_type="multipart/form-data",
    ))
    assert payload["ok"] is True
    return payload


class _FakeCam:
    def __init__(self, result):
        self._result = result

    def is_running(self):
        return True

    def get_status(self):
        return {"running": True, "stopping": False, "recognition_running": False,
                "inference_running": False, "recognition_generation": 0,
                "recovering": False}

    def get_result(self):
        return dict(self._result)

    def invalidate_results(self):
        self._result = {"timestamp": "", "counts": {}, "weight": None,
                        "package_id": None, "package_display_name": None,
                        "model_generation": 0, "annotated_b64": None}


# ── 1. /status must not expose A's counts under B ───────────────────────────

def test_status_drops_counts_from_a_superseded_package(api):
    client, manager = api
    _produce_result(client)
    assert _json(client.get("/status"))["counts"] == {"widget": 2}

    # Switch the manager directly — this is exactly the window between publish
    # and the route clearing application state.
    manager.activate("obgyn")

    status = _json(client.get("/status"))

    assert status["active_package"] == "obgyn"
    assert status["result_stale"] is True
    assert status["counts"] == {}                 # never obgyn + ortho counts
    assert status["weight_verification"] is None
    assert status["timestamp"] == ""
    # Diagnostics survive so the blank panel is explainable.
    assert status["result_package_id"] == "ortho"
    assert status["result_model_generation"] == 1


# ── 2. /camera/result must not verify A's result with B's standards ─────────

def test_camera_result_does_not_reinterpret_a_stale_result(api, monkeypatch):
    client, manager = api
    stale = {
        "timestamp": "2026-05-21 10:00:00",
        "counts": {"widget": 2},
        "weight": 20.0,
        "package_id": "ortho",
        "package_display_name": "骨科",
        "model_generation": 1,
        "annotated_b64": None,
    }
    monkeypatch.setattr(web_pkg, "camera_thread", _FakeCam(stale), raising=False)

    fresh = _json(client.get("/camera/result"))
    assert fresh["counts"] == {"widget": 2}
    assert fresh["result_stale"] is False
    assert fresh["weight_verification"]["expected"] == 20.0

    manager.activate("obgyn")

    result = _json(client.get("/camera/result"))

    assert result["result_stale"] is True
    assert result["counts"] == {}
    # The killer case: obgyn expects 30 g; verifying ortho's tray against that
    # would invent a discrepancy in a department that was never weighed.
    assert result["weight_verification"] is None
    assert result["active_package"] == "obgyn"


# ── 3. /bom_report must not mix A counts with B standards ───────────────────

def test_bom_report_refuses_a_stale_result_instead_of_mixing_packages(api):
    """Not merely "drop the stale counts" — refuse the report entirely.

    A BOM listing every obstetric instrument as detected=0 is indistinguishable
    from a tray that was actually scanned and found empty, and it would be
    downloaded, filed, and believed.
    """
    client, manager = api
    _produce_result(client)

    before = client.get("/bom_report")
    assert before.status_code == 200
    assert "widget" in before.data.decode("utf-8-sig")

    manager.activate("obgyn")

    after = client.get("/bom_report")
    payload = _json(after)

    assert after.status_code == 409
    assert payload["ok"] is False
    assert payload["result_status"] == "stale"
    assert payload["active_package"] == "obgyn"
    # No fabricated obstetric report was produced at all.
    assert "forceps" not in after.data.decode("utf-8")


# ── 4 & 5. generation, not just the package id ──────────────────────────────

def test_generation_mismatch_alone_makes_a_result_stale(api):
    _client, manager = api
    state = manager.state()

    assert result_matches_active("ortho", state.generation, state) is True
    # Same package, older activation: the profile may have been reseeded, so the
    # result is not comparable either.
    assert result_matches_active("ortho", state.generation - 1, state) is False
    assert result_matches_active("obgyn", state.generation, state) is False
    assert result_matches_active(None, state.generation, state) is False
    assert result_matches_active("ortho", None, state) is False


def test_camera_result_stale_on_generation_mismatch_only(api, monkeypatch):
    client, manager = api
    # Same package id, but produced under an activation that no longer exists.
    stale = {
        "timestamp": "2026-05-21 10:00:00",
        "counts": {"widget": 2},
        "weight": 20.0,
        "package_id": "ortho",
        "package_display_name": "骨科",
        "model_generation": 99,
        "annotated_b64": None,
    }
    monkeypatch.setattr(web_pkg, "camera_thread", _FakeCam(stale), raising=False)

    result = _json(client.get("/camera/result"))

    assert result["result_stale"] is True
    assert result["counts"] == {}


def test_empty_result_is_not_reported_as_stale(api):
    client, _manager = api
    status = _json(client.get("/status"))
    assert status["counts"] == {}
    assert status["result_stale"] is False    # nothing has happened yet


# ── the switch listener closes the window entirely ──────────────────────────

def test_switch_listener_clears_state_inside_the_transaction(api):
    """With the listener wired up, the window does not exist at all."""
    client, manager = api
    _produce_result(client)

    cleared = {}

    def listener(state):
        # Runs inside the lifecycle gate, before activate() returns.
        with web_pkg.state_lock:
            web_pkg.latest_state.update({"timestamp": "", "counts": {},
                                         "weight": None, "package_id": None,
                                         "weight_verification": None,
                                         "model_generation": state.generation})
        cleared["at_generation"] = state.generation

    manager.add_switch_listener(listener)
    manager.activate("obgyn")

    assert cleared["at_generation"] == 2
    status = _json(client.get("/status"))
    assert status["counts"] == {}
    assert status["result_stale"] is False     # cleared, not merely suppressed


def test_readers_are_safe_while_the_switch_is_still_inside_the_gate(api):
    """A request landing during the switch cannot see B + A."""
    client, manager = api
    _produce_result(client)

    in_window = threading.Barrier(2, timeout=10)
    release = threading.Event()

    def blocking_listener(_state):
        in_window.wait()          # B is published; state not yet cleared
        release.wait(timeout=10)

    manager.add_switch_listener(blocking_listener)

    switch = threading.Thread(target=lambda: manager.activate("obgyn"), daemon=True)
    switch.start()
    in_window.wait(timeout=10)

    try:
        status = _json(client.get("/status"))
        camera = _json(client.get("/camera/result")) if web_pkg.camera_thread else None
        bom = client.get("/bom_report").data.decode("utf-8-sig")

        assert status["active_package"] == "obgyn"
        assert status["counts"] == {}, "B package served with A counts"
        assert status["result_stale"] is True
        assert "widget,2" not in bom.replace(", ", ",")
        if camera is not None:
            assert camera["counts"] == {}
    finally:
        release.set()
        switch.join(timeout=10)


# ── the helper itself ───────────────────────────────────────────────────────

def test_sanitize_keeps_a_current_result_untouched():
    class _State:
        configured = True
        package_id = "ortho"
        generation = 3

    result = {"timestamp": "t", "counts": {"a": 1}, "weight": 5.0,
              "package_id": "ortho", "model_generation": 3,
              "weight_verification": {"passed": True}}

    out = sanitize_runtime_result(result, _State())

    assert out["result_stale"] is False
    assert out["counts"] == {"a": 1}
    assert out["weight_verification"] == {"passed": True}


def test_sanitize_without_an_active_package_is_stale():
    class _State:
        configured = False
        package_id = ""
        generation = 0

    result = {"timestamp": "t", "counts": {"a": 1}, "package_id": "ortho",
              "model_generation": 1}

    out = sanitize_runtime_result(result, _State())

    assert out["result_stale"] is True
    assert out["counts"] == {}

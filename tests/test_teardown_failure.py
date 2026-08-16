"""Failure paths where releasing a model is what goes wrong.

The invariant these protect is the one the whole design rests on: at most one
heavy model resident at a time.  A 4 GB Jetson does not survive two, and the
most dangerous moment is a rollback — the point where the code is most tempted
to load something while the previous model's residency is in doubt.
"""

from __future__ import annotations

import json

import pytest

import app.web as web_pkg
from app.inference.manager import (
    ModelManagerError,
    ModelNotReadyError,
    ModelTeardownError,
)
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import RESIDENT, FakeCamera, make_manager, write_package


def _json(response):
    return json.loads(response.data.decode("utf-8"))


def _manager_with(tmp_path, target_options):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(manager.packages_dir, "b",
                  class_weights={"forceps": 7.5}, standards={"forceps": 4},
                  adapter_options=dict(counts={"forceps": 4}, **target_options))
    write_package(manager.packages_dir, "c",
                  class_weights={"clamp": 1.0}, standards={"clamp": 1},
                  adapter_options={"counts": {"clamp": 1}})
    manager.activate("a")
    return manager


# ── 1-6. the target loaded, then failed, then would not let go ──────────────

def test_target_cleanup_failure_never_reloads_the_previous_model(tmp_path):
    """B loads, B's warmup fails, B refuses to unload.

    Reloading A now could leave BOTH resident.  The rollback is abandoned and
    the manager says so, rather than gambling the memory the unit runs on.
    """
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    old_adapter = manager.active_adapter
    loads_before = old_adapter.load_calls
    generation_before = manager.generation

    with pytest.raises(ModelTeardownError) as excinfo:
        manager.activate("b")

    message = str(excinfo.value)
    assert "could not be released" in message

    # A was NOT brought back.
    assert old_adapter.load_calls == loads_before
    assert old_adapter.loaded is False

    # One model's worth of memory is unaccounted for — and only one.
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1

    assert manager.generation == generation_before
    assert manager.state().ready is False
    assert manager.model_error
    assert "released" in manager.model_error
    assert manager.last_switch_error
    assert manager.recovery_required is True


def test_target_compatibility_failure_with_cleanup_failure_also_stops(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  adapter="class_listing",
                  class_weights={"widget": 10.0}, standards={"widget": 2},
                  adapter_options={"model_classes": ["widget"],
                                   "counts": {"widget": 2}})
    write_package(manager.packages_dir, "b",
                  adapter="class_listing",
                  class_weights={"forceps": 7.5, "ghost": 1.0},
                  standards={"forceps": 4, "ghost": 1},
                  adapter_options={"model_classes": ["forceps"],   # ghost missing
                                   "fail_unload": True})
    manager.activate("a")
    old_adapter = manager.active_adapter
    loads_before = old_adapter.load_calls

    with pytest.raises(ModelTeardownError):
        manager.activate("b")

    assert old_adapter.load_calls == loads_before      # A never reloaded
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1
    assert manager.recovery_required is True


def test_load_failure_alone_still_rolls_back_normally(tmp_path):
    """A target that never loaded has nothing to release — rollback proceeds."""
    manager = _manager_with(tmp_path, {"fail_load": True, "fail_unload": True})
    old_adapter = manager.active_adapter

    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("b")
    assert not isinstance(excinfo.value, ModelTeardownError)

    assert manager.state().package_id == "a"
    assert manager.state().ready is True
    assert old_adapter.loaded is True
    assert manager.recovery_required is False
    assert RESIDENT.current == 1


# ── 7. nothing may load once residency is unknown ───────────────────────────

def test_no_further_model_may_load_after_an_unsafe_teardown(tmp_path):
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    with pytest.raises(ModelTeardownError):
        manager.activate("b")

    resident_after = RESIDENT.current

    with pytest.raises(ModelTeardownError) as excinfo:
        manager.activate("c")
    assert "restart" in str(excinfo.value)

    # C was never even constructed, let alone loaded.
    assert RESIDENT.current == resident_after
    assert RESIDENT.peak <= 1


def test_bootstrap_is_also_refused_after_an_unsafe_teardown(tmp_path):
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    with pytest.raises(ModelTeardownError):
        manager.activate("b")

    assert manager.bootstrap("c", background=False) is None
    assert RESIDENT.peak <= 1


def test_inference_is_refused_after_an_unsafe_teardown(tmp_path):
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    with pytest.raises(ModelTeardownError):
        manager.activate("b")

    with pytest.raises(ModelNotReadyError):
        with manager.inference_session():
            pass


def test_unsafe_teardown_is_reported_over_http(tmp_path, monkeypatch):
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.app.config["TESTING"] = True

    with web_pkg.app.test_client() as client:
        first = client.post("/api/model-package", json={"id": "b"})
        payload = _json(first)

        assert first.status_code == 503
        assert payload["ok"] is False
        assert payload["recovery_required"] is True
        assert payload["model_ready"] is False
        assert payload["recognition_resumed"] is False
        assert payload["model_error"]

        # And it stays refused.
        second = client.post("/api/model-package", json={"id": "c"})
        assert second.status_code == 503
        assert _json(second)["recovery_required"] is True

        status = _json(client.get("/status"))
        assert status["model_ready"] is False
        assert status["recovery_required"] is True


# ── 20. recognition must not resume into an unusable model ──────────────────

def test_recognition_is_not_resumed_after_an_unsafe_teardown(tmp_path, monkeypatch):
    manager = _manager_with(tmp_path, {"fail_warmup": True, "fail_unload": True})
    cam = FakeCamera(running=True, recognizing=True)
    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", cam, raising=False)
    web_pkg.app.config["TESTING"] = True

    with web_pkg.app.test_client() as client:
        response = client.post("/api/model-package", json={"id": "b"})

    payload = _json(response)
    assert response.status_code == 503
    assert payload["recognition_resumed"] is False
    assert cam.start_calls == 0                 # never restarted
    assert cam.get_status()["recognition_running"] is False
    # The camera itself keeps running — preview is a platform function and does
    # not depend on the model.
    assert payload["camera_running"] is True


# ── 8 & 9. a rollback that reloads but cannot warm up ───────────────────────

def test_rollback_warmup_failure_is_not_reported_ready(tmp_path, monkeypatch):
    """A model that reloads but cannot run is not "back in service"."""
    manager = _manager_with(tmp_path, {"fail_load": True})
    old_adapter = manager.active_adapter

    def refuse_warmup():
        raise RuntimeError("simulated: old model cannot warm up again")

    monkeypatch.setattr(old_adapter, "_do_warmup", refuse_warmup)

    with pytest.raises(ModelManagerError):
        manager.activate("b")

    assert manager.state().ready is False
    assert manager.model_error
    assert "rollback failed" in manager.model_error
    assert manager.last_switch_error
    assert "activate package 'b'" in manager.last_switch_error


def test_rollback_warmup_failure_does_not_resume_recognition(tmp_path, monkeypatch):
    manager = _manager_with(tmp_path, {"fail_load": True})
    old_adapter = manager.active_adapter

    def refuse_warmup():
        raise RuntimeError("simulated warmup failure on rollback")

    monkeypatch.setattr(old_adapter, "_do_warmup", refuse_warmup)

    cam = FakeCamera(running=True, recognizing=True)
    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", cam, raising=False)
    web_pkg.app.config["TESTING"] = True

    with web_pkg.app.test_client() as client:
        response = client.post("/api/model-package", json={"id": "b"})
    payload = _json(response)

    assert response.status_code == 400
    assert payload["model_ready"] is False
    assert payload["recognition_resumed"] is False
    assert cam.start_calls == 0


# ── 10 & 11. a successful rollback restores A completely ────────────────────

def test_successful_rollback_restores_the_previous_compatibility(tmp_path):
    """Diagnostics must describe the model that is actually active."""
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  adapter="class_listing",
                  class_weights={"widget": 10.0, "spare": 1.0},
                  standards={"widget": 2},
                  adapter_options={"model_classes": ["widget"],
                                   "counts": {"widget": 2}})
    write_package(manager.packages_dir, "b",
                  adapter="class_listing",
                  class_weights={"forceps": 7.5}, standards={"forceps": 4},
                  adapter_options={"model_classes": ["forceps"],
                                   "fail_warmup": True})
    manager.activate("a")
    compatibility_before = manager.compatibility
    assert compatibility_before["model_classes"] == 1
    assert compatibility_before["unused_class_weights"] == ["spare"]

    with pytest.raises(ModelManagerError):
        manager.activate("b")

    after = manager.compatibility
    assert after == compatibility_before, "B's diagnostics survived the rollback"
    assert manager.state().package_id == "a"
    assert manager.state().ready is True
    assert manager.model_error is None
    assert manager.last_switch_error


def test_successful_rollback_is_ready_and_serves_inference(tmp_path):
    manager = _manager_with(tmp_path, {"fail_warmup": True})

    with pytest.raises(ModelManagerError):
        manager.activate("b")

    assert manager.state().ready is True
    assert manager.state().package_id == "a"
    assert manager.model_error is None
    assert manager.recovery_required is False
    assert manager.infer(object()).counts == {"widget": 2}
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1


# ── 11 (base). lenient teardown must not claim the memory came back ─────────

def test_lenient_unload_reports_an_unknown_resource_state(tmp_path):
    manager = _manager_with(tmp_path, {})
    write_package(manager.packages_dir, "sticky",
                  class_weights={"x": 1.0}, standards={"x": 1},
                  adapter_options={"fail_unload": True})
    manager.activate("sticky")
    adapter = manager.active_adapter
    assert adapter.resource_state == "held"

    adapter.unload(strict=False)          # shutdown-style best effort

    assert adapter.loaded is False        # we will not use it again
    assert adapter.teardown_error         # ...but we do not claim it was freed
    assert adapter.resource_state == "unknown"


def test_clean_unload_reports_released(tmp_path):
    manager = _manager_with(tmp_path, {})
    adapter = manager.active_adapter
    assert adapter.resource_state == "held"
    adapter.unload(strict=False)
    assert adapter.teardown_error is None
    assert adapter.resource_state == "released"

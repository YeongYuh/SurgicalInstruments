"""Startup readiness gate and model teardown safety.

Startup must apply exactly the same correctness bar as a runtime switch — a
unit that boots into a state a switch would have refused is worse than one that
refuses to start, because nobody notices.
"""

from __future__ import annotations

import io
import json
import threading

import pytest

import app.web as web_pkg
from app.inference.base import AdapterError, AdapterUnavailableError
from app.inference.manager import ModelManagerError, ModelNotReadyError
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import RESIDENT, control_for, make_manager, write_package


COMPLETE_WAIT_SEC = 10.0
BLOCKED_CHECK_SEC = 0.3


def _json(response):
    return json.loads(response.data.decode("utf-8"))


class _ClassListAdapterPackage:
    """Helper marker — see write_class_listing_package below."""


def write_class_listing_package(packages_dir, package_id, *, model_classes,
                                class_weights, standards, **kwargs):
    """A package whose adapter publishes a fixed model class list."""
    return write_package(
        packages_dir, package_id,
        adapter="class_listing",
        class_weights=class_weights,
        standards=standards,
        adapter_options=dict(model_classes=list(model_classes), **kwargs),
    )


# ── 6. startup must enforce compatibility, like a switch does ───────────────

def test_startup_refuses_a_model_that_cannot_see_an_expected_instrument(tmp_path):
    manager = make_manager(tmp_path)
    write_class_listing_package(
        manager.packages_dir, "mismatch",
        model_classes=["scissors"],
        class_weights={"scissors": 5.0, "forceps": 3.0},
        standards={"scissors": 1, "forceps": 2},   # forceps: model cannot detect it
    )

    manager.bootstrap("mismatch", background=False)

    state = manager.state()
    assert state.configured is True      # the UI can still show what is broken
    assert state.ready is False
    assert manager.model_error
    assert "forceps" in manager.model_error
    assert manager.compatibility["standards_not_in_model"] == ["forceps"]


def test_startup_accepts_a_fully_compatible_model(tmp_path):
    manager = make_manager(tmp_path)
    write_class_listing_package(
        manager.packages_dir, "good",
        model_classes=["scissors", "forceps"],
        class_weights={"scissors": 5.0, "forceps": 3.0},
        standards={"scissors": 1, "forceps": 2},
    )

    manager.bootstrap("good", background=False)

    assert manager.state().ready is True
    assert manager.model_error is None
    assert manager.compatibility["standards_not_in_model"] == []


def test_startup_and_switch_apply_the_same_bar(tmp_path):
    """The same package must be refused by both paths, not just one."""
    manager = make_manager(tmp_path)
    write_class_listing_package(
        manager.packages_dir, "good", model_classes=["a"],
        class_weights={"a": 1.0}, standards={"a": 1})
    write_class_listing_package(
        manager.packages_dir, "mismatch", model_classes=["a"],
        class_weights={"a": 1.0, "b": 1.0}, standards={"a": 1, "b": 1})

    manager.bootstrap("good", background=False)
    assert manager.state().ready is True

    with pytest.raises(ModelManagerError):
        manager.activate("mismatch")
    assert manager.state().package_id == "good"

    other = make_manager(tmp_path / "second")
    write_class_listing_package(
        other.packages_dir, "mismatch", model_classes=["a"],
        class_weights={"a": 1.0, "b": 1.0}, standards={"a": 1, "b": 1})
    other.bootstrap("mismatch", background=False)
    assert other.state().ready is False


# ── 7. startup warmup failure means not ready ───────────────────────────────

def test_startup_warmup_failure_is_not_ready(tmp_path):
    """Warmup runs a real inference — if it fails the model cannot run."""
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "cold",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_warmup": True})

    manager.bootstrap("cold", background=False)

    state = manager.state()
    assert state.configured is True
    assert state.ready is False
    assert state.adapter.warmup_error
    assert manager.model_error
    assert "warmup" in manager.model_error
    assert manager.wait_until_ready(timeout=0.1) is False


def test_startup_warmup_failure_blocks_inference(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "cold",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_warmup": True})
    manager.bootstrap("cold", background=False)

    with pytest.raises(ModelNotReadyError):
        with manager.inference_session():
            pass


# ── 8 & 9. no inference may bypass the readiness gate ───────────────────────

def test_inference_session_refuses_while_loading(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "slow",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"load_delay": 0.4})
    manager.bootstrap("slow", background=True)

    # Refused immediately, without queueing behind the load and then quietly
    # succeeding — and without lazily loading the model itself.
    with pytest.raises(ModelNotReadyError) as excinfo:
        with manager.inference_session():
            pass
    assert excinfo.value.loading is True

    assert manager.wait_until_ready(timeout=COMPLETE_WAIT_SEC) is True
    with manager.inference_session() as session:
        assert session.package_id == "slow"


def test_upload_during_startup_load_returns_503(tmp_path, monkeypatch):
    """An early upload must not lazily load an unverified model."""
    import cv2
    import numpy as np

    manager = make_manager(tmp_path)
    control = control_for("slow")
    write_package(manager.packages_dir, "slow",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"load_delay": 0.5, "counts": {"widget": 1}})

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(1.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.reset_latest_state()
    web_pkg.app.config["TESTING"] = True

    manager.bootstrap("slow", background=True)

    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    with web_pkg.app.test_client() as client:
        response = client.post(
            "/upload",
            data={"image": (io.BytesIO(buf.tobytes()), "f.jpg")},
            content_type="multipart/form-data",
        )
        payload = _json(response)

        assert response.status_code == 503
        assert payload["ok"] is False
        assert payload["model_loading"] is True
        # Nothing was published while the model was unverified.
        assert hist.load_history() == []
        assert _json(client.get("/status"))["counts"] == {}

        assert manager.wait_until_ready(timeout=COMPLETE_WAIT_SEC) is True
        later = client.post(
            "/upload",
            data={"image": (io.BytesIO(buf.tobytes()), "f.jpg")},
            content_type="multipart/form-data",
        )
        assert later.status_code == 200
        assert _json(later)["ok"] is True

    control.release.set()


def test_upload_on_a_broken_model_returns_503_not_loading(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "boom",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_load": True})
    manager.bootstrap("boom", background=False)

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(1.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(web_pkg, "camera_thread", None, raising=False)
    web_pkg.app.config["TESTING"] = True

    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    with web_pkg.app.test_client() as client:
        response = client.post(
            "/upload",
            data={"image": (io.BytesIO(buf.tobytes()), "f.jpg")},
            content_type="multipart/form-data",
        )
    payload = _json(response)
    assert response.status_code == 503
    assert payload["model_loading"] is False
    assert payload["model_error"]


# ── 10, 11 & 12. a failed unload must stop the switch ───────────────────────

def test_unload_failure_prevents_loading_the_replacement(tmp_path):
    """If teardown fails the weights may still be held.

    Loading a second model on top of them would break the one-resident-model
    invariant that keeps a 4 GB board alive, so the switch is abandoned.
    """
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_unload": True, "counts": {"widget": 1}})
    write_package(manager.packages_dir, "b",
                  class_weights={"forceps": 1.0}, standards={"forceps": 1},
                  adapter_options={"counts": {"forceps": 1}})
    manager.activate("a")
    old = manager.active_adapter
    generation_before = manager.generation

    with pytest.raises(ModelManagerError) as excinfo:
        manager.activate("b")
    assert "releasing the current model failed" in str(excinfo.value)

    # The replacement was never touched.
    entry = manager.get_discovered("b")
    assert entry is not None
    assert manager.state().package_id == "a"
    assert manager.generation == generation_before
    assert old.loaded is True            # not falsely reported as unloaded
    assert RESIDENT.current == 1
    assert RESIDENT.peak <= 1


def test_unload_failure_is_reported_as_a_switch_error_not_a_model_error(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_unload": True, "counts": {"widget": 1}})
    write_package(manager.packages_dir, "b",
                  class_weights={"forceps": 1.0}, standards={"forceps": 1})
    manager.activate("a")

    with pytest.raises(ModelManagerError):
        manager.activate("b")

    assert manager.last_switch_error
    assert manager.model_error is None          # A is still perfectly usable
    assert manager.state().ready is True
    assert manager.infer(object()).counts == {"widget": 1}


def test_strict_unload_does_not_claim_the_model_was_released(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"fail_unload": True})
    manager.activate("a")
    adapter = manager.active_adapter

    with pytest.raises(AdapterError):
        adapter.unload(strict=True)
    assert adapter.loaded is True
    assert RESIDENT.current == 1

    # Lenient teardown (shutdown/atexit) still gives up gracefully.
    adapter.unload(strict=False)
    assert adapter.loaded is False


# ── 13. retirement must be race-free ────────────────────────────────────────

def test_retire_blocks_new_work_before_teardown_begins(tmp_path):
    """The retired flag is the linearization point, not the end of teardown."""
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 1.0}, standards={"widget": 1},
                  adapter_options={"blocking": True, "counts": {"widget": 1}})
    ctl = control_for("a")
    manager.activate("a")
    adapter = manager.active_adapter

    started = threading.Event()
    finished = threading.Event()

    def infer():
        try:
            adapter.infer(object())
        finally:
            finished.set()

    threading.Thread(target=infer, daemon=True).start()
    assert ctl.entered.wait(COMPLETE_WAIT_SEC)
    started.set()

    retire_done = threading.Event()
    threading.Thread(target=lambda: (adapter.retire(), retire_done.set()),
                     daemon=True).start()

    # Retirement takes effect immediately, even though teardown is still
    # waiting for the running inference to finish.
    with pytest.raises(AdapterUnavailableError):
        adapter.infer(object())
    with pytest.raises(AdapterUnavailableError):
        adapter.load()
    assert not retire_done.wait(BLOCKED_CHECK_SEC)

    ctl.release.set()
    assert finished.wait(COMPLETE_WAIT_SEC)
    assert retire_done.wait(COMPLETE_WAIT_SEC)
    assert adapter.retired is True
    assert adapter.loaded is False
    assert RESIDENT.current == 0


def test_adapter_state_is_reportable(tmp_path):
    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "a",
                  class_weights={"widget": 1.0}, standards={"widget": 1})
    manager.activate("a")
    adapter = manager.active_adapter

    assert adapter.state == "loaded"
    adapter.retire()
    assert adapter.state == "retired"

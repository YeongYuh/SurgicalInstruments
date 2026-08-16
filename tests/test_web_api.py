"""HTTP-level checks: package API, per-package standards, the weight gate.

Runs against the real Flask app with a ModelManager rooted in a temp directory
and a FakeAdapter, so no model binary, camera, or scale is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.web as web_pkg
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import make_manager, write_package, write_raw_manifest


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Flask test client wired to a throwaway manager, scale, and history."""
    manager = make_manager(tmp_path)
    packages_dir = manager.packages_dir
    write_package(packages_dir, "ortho",
                  display_name="骨科 TKA",
                  class_weights={"widget": 10.0},
                  standards={"widget": 2},
                  adapter_options={"counts": {"widget": 2}})
    write_package(packages_dir, "obgyn",
                  display_name="婦產科",
                  class_weights={"forceps": 7.5},
                  standards={"forceps": 4},
                  adapter_options={"counts": {"forceps": 4}})
    write_package(packages_dir, "broken_load",
                  display_name="壞掉的套件",
                  adapter_options={"fail_load": True})
    write_raw_manifest(packages_dir, "demo", {
        "schema_version": 1, "id": "demo", "display_name": "Demo 範本",
        "adapter": "fake", "template": True, "model_file": "m.bin",
        "inventory": {"class_weights": "cw.json"},
    })
    manager.activate("ortho")

    monkeypatch.setattr(web_pkg, "model_manager", manager)
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0))
    monkeypatch.setattr(hist, "HISTORY_FILE", tmp_path / "history.json")
    web_pkg.reset_latest_state()

    web_pkg.app.config["TESTING"] = True
    with web_pkg.app.test_client() as client:
        yield client, manager


def _json(response):
    return json.loads(response.data.decode("utf-8"))


# ── package listing ─────────────────────────────────────────────────────────

def test_list_packages_reports_state_of_each(api):
    client, _ = api
    payload = _json(client.get("/api/model-packages"))

    assert payload["ok"] is True
    assert payload["active"] == "ortho"
    listing = {p["id"]: p for p in payload["packages"]}
    assert listing["ortho"]["active"] is True
    assert listing["ortho"]["display_name"] == "骨科 TKA"
    assert listing["demo"]["template"] is True
    assert listing["demo"]["activatable"] is False
    assert "fake" in payload["adapters"]


def test_get_active_package(api):
    client, _ = api
    payload = _json(client.get("/api/model-package"))
    assert payload["ok"] is True
    assert payload["package_id"] == "ortho"
    assert payload["generation"] == 1
    assert payload["loaded"] is True


# ── switching ───────────────────────────────────────────────────────────────

def test_switch_package_is_complete_when_the_response_arrives(api):
    client, manager = api

    payload = _json(client.post("/api/model-package", json={"id": "obgyn"}))

    assert payload["ok"] is True
    assert payload["package_id"] == "obgyn"
    # "ok" must mean usable, not merely accepted.
    assert payload["loaded"] is True
    assert manager.active_adapter.loaded is True
    assert manager.generation == 2


def test_switch_swaps_standards_and_class_weights_together(api):
    client, _ = api
    assert _json(client.get("/standards")) == {"widget": 2}
    assert _json(client.get("/class_weights")) == {"widget": 10.0}

    client.post("/api/model-package", json={"id": "obgyn"})

    assert _json(client.get("/standards")) == {"forceps": 4}
    assert _json(client.get("/class_weights")) == {"forceps": 7.5}


def test_switch_clears_results_from_the_previous_package(api):
    client, _ = api
    client.post("/api/model-package", json={"id": "obgyn"})

    status = _json(client.get("/status"))
    assert status["counts"] == {}
    assert status["weight"] is None
    assert status["active_package"] == "obgyn"
    assert status["active_package_name"] == "婦產科"


def test_failed_switch_leaves_the_previous_package_serving(api):
    client, manager = api

    response = client.post("/api/model-package", json={"id": "broken_load"})
    payload = _json(response)

    assert response.status_code == 400
    assert payload["ok"] is False
    assert "failed to activate package" in payload["error"]
    assert payload["active"] == "ortho"
    # Still fully operational on the old package.
    assert manager.state().package_id == "ortho"
    assert _json(client.get("/standards")) == {"widget": 2}
    assert _json(client.get("/api/model-package"))["package_id"] == "ortho"


def test_switching_to_a_template_is_refused(api):
    client, _ = api
    response = client.post("/api/model-package", json={"id": "demo"})
    assert response.status_code == 400
    assert "template" in _json(response)["error"]


def test_switch_requires_an_id(api):
    client, _ = api
    assert client.post("/api/model-package", json={}).status_code == 400
    assert client.post("/api/model-package", json={"id": ""}).status_code == 400


def test_switching_to_the_active_package_is_idempotent(api):
    client, manager = api
    payload = _json(client.post("/api/model-package", json={"id": "ortho"}))
    assert payload["status"] == "already_active"
    assert manager.generation == 1


# ── per-package standards editing ───────────────────────────────────────────

def test_standards_edits_are_scoped_to_the_active_package(api):
    client, _ = api

    assert _json(client.post("/standards", json={"widget": 9}))["ok"] is True
    assert _json(client.get("/standards")) == {"widget": 9}

    client.post("/api/model-package", json={"id": "obgyn"})
    assert _json(client.get("/standards")) == {"forceps": 4}   # ortho's edit did not leak

    client.post("/api/model-package", json={"id": "ortho"})
    assert _json(client.get("/standards")) == {"widget": 9}    # ...and survived


def test_standards_rejects_non_numeric_values(api):
    client, _ = api
    response = client.post("/standards", json={"widget": "two"})
    assert response.status_code == 400


def test_unit_weights_round_trip(api):
    client, _ = api
    assert _json(client.post("/unit_weights", json={"widget": 11.5}))["ok"] is True
    assert _json(client.get("/unit_weights"))["widget"] == 11.5


# ── weight gate over HTTP ───────────────────────────────────────────────────

def test_api_weight_reports_the_gate(api):
    client, _ = api
    payload = _json(client.get("/api/weight"))

    assert payload["ok"] is True
    assert payload["weight"] == 20.0
    assert payload["stable"] is True
    assert payload["fresh"] is True
    assert payload["ready_for_verification"] is True
    wv = payload["weight_verification"]
    assert wv["expected"] == 20.0        # 2 widgets × 10 g
    assert wv["passed"] is True
    assert wv["state"] == "passed"


def test_api_weight_is_pending_not_failed_while_unsettled(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(web_pkg, "scale_reader", MockScaleReader(20.0, stable=False))

    payload = _json(client.get("/api/weight"))

    assert payload["stable"] is False
    assert payload["ready_for_verification"] is False
    wv = payload["weight_verification"]
    assert wv["passed"] is None          # NOT False — this is not a failure
    assert wv["state"] == "stabilizing"


def test_api_weight_uses_the_active_packages_standard_weight(api):
    client, _ = api
    # ortho expects 20 g and the scale reads 20 g -> pass
    assert _json(client.get("/api/weight"))["weight_verification"]["passed"] is True

    client.post("/api/model-package", json={"id": "obgyn"})

    # obgyn expects 4 × 7.5 = 30 g, so the same 20 g reading now fails.
    wv = _json(client.get("/api/weight"))["weight_verification"]
    assert wv["expected"] == 30.0
    assert wv["passed"] is False


# ── inference through the platform ──────────────────────────────────────────

def test_upload_inference_publishes_package_aware_state_and_history(api, tmp_path):
    client, _ = api
    import cv2
    import numpy as np

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok

    response = client.post(
        "/upload",
        data={"image": (__import__("io").BytesIO(buf.tobytes()), "frame.jpg")},
        content_type="multipart/form-data",
    )
    payload = _json(response)

    assert payload["ok"] is True
    assert payload["counts"] == {"widget": 2}
    assert payload["package_id"] == "ortho"
    assert payload["package_display_name"] == "骨科 TKA"
    assert payload["weight_verification"]["passed"] is True

    records = hist.load_history()
    assert len(records) == 1
    assert records[0]["package_id"] == "ortho"
    assert records[0]["standards_snapshot"] == {"widget": 2}
    assert records[0]["class_weights_snapshot"] == {"widget": 10.0}


def test_report_export_is_package_aware(api):
    client, _ = api
    hist.append_record(hist.make_record(
        "webcam", {"forceps": 4}, 30.0, {"forceps": 4},
        package_id="obgyn", package_display_name="婦產科"))

    body = client.get("/report").data.decode("utf-8-sig")

    assert "obgyn" in body
    # Judged against the record's own standards (4 expected, 4 found), even
    # though the active package is orthopaedics and has no 'forceps' class.
    assert "正常" in body
    assert "多出" not in body


# ── SI-PLATFORM-002: model_ready must mean the model can actually run ───────

def test_status_reports_ready_when_the_model_is_loaded(api):
    client, _ = api
    status = _json(client.get("/status"))
    assert status["model_ready"] is True
    assert status["model_configured"] is True
    assert status["model_loading"] is False
    assert status["model_error"] is None


def test_status_is_not_ready_while_the_model_is_still_loading(api, tmp_path, monkeypatch):
    """A selected package is not the same as a usable model."""
    from conftest import make_manager, write_package as _write

    manager = make_manager(tmp_path / "slow")
    _write(manager.packages_dir, "slow",
           class_weights={"widget": 1.0}, standards={"widget": 1},
           adapter_options={"load_delay": 0.4})
    manager.bootstrap("slow", background=True)
    monkeypatch.setattr(web_pkg, "model_manager", manager)

    status = _json(client_of(api).get("/status"))
    assert status["model_configured"] is True
    assert status["model_ready"] is False
    assert status["model_loading"] is True

    assert manager.wait_until_ready(timeout=10.0) is True
    status = _json(client_of(api).get("/status"))
    assert status["model_ready"] is True
    assert status["model_loading"] is False


def test_status_exposes_a_startup_load_failure(api, tmp_path, monkeypatch):
    from conftest import make_manager, write_package as _write

    manager = make_manager(tmp_path / "broken")
    _write(manager.packages_dir, "boom",
           class_weights={"widget": 1.0}, standards={"widget": 1},
           adapter_options={"fail_load": True})
    manager.bootstrap("boom", background=False)
    monkeypatch.setattr(web_pkg, "model_manager", manager)

    status = _json(client_of(api).get("/status"))

    assert status["model_ready"] is False
    assert status["model_loading"] is False
    assert status["model_error"]
    info = _json(client_of(api).get("/api/model-package"))
    assert info["ready"] is False
    assert info["loaded"] is False


def client_of(api):
    return api[0]


def test_recognition_start_refuses_while_the_model_is_unusable(api, tmp_path, monkeypatch):
    """Better a clear 503 than a UI that says 辨識中 over a broken model."""
    from conftest import make_manager, write_package as _write

    manager = make_manager(tmp_path / "broken2")
    _write(manager.packages_dir, "boom",
           class_weights={"widget": 1.0}, standards={"widget": 1},
           adapter_options={"fail_load": True})
    manager.bootstrap("boom", background=False)
    monkeypatch.setattr(web_pkg, "model_manager", manager)

    class _Cam:
        def is_running(self):
            return True

        def get_status(self):
            return {"running": True, "recognition_running": False,
                    "inference_running": False, "recognition_generation": 0}

        def start_recognition(self):
            raise AssertionError("recognition must not start on an unusable model")

    monkeypatch.setattr(web_pkg, "camera_thread", _Cam(), raising=False)

    response = client_of(api).post("/camera/recognition/start")

    assert response.status_code == 503
    assert _json(response)["ok"] is False


def test_model_package_info_includes_compatibility_diagnostics(api):
    client, _ = api
    info = _json(client.get("/api/model-package"))
    assert "compatibility" in info
    diagnostics = info["compatibility"]
    # FakeAdapter publishes no class list, so the check is skipped rather than
    # guessed — and says so.
    assert diagnostics["has_model_class_list"] is False
    assert diagnostics["standards_missing_class_weight"] == []


def test_upload_result_carries_the_generation_it_ran_under(api):
    client, _ = api
    import io as _io

    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    payload = _json(client.post(
        "/upload",
        data={"image": (_io.BytesIO(buf.tobytes()), "f.jpg")},
        content_type="multipart/form-data",
    ))
    assert payload["model_generation"] == 1

    status = _json(client.get("/status"))
    assert status["result_model_generation"] == 1
    assert status["model_generation"] == 1

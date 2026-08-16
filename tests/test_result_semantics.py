"""no-result versus zero-detection.

"Nothing has been recognised yet" and "the model looked and found nothing" are
different states with the same empty counts.  Conflating them turns a tray that
was never scanned into a report saying every instrument is missing.
"""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest

import app.web as web_pkg
from app.runtime_result import (
    STATUS_CURRENT,
    STATUS_NO_RESULT,
    STATUS_STALE,
    has_current_result,
    is_empty_result,
)
from app.scale_reader import MockScaleReader
from app.web import history as hist
from conftest import FakeCamera, make_manager, write_package


def _json(response):
    return json.loads(response.data.decode("utf-8"))


@pytest.fixture
def api(tmp_path, monkeypatch, request):
    """`counts` marker controls what the fake model 'detects'."""
    marker = request.node.get_closest_marker("model_counts")
    counts = marker.args[0] if marker else {"widget": 2}

    manager = make_manager(tmp_path)
    write_package(manager.packages_dir, "ortho",
                  display_name="骨科", class_weights={"widget": 10.0},
                  standards={"widget": 2}, adapter_options={"counts": counts})
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


def _upload(client):
    ok, buf = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return _json(client.post(
        "/upload",
        data={"image": (io.BytesIO(buf.tobytes()), "f.jpg")},
        content_type="multipart/form-data",
    ))


# ── 12. nothing has run yet ─────────────────────────────────────────────────

def test_status_reports_no_result_before_any_inference(api):
    client, _ = api
    status = _json(client.get("/status"))

    assert status["has_result"] is False
    assert status["result_status"] == STATUS_NO_RESULT
    assert status["result_stale"] is False
    assert status["counts"] == {}
    assert status["weight_verification"] is None


def test_camera_result_reports_no_result_before_the_first_recognition(api, monkeypatch):
    """Camera on, nothing recognised yet — not "recognised, found zero"."""
    client, _ = api
    empty = {"timestamp": "", "counts": {}, "weight": None,
             "package_id": None, "package_display_name": None,
             "model_generation": 0, "annotated_b64": None}
    monkeypatch.setattr(web_pkg, "camera_thread", FakeCamera(result=empty),
                        raising=False)

    result = _json(client.get("/camera/result"))

    assert result["ok"] is True
    assert result["has_result"] is False
    assert result["result_status"] == STATUS_NO_RESULT
    assert result["counts"] == {}
    # No verdict may be attached to an inventory that never happened.
    assert result["weight_verification"] is None


# ── 13. the model looked and found nothing ──────────────────────────────────

@pytest.mark.model_counts({})
def test_zero_detection_result_is_a_real_result(api):
    client, _ = api
    payload = _upload(client)

    assert payload["ok"] is True
    assert payload["counts"] == {}

    status = _json(client.get("/status"))
    assert status["has_result"] is True
    assert status["result_status"] == STATUS_CURRENT
    assert status["counts"] == {}
    assert status["timestamp"]
    # A real measurement, so it does get a verdict.
    assert status["weight_verification"] is not None


@pytest.mark.model_counts({})
def test_camera_result_distinguishes_zero_detections(api, monkeypatch):
    client, manager = api
    state = manager.state()
    real = {"timestamp": "2026-05-21 10:00:00", "counts": {}, "weight": 20.0,
            "package_id": state.package_id, "package_display_name": "骨科",
            "model_generation": state.generation, "annotated_b64": None}
    monkeypatch.setattr(web_pkg, "camera_thread", FakeCamera(result=real),
                        raising=False)

    result = _json(client.get("/camera/result"))

    assert result["has_result"] is True
    assert result["result_status"] == STATUS_CURRENT
    assert result["counts"] == {}
    assert result["weight_verification"] is not None


# ── 14, 15 & 16. the BOM report ─────────────────────────────────────────────

def test_bom_report_refuses_when_nothing_has_been_recognised(api):
    """Not a report of "everything missing" — no report at all."""
    client, _ = api

    response = client.get("/bom_report")
    payload = _json(response)

    assert response.status_code == 409
    assert payload["ok"] is False
    assert payload["result_status"] == STATUS_NO_RESULT
    assert "尚無有效辨識結果" in payload["error"]


def test_bom_report_refuses_a_stale_result(api):
    client, manager = api
    _upload(client)
    assert client.get("/bom_report").status_code == 200

    manager.activate("obgyn")

    response = client.get("/bom_report")
    assert response.status_code == 409
    assert _json(response)["result_status"] == STATUS_STALE


@pytest.mark.model_counts({})
def test_bom_report_is_produced_for_a_genuine_zero_detection(api):
    """A tray that really was scanned and found empty DOES get a report.

    This is the case the 409 must not swallow: every expected instrument shows
    detected=0 because that is what the model saw, not because nothing ran.
    """
    client, _ = api
    _upload(client)

    response = client.get("/bom_report")
    body = response.data.decode("utf-8-sig")

    assert response.status_code == 200
    assert "widget" in body
    rows = [line.split(",") for line in body.splitlines() if line.startswith("widget")]
    assert rows, "the expected instrument is missing from the BOM"
    assert rows[0][1] == "2"      # standard
    assert rows[0][2] == "0"      # detected


def test_bom_report_works_for_a_normal_result(api):
    client, _ = api
    _upload(client)

    response = client.get("/bom_report")

    assert response.status_code == 200
    body = response.data.decode("utf-8-sig")
    assert "骨科" in body
    rows = [line.split(",") for line in body.splitlines() if line.startswith("widget")]
    assert rows[0][2] == "2"


# ── the helper ──────────────────────────────────────────────────────────────

class _State:
    configured = True
    package_id = "ortho"
    generation = 3


def test_is_empty_result_keys_on_the_timestamp_not_the_counts():
    """Empty counts are a legitimate result; an empty timestamp is not."""
    assert is_empty_result({"timestamp": "", "counts": {}}) is True
    assert is_empty_result({"timestamp": "", "counts": {"a": 1}}) is True
    assert is_empty_result({"timestamp": "t", "counts": {}}) is False
    assert is_empty_result({}) is True


def test_has_current_result_requires_publication_and_currency():
    assert has_current_result(
        {"timestamp": "t", "package_id": "ortho", "model_generation": 3}, _State()) is True
    assert has_current_result(
        {"timestamp": "", "package_id": "ortho", "model_generation": 3}, _State()) is False
    assert has_current_result(
        {"timestamp": "t", "package_id": "obgyn", "model_generation": 3}, _State()) is False
    assert has_current_result(
        {"timestamp": "t", "package_id": "ortho", "model_generation": 2}, _State()) is False

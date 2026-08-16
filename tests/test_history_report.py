"""History records carry package identity; reports read them back correctly."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from app.web import history as hist
from app.web import report as rpt


@pytest.fixture
def history_file(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    monkeypatch.setattr(hist, "HISTORY_FILE", path)
    return path


def _rows(csv_text):
    return list(csv.reader(io.StringIO(csv_text)))


# ── 16 & 18. new records carry package identity ─────────────────────────────

def test_record_carries_package_identity(history_file):
    record = hist.make_record(
        source="webcam",
        counts={"widget": 2},
        weight=20.0,
        standards_snapshot={"widget": 2},
        package_id="ortho_tka",
        package_display_name="骨科 TKA",
        model_identity={"adapter": "ultralytics", "backend": "onnx",
                        "model_file": "models/best.onnx"},
        class_weights_snapshot={"widget": 10.0},
        weight_verification={"ready": True, "passed": True},
    )
    hist.append_record(record)

    stored = hist.load_history()
    assert len(stored) == 1
    saved = stored[0]
    assert saved["package_id"] == "ortho_tka"
    assert saved["package_display_name"] == "骨科 TKA"
    assert saved["model"]["adapter"] == "ultralytics"
    assert saved["model"]["backend"] == "onnx"
    assert saved["class_weights_snapshot"] == {"widget": 10.0}
    assert saved["standards_snapshot"] == {"widget": 2}
    assert saved["weight_verification"]["passed"] is True


def test_record_without_package_still_builds(history_file):
    """The positional signature used before packages existed must keep working."""
    record = hist.make_record("upload", {"widget": 1}, 10.0, {"widget": 1})
    hist.append_record(record)
    assert hist.load_history()[0]["package_id"] is None


def test_history_is_capped(history_file, monkeypatch):
    monkeypatch.setattr(hist, "MAX_RECORDS", 5)
    for i in range(8):
        hist.append_record(hist.make_record("upload", {"w": i}, 1.0, {}))
    stored = hist.load_history()
    assert len(stored) == 5
    assert stored[-1]["counts"] == {"w": 7}


# ── 17. old records remain readable ─────────────────────────────────────────

def test_pre_package_history_is_readable(history_file):
    legacy = [{
        "id": "2026-05-21_19-59-28_23f2db",
        "timestamp": "2026-05-21 19:59:28",
        "source": "webcam",
        "counts": {"widget": 1},
        "weight": 10.0,
        "standards_snapshot": {"widget": 2},
    }]
    history_file.write_text(json.dumps(legacy), encoding="utf-8")

    stored = hist.load_history()

    assert len(stored) == 1
    assert stored[0].get("package_id") is None       # absent, not an error
    assert hist.record_standards(stored[0]) == {"widget": 2}


def test_corrupt_history_reads_as_empty(history_file):
    history_file.write_text("{ not json", encoding="utf-8")
    assert hist.load_history() == []


def test_record_standards_falls_back_only_without_a_snapshot():
    with_snapshot = {"standards_snapshot": {"widget": 2}}
    assert hist.record_standards(with_snapshot, {"forceps": 9}) == {"widget": 2}

    without = {"counts": {}}
    assert hist.record_standards(without, {"forceps": 9}) == {"forceps": 9}
    assert hist.record_standards(without) == {}


# ── 19 & 20. reports must not judge a record by another package's standards ──

def test_report_uses_each_records_own_standards():
    records = [
        {   # yesterday: obstetrics, 5 forceps expected, 5 found -> 正常
            "timestamp": "2026-05-20 10:00:00", "source": "webcam",
            "package_id": "obgyn", "package_display_name": "婦產科",
            "counts": {"forceps": 5}, "weight": 37.5,
            "standards_snapshot": {"forceps": 5},
        },
        {   # today: orthopaedics, 2 widgets expected, 2 found -> 正常
            "timestamp": "2026-05-21 10:00:00", "source": "webcam",
            "package_id": "ortho_tka", "package_display_name": "骨科",
            "counts": {"widget": 2}, "weight": 20.0,
            "standards_snapshot": {"widget": 2},
        },
    ]
    # The currently active package is orthopaedics and knows nothing of forceps.
    current_standards = {"widget": 2}

    rows = _rows(rpt.generate_csv(records, current_standards))
    header, body = rows[0], rows[1:]

    assert header[:4] == ["timestamp", "source", "package_id", "package_name"]
    by_class = {row[4]: row for row in body}

    # Had the report used the active standards, forceps would show standard=0
    # and status=多出 — a fabricated discrepancy in an unrelated department.
    assert by_class["forceps"][2] == "obgyn"
    assert by_class["forceps"][6] == "5"
    assert by_class["forceps"][7] == "正常"

    assert by_class["widget"][2] == "ortho_tka"
    assert by_class["widget"][6] == "2"
    assert by_class["widget"][7] == "正常"


def test_report_falls_back_to_current_standards_for_legacy_records():
    records = [{
        "timestamp": "2026-01-01 00:00:00", "source": "upload",
        "counts": {"widget": 1}, "weight": 10.0,
    }]
    rows = _rows(rpt.generate_csv(records, {"widget": 3}))
    assert rows[1][6] == "3"
    assert rows[1][7] == "缺少"


def test_report_handles_records_with_no_counts():
    records = [{"timestamp": "t", "source": "webcam", "counts": {}, "weight": None,
                "standards_snapshot": {"widget": 2}}]
    rows = _rows(rpt.generate_csv(records))
    assert rows[1][5] == "0"


def test_bom_report_includes_package_and_gate_state():
    csv_text = rpt.generate_bom_csv(
        counts={"widget": 2},
        standards={"widget": 2},
        unit_weights={"widget": 10.0},
        actual_weight=20.0,
        tolerance=0.5,
        timestamp="2026-05-21 10:00:00",
        package_id="ortho_tka",
        package_display_name="骨科 TKA",
        weight_verification={"ready": True, "passed": True, "expected": 20.0,
                             "difference": 0.0, "stable": True},
    )
    assert "ortho_tka" in csv_text
    assert "骨科 TKA" in csv_text
    assert "PASSED" in csv_text


def test_bom_report_does_not_claim_failure_while_unsettled():
    csv_text = rpt.generate_bom_csv(
        counts={"widget": 2},
        standards={"widget": 2},
        unit_weights={"widget": 10.0},
        actual_weight=18.0,
        tolerance=0.5,
        timestamp="2026-05-21 10:00:00",
        weight_verification={"ready": False, "passed": None, "expected": 20.0,
                             "difference": None, "stable": False,
                             "message": "重量穩定中"},
    )
    assert "FAILED" not in csv_text
    assert "尚未完成量測" in csv_text

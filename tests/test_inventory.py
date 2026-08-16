"""Inventory correctness: an instrument that was missed must be reported.

The kiosk table and the CSV export share these semantics (the UI mirrors
``compare_inventory`` in JavaScript), so this is the single place the rules are
pinned down.
"""

from __future__ import annotations

import csv
import io

import pytest

from app.inventory import (
    STATUS_EXTRA,
    STATUS_MISSING,
    STATUS_NORMAL,
    compare_inventory,
    status_for,
    summarise,
    tally,
)
from app.web import report as rpt


def _rows(csv_text):
    return list(csv.reader(io.StringIO(csv_text)))


def _by_class(csv_text):
    return {row[4]: row for row in _rows(csv_text)[1:] if row[4]}


# ── 26. a class the model missed entirely ───────────────────────────────────

def test_expected_class_absent_from_counts_is_reported_missing():
    """scissors detected, forceps not detected at all -> forceps 缺少.

    Iterating detections alone would drop forceps completely: the operator
    would see a tidy all-正常 table for a tray with an instrument still inside
    the patient.
    """
    rows = compare_inventory({"scissors": 2}, {"scissors": 2, "forceps": 1})

    assert [r.class_name for r in rows] == ["forceps", "scissors"]
    forceps = rows[0]
    assert forceps.detected == 0
    assert forceps.standard == 1
    assert forceps.status == STATUS_MISSING
    assert rows[1].status == STATUS_NORMAL


def test_nothing_detected_at_all_reports_every_expected_class():
    rows = compare_inventory({}, {"scissors": 2, "forceps": 1})
    assert [(r.class_name, r.detected, r.status) for r in rows] == [
        ("forceps", 0, STATUS_MISSING),
        ("scissors", 0, STATUS_MISSING),
    ]


def test_partial_count_is_missing_not_normal():
    rows = compare_inventory({"scissors": 1}, {"scissors": 2})
    assert rows[0].status == STATUS_MISSING


# ── 29. detected but not expected ───────────────────────────────────────────

def test_unexpected_class_is_reported_extra():
    rows = compare_inventory({"clamp": 1}, {"scissors": 1})
    statuses = {r.class_name: r.status for r in rows}
    assert statuses == {"clamp": STATUS_EXTRA, "scissors": STATUS_MISSING}


def test_class_with_no_expectation_and_no_detection_is_dropped():
    """Auto-registered classes this tray does not use must not fill the table."""
    rows = compare_inventory({"scissors": 1}, {"scissors": 1, "unused": 0})
    assert [r.class_name for r in rows] == ["scissors"]


def test_tally_counts_each_status():
    rows = compare_inventory({"a": 1, "c": 3}, {"a": 1, "b": 2, "c": 1})
    counts = tally(rows)
    assert (counts.normal, counts.missing, counts.extra) == (1, 1, 1)
    assert counts.total == 3


@pytest.mark.parametrize("detected,standard,expected", [
    (2, 2, STATUS_NORMAL),
    (0, 1, STATUS_MISSING),
    (3, 1, STATUS_EXTRA),
    (0, 0, STATUS_NORMAL),
])
def test_status_for(detected, standard, expected):
    assert status_for(detected, standard) == expected


def test_summarise_shape():
    payload = summarise({"a": 1}, {"a": 2})
    assert payload["missing"] == 1
    assert payload["rows"][0]["class_name"] == "a"
    assert payload["rows"][0]["detected"] == 1


def test_non_numeric_values_are_treated_as_zero():
    rows = compare_inventory({"a": "two"}, {"a": 1})
    assert rows[0].detected == 0
    assert rows[0].status == STATUS_MISSING


# ── 27 & 28. the CSV export must carry the same rows ────────────────────────

def test_history_csv_lists_zero_detected_expected_class():
    records = [{
        "timestamp": "2026-05-21 10:00:00", "source": "webcam",
        "package_id": "ortho", "package_display_name": "骨科",
        "counts": {"scissors": 2},
        "weight": 30.0,
        "standards_snapshot": {"scissors": 2, "forceps": 1},
    }]

    by_class = _by_class(rpt.generate_csv(records))

    assert "forceps" in by_class, "an instrument that was never detected vanished"
    assert by_class["forceps"][5] == "0"     # detected
    assert by_class["forceps"][6] == "1"     # standard
    assert by_class["forceps"][7] == STATUS_MISSING
    assert by_class["scissors"][7] == STATUS_NORMAL


def test_history_csv_with_empty_counts_is_not_a_single_blank_row():
    records = [{
        "timestamp": "t", "source": "webcam", "counts": {}, "weight": None,
        "standards_snapshot": {"forceps": 2, "scissors": 1},
    }]

    rows = _rows(rpt.generate_csv(records))[1:]

    assert len(rows) == 2
    assert {r[4] for r in rows} == {"forceps", "scissors"}
    assert all(r[5] == "0" for r in rows)
    assert all(r[7] == STATUS_MISSING for r in rows)


def test_history_csv_blank_row_only_when_there_is_nothing_at_all():
    records = [{"timestamp": "t", "source": "webcam", "counts": {},
                "weight": None, "standards_snapshot": {}}]
    rows = _rows(rpt.generate_csv(records))[1:]
    assert len(rows) == 1
    assert rows[0][4] == ""


def test_history_csv_reports_extra_against_the_record_snapshot():
    records = [{
        "timestamp": "t", "source": "webcam", "counts": {"clamp": 2},
        "weight": None, "standards_snapshot": {"clamp": 0, "forceps": 1},
    }]
    by_class = _by_class(rpt.generate_csv(records))
    assert by_class["clamp"][7] == STATUS_EXTRA
    assert by_class["forceps"][7] == STATUS_MISSING

"""Scale sample freshness/stability, and the weight verification gate."""

from __future__ import annotations

import pytest

from app.scale_reader import MockScaleReader
from app.scale_sample import (
    REASON_NO_READING,
    REASON_OK,
    REASON_STALE,
    REASON_UNSTABLE,
    REASON_WARMING_UP,
    ScaleSample,
    StabilityTracker,
    constant_sample,
)
from app.weight_verification import (
    REASON_NO_STANDARD_WEIGHT,
    REASON_NO_WEIGHT,
    REASON_WEIGHT_STALE,
    REASON_WEIGHT_UNSTABLE,
    compute_weight_verification,
    expected_weight,
)


class FakeClock:
    """Deterministic monotonic clock so stability is testable without sleeping."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def make_tracker(clock=None, **kwargs):
    clock = clock or FakeClock()
    defaults = dict(window_sec=1.5, range_grams=1.0, min_samples=3,
                    max_age_sec=2.0, min_coverage_ratio=0.5)
    defaults.update(kwargs)
    return StabilityTracker(monotonic=clock, wallclock=lambda: 0.0, **defaults), clock


def feed(tracker, clock, values, step=0.1):
    for value in values:
        tracker.add(value)
        clock.advance(step)


def feed_steady(tracker, clock, value, count=10, step=0.1):
    """Enough samples to satisfy both min_samples and the coverage requirement.

    At the production 10 Hz poll rate the default gate needs ~0.75 s of samples,
    i.e. about 8 readings — a burst of 3 is deliberately not enough.
    """
    feed(tracker, clock, [value] * count, step=step)


# ── 21. freshness ───────────────────────────────────────────────────────────

def test_no_reading_yet():
    tracker, _ = make_tracker()
    sample = tracker.snapshot()
    assert sample.value is None
    assert sample.fresh is False
    assert sample.stable is False
    assert sample.reason == REASON_NO_READING


def test_sample_reports_value_age_and_freshness():
    tracker, clock = make_tracker()
    tracker.add(50.0)
    clock.advance(0.5)

    sample = tracker.snapshot()

    assert sample.value == 50.0
    assert sample.age_sec == pytest.approx(0.5)
    assert sample.fresh is True


# ── 22. stable ──────────────────────────────────────────────────────────────

def test_settled_readings_become_stable():
    tracker, clock = make_tracker()
    feed(tracker, clock, [50.0, 50.1, 49.9, 50.0, 50.05] * 2)

    sample = tracker.snapshot()

    assert sample.stable is True
    assert sample.fresh is True
    assert sample.reason == REASON_OK
    assert sample.spread == pytest.approx(0.2, abs=1e-6)
    assert sample.sample_count >= 3


def test_stability_needs_enough_samples():
    tracker, clock = make_tracker()
    feed(tracker, clock, [50.0, 50.0])          # only 2, min is 3

    sample = tracker.snapshot()

    assert sample.stable is False
    assert sample.reason == REASON_WARMING_UP


def test_stability_needs_the_window_to_be_covered():
    """Three samples taken in one burst are not 1.5 s of steady weighing."""
    tracker, clock = make_tracker()
    feed(tracker, clock, [50.0, 50.0, 50.0], step=0.001)

    sample = tracker.snapshot()

    assert sample.stable is False
    assert sample.reason == REASON_WARMING_UP
    assert sample.coverage_sec < 0.75


# ── 23. unstable ────────────────────────────────────────────────────────────

def test_moving_readings_are_unstable():
    tracker, clock = make_tracker()
    feed(tracker, clock, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 44.0, 48.0, 50.0])

    sample = tracker.snapshot()

    assert sample.stable is False
    assert sample.fresh is True
    assert sample.reason == REASON_UNSTABLE
    assert sample.spread > 1.0


def test_becomes_stable_once_the_moving_samples_age_out():
    tracker, clock = make_tracker()
    feed(tracker, clock, [10.0, 25.0, 40.0], step=0.15)   # still being loaded
    assert tracker.snapshot().stable is False

    # Settled on the pan for long enough that the moving samples fall out of the
    # 1.5 s window entirely.
    feed(tracker, clock, [50.0] * 11 + [49.95], step=0.15)

    sample = tracker.snapshot()
    assert sample.stable is True
    assert sample.value == pytest.approx(49.95)


# ── 24. stale ───────────────────────────────────────────────────────────────

def test_reading_goes_stale_when_the_scale_stops_reporting():
    """The cable is pulled: the cached number must stop pretending to be live."""
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 50.0)
    assert tracker.snapshot().stable is True

    clock.advance(5.0)                       # no new samples arrive

    sample = tracker.snapshot()
    assert sample.value == 50.0              # last known value is still reported
    assert sample.fresh is False             # ...but explicitly not current
    assert sample.stable is False
    assert sample.reason == REASON_STALE
    assert sample.age_sec >= 5.0


def test_reset_clears_history():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 50.0)
    tracker.reset()
    assert tracker.snapshot().reason == REASON_NO_READING


# ── 25. verification is pending, never FAIL, while unsettled ────────────────

def test_verification_pending_when_weight_is_unstable():
    tracker, clock = make_tracker()
    feed(tracker, clock, [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 44.0, 48.0])
    sample = tracker.snapshot()

    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, sample, 0.5)

    assert wv["ready"] is False
    assert wv["passed"] is None          # explicitly NOT False
    assert wv["reason"] == REASON_WEIGHT_UNSTABLE
    assert wv["state"] == "stabilizing"
    assert wv["expected"] == 20.0


def test_verification_pending_when_weight_is_stale():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 20.0)
    clock.advance(10.0)
    sample = tracker.snapshot()

    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, sample, 0.5)

    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["reason"] == REASON_WEIGHT_STALE
    assert wv["state"] == "waiting_weight"


def test_verification_pending_without_any_reading():
    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, None, 0.5)
    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["reason"] == REASON_NO_WEIGHT


def test_verification_pending_when_no_standard_weight_is_configured():
    wv = compute_weight_verification({}, {}, constant_sample(20.0), 0.5)
    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["reason"] == REASON_NO_STANDARD_WEIGHT
    assert wv["state"] == "no_standard_weight"


# ── 26. a verdict only once the reading has settled ─────────────────────────

def test_verification_passes_only_when_stable():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 20.0)
    sample = tracker.snapshot()
    assert sample.stable is True

    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, sample, 0.5)

    assert wv["ready"] is True
    assert wv["passed"] is True
    assert wv["expected"] == 20.0
    assert wv["actual"] == 20.0
    assert wv["difference"] == 0.0
    assert wv["state"] == "passed"


def test_verification_fails_when_stable_and_out_of_tolerance():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 15.0)
    sample = tracker.snapshot()

    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, sample, 0.5)

    assert wv["ready"] is True
    assert wv["passed"] is False
    assert wv["difference"] == 5.0
    assert wv["state"] == "failed"


def test_expected_weight_ignores_classes_without_a_unit_weight():
    """A missing unit weight contributes 0 g — it must not crash the盤點."""
    assert expected_weight({"widget": 2, "unknown": 3}, {"widget": 10.0}) == 20.0
    assert expected_weight({"widget": 0}, {"widget": 10.0}) == 0.0
    assert expected_weight({"widget": "bad"}, {"widget": 10.0}) == 0.0


def test_bare_float_is_treated_as_a_current_reading():
    """Legacy callers passing a plain number still get a verdict."""
    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0}, 20.0, 0.5)
    assert wv["ready"] is True
    assert wv["passed"] is True


# ── 27. the mock scale ──────────────────────────────────────────────────────

def test_mock_scale_reports_a_stable_fresh_sample():
    """A constant really is settled — that is what makes mock mode useful."""
    reader = MockScaleReader(mock_weight=1520.35)

    sample = reader.get_latest_sample()

    assert isinstance(sample, ScaleSample)
    assert sample.value == 1520.35
    assert sample.fresh is True
    assert sample.stable is True
    assert sample.reason == REASON_OK
    assert reader.get_latest_weight() == 1520.35


def test_mock_scale_can_simulate_an_unsettled_reading():
    reader = MockScaleReader(mock_weight=100.0, stable=False)
    sample = reader.get_latest_sample()
    assert sample.stable is False

    wv = compute_weight_verification({"widget": 10}, {"widget": 10.0}, sample, 0.5)
    assert wv["ready"] is False
    assert wv["passed"] is None


def test_mock_scale_drives_a_full_pass(tmp_path):
    reader = MockScaleReader(mock_weight=20.0)
    wv = compute_weight_verification({"widget": 2}, {"widget": 10.0},
                                     reader.get_latest_sample(), 0.5)
    assert wv["ready"] is True
    assert wv["passed"] is True


def test_sample_to_dict_is_json_friendly():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 50.0)
    payload = tracker.snapshot().to_dict()

    import json
    json.dumps(payload)     # must not raise
    assert set(payload) >= {"value", "fresh", "stable", "age_sec", "reason"}

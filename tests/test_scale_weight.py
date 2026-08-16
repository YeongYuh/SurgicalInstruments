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


# ── SI-PLATFORM-002: coverage must measure samples, not wall time ───────────

def test_time_passing_without_new_samples_does_not_create_coverage():
    """Three readings in one burst are not a second of steady weighing.

    Measuring coverage as (now - first sample) let the clock alone satisfy the
    window: the scale went quiet after a burst and the reading "settled" purely
    because time passed.
    """
    tracker, clock = make_tracker()
    tracker.add(50.0); clock.advance(0.001)
    tracker.add(50.0); clock.advance(0.001)
    tracker.add(50.0)

    # Enough time for the old wall-clock formula to claim full coverage, but
    # still well inside max_sample_age so the reading is not merely stale.
    clock.advance(0.9)

    sample = tracker.snapshot()

    assert sample.fresh is True                  # not a staleness problem
    assert sample.coverage_sec == pytest.approx(0.002, abs=1e-6)
    assert sample.stable is False
    assert sample.reason == REASON_WARMING_UP


def test_coverage_is_the_span_of_the_samples_themselves():
    tracker, clock = make_tracker()
    feed(tracker, clock, [50.0] * 10, step=0.1)   # 9 intervals -> 0.9 s span

    sample = tracker.snapshot()

    assert sample.coverage_sec == pytest.approx(0.9, abs=1e-6)
    assert sample.stable is True


def test_single_sample_has_no_coverage():
    tracker, clock = make_tracker()
    tracker.add(50.0)
    clock.advance(1.0)
    sample = tracker.snapshot()
    assert sample.coverage_sec == 0.0
    assert sample.stable is False


def test_burst_then_real_sampling_becomes_stable():
    tracker, clock = make_tracker()
    tracker.add(50.0); clock.advance(0.001)
    tracker.add(50.0); clock.advance(0.001)
    tracker.add(50.0); clock.advance(0.9)
    assert tracker.snapshot().stable is False

    feed_steady(tracker, clock, 50.0)             # genuine 10 Hz sampling

    assert tracker.snapshot().stable is True


# ── SI-PLATFORM-002: a missing class weight can never yield PASS/FAIL ───────

def test_missing_class_weight_blocks_any_verdict():
    """standards A=2 (100 g) + B=1 with no weight for B.

    Treating the missing weight as 0 g would expect 200 g, and a tray holding
    only the A instruments would weigh exactly that and be declared complete —
    with instrument B still missing.
    """
    wv = compute_weight_verification(
        {"A": 2, "B": 1}, {"A": 100.0}, constant_sample(200.0), 0.5)

    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["reason"] == "missing_class_weights"
    assert wv["missing_class_weights"] == ["B"]
    assert wv["state"] == "no_standard_weight"
    assert "B" in wv["message"]


def test_missing_class_weight_blocks_even_a_perfectly_stable_reading():
    tracker, clock = make_tracker()
    feed_steady(tracker, clock, 200.0)
    sample = tracker.snapshot()
    assert sample.stable is True

    wv = compute_weight_verification({"A": 2, "B": 1}, {"A": 100.0}, sample, 0.5)

    assert wv["ready"] is False
    assert wv["passed"] is None


def test_classes_with_zero_standard_do_not_need_a_weight():
    """Only instruments that are actually expected need a unit weight."""
    wv = compute_weight_verification(
        {"A": 2, "B": 0}, {"A": 100.0}, constant_sample(200.0), 0.5)
    assert wv["missing_class_weights"] == []
    assert wv["ready"] is True
    assert wv["passed"] is True


@pytest.mark.parametrize("bad_weight", [
    0, 0.0, -5.0, float("nan"), float("inf"), float("-inf"), "heavy", None, True,
])
def test_invalid_class_weights_are_treated_as_missing(bad_weight):
    wv = compute_weight_verification({"A": 1}, {"A": bad_weight},
                                     constant_sample(10.0), 0.5)
    assert wv["ready"] is False
    assert wv["passed"] is None
    assert wv["missing_class_weights"] == ["A"]


def test_expected_weight_detail_reports_total_and_missing():
    from app.weight_verification import expected_weight_detail

    total, missing = expected_weight_detail({"A": 2, "B": 1, "C": 0},
                                            {"A": 10.0, "C": 3.0})
    assert total == 20.0
    assert missing == ["B"]


def test_fully_configured_package_is_unaffected():
    wv = compute_weight_verification({"A": 2, "B": 1}, {"A": 100.0, "B": 5.0},
                                     constant_sample(205.0), 0.5)
    assert wv["missing_class_weights"] == []
    assert wv["expected"] == 205.0
    assert wv["ready"] is True
    assert wv["passed"] is True


# ── SI-PLATFORM-003: the 2.5 s window sized for a ~2.2 Hz scale ─────────────

def test_production_default_window_is_sized_for_the_measured_scale_rate():
    """The scale on this unit was measured emitting ~2.2 Hz, not 10 Hz.

    A 1.5 s window needed 3 samples spanning 0.75 s, which at that rate is
    exactly the 3-sample minimum with no margin: one dropped line and the
    reading would never settle.
    """
    import app.config as config

    assert config.SCALE_STABLE_WINDOW_SEC == 2.5
    assert config.SCALE_STABLE_MIN_SAMPLES == 3          # not weakened to 1 or 2
    assert config.SCALE_STABLE_MIN_COVERAGE_RATIO == 0.5

    required_coverage = (config.SCALE_STABLE_WINDOW_SEC
                         * config.SCALE_STABLE_MIN_COVERAGE_RATIO)
    measured_rate = 2.2
    samples_needed = required_coverage * measured_rate + 1
    assert samples_needed >= 3.5, "no margin above the minimum sample count"
    assert samples_needed <= config.SCALE_STABLE_WINDOW_SEC * measured_rate


@pytest.mark.parametrize("rate_hz", [2.0, 2.2, 2.5, 10.0])
def test_scale_settles_at_the_rates_this_hardware_produces(rate_hz):
    """A steady reading must actually reach stable at each plausible rate."""
    tracker, clock = make_tracker(window_sec=2.5)
    step = 1.0 / rate_hz

    elapsed = 0.0
    while elapsed < 6.0 and not tracker.snapshot().stable:
        tracker.add(50.0)
        clock.advance(step)
        elapsed += step

    sample = tracker.snapshot()
    assert sample.stable is True, "never settled at %.1f Hz" % rate_hz
    assert elapsed <= 2.5, "took %.2fs to settle at %.1f Hz" % (elapsed, rate_hz)


def test_slow_scale_settles_within_a_usable_time():
    """At the measured 2.2 Hz the operator should not wait more than ~2 s."""
    tracker, clock = make_tracker(window_sec=2.5)
    step = 1.0 / 2.2
    elapsed = 0.0
    while not tracker.snapshot().stable:
        tracker.add(120.0)
        clock.advance(step)
        elapsed += step
        assert elapsed < 5.0, "did not settle"

    assert 1.0 <= elapsed <= 2.5
    assert tracker.snapshot().sample_count >= 3


def test_wider_window_still_rejects_a_burst_followed_by_silence():
    tracker, clock = make_tracker(window_sec=2.5)
    for _ in range(6):
        tracker.add(50.0)
        clock.advance(0.001)
    clock.advance(1.5)          # still fresh, but no new samples

    sample = tracker.snapshot()

    assert sample.fresh is True
    assert sample.stable is False
    assert sample.reason == REASON_WARMING_UP
    assert sample.coverage_sec < 0.05


def test_wider_window_still_goes_stale(tmp_path=None):
    tracker, clock = make_tracker(window_sec=2.5)
    feed(tracker, clock, [50.0] * 8, step=0.45)
    assert tracker.snapshot().stable is True

    clock.advance(5.0)

    sample = tracker.snapshot()
    assert sample.fresh is False
    assert sample.stable is False
    assert sample.reason == REASON_STALE


def test_serial_reader_uses_the_production_window_by_default():
    from app.scale_reader import SerialScaleReader

    reader = SerialScaleReader(port="/dev/null")
    assert reader._tracker.window_sec == 2.5
    assert reader._tracker.min_samples == 3

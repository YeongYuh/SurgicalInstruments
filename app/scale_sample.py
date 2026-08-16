"""Scale sample metadata: timestamps, freshness, stability.

A bare float is not enough to verify a weight.  "1520.3 g" read four seconds
ago while the operator is still putting instruments on the tray is not a
measurement — treating it as one produces a confident FAIL that is really just
an unfinished weighing.

This module keeps the recent accepted readings with their timestamps and
answers two questions:

    fresh   — did we hear from the scale recently enough to trust the number?
    stable  — has the reading settled (small spread over a real time window)?

Both are pure functions of the buffer plus "now", so the whole thing is
testable without any hardware.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

# Snapshot reasons, in increasing order of goodness.
REASON_NO_READING = "no_reading"   # nothing has ever been read
REASON_STALE = "stale"             # last reading is too old to trust
REASON_WARMING_UP = "warming_up"   # too few samples / window not covered yet
REASON_UNSTABLE = "unstable"       # readings are still moving
REASON_OK = "ok"                   # settled


@dataclass
class ScaleSample:
    """One point-in-time view of the scale."""

    value: Optional[float] = None
    timestamp: Optional[float] = None     # wall clock (time.time) of last reading
    monotonic: Optional[float] = None     # monotonic clock of last reading
    age_sec: Optional[float] = None
    fresh: bool = False
    stable: bool = False
    sample_count: int = 0                 # samples inside the stability window
    spread: Optional[float] = None        # max-min inside the window
    coverage_sec: float = 0.0             # time span covered by the window
    window_sec: float = 0.0
    reason: str = REASON_NO_READING

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "age_sec": (round(self.age_sec, 3) if self.age_sec is not None else None),
            "fresh": self.fresh,
            "stable": self.stable,
            "sample_count": self.sample_count,
            "spread": (round(self.spread, 4) if self.spread is not None else None),
            "coverage_sec": round(self.coverage_sec, 3),
            "window_sec": self.window_sec,
            "reason": self.reason,
        }


class StabilityTracker:
    """Rolling buffer of accepted readings + freshness/stability evaluation.

    The clocks are injectable so tests can drive time deterministically.
    """

    def __init__(
        self,
        window_sec: float = 2.5,
        range_grams: float = 1.0,
        min_samples: int = 3,
        max_age_sec: float = 2.0,
        min_coverage_ratio: float = 0.5,
        maxlen: int = 512,
        monotonic: Callable[[], float] = time.monotonic,
        wallclock: Callable[[], float] = time.time,
    ) -> None:
        self.window_sec = max(0.0, float(window_sec))
        self.range_grams = max(0.0, float(range_grams))
        self.min_samples = max(1, int(min_samples))
        self.max_age_sec = max(0.0, float(max_age_sec))
        self.min_coverage_ratio = min(1.0, max(0.0, float(min_coverage_ratio)))
        self._monotonic = monotonic
        self._wallclock = wallclock
        self._lock = threading.Lock()
        # (monotonic, wallclock, value)
        self._samples: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)

    # ── writes ────────────────────────────────────────────────────────────

    def add(self, value: float, mono: Optional[float] = None,
            wall: Optional[float] = None) -> None:
        """Record one accepted (already filtered) reading."""
        if value is None:
            return
        m = self._monotonic() if mono is None else float(mono)
        w = self._wallclock() if wall is None else float(wall)
        with self._lock:
            self._samples.append((m, w, float(value)))

    def reset(self) -> None:
        """Drop all history — used when the scale connection is lost."""
        with self._lock:
            self._samples.clear()

    # ── reads ─────────────────────────────────────────────────────────────

    @property
    def latest_value(self) -> Optional[float]:
        with self._lock:
            return self._samples[-1][2] if self._samples else None

    def snapshot(self, now: Optional[float] = None) -> ScaleSample:
        """Evaluate freshness and stability as of ``now`` (monotonic seconds)."""
        current = self._monotonic() if now is None else float(now)
        with self._lock:
            if not self._samples:
                return ScaleSample(window_sec=self.window_sec, reason=REASON_NO_READING)
            last_mono, last_wall, last_value = self._samples[-1]
            cutoff = current - self.window_sec
            window = [s for s in self._samples if s[0] >= cutoff]

        age = current - last_mono
        fresh = age <= self.max_age_sec

        values = [s[2] for s in window]
        count = len(values)
        spread = (max(values) - min(values)) if values else None
        # Coverage is the span the SAMPLES cover, not how much wall time has
        # passed.  Using wall time would let three readings taken in a single
        # millisecond "become" a stable second simply because the clock moved
        # on after the scale went quiet.
        coverage = (window[-1][0] - window[0][0]) if count >= 2 else 0.0
        required_coverage = self.window_sec * self.min_coverage_ratio

        if not fresh:
            reason = REASON_STALE
            stable = False
        elif count < self.min_samples or coverage < required_coverage:
            reason = REASON_WARMING_UP
            stable = False
        elif spread is not None and spread > self.range_grams:
            reason = REASON_UNSTABLE
            stable = False
        else:
            reason = REASON_OK
            stable = True

        return ScaleSample(
            value=last_value,
            timestamp=last_wall,
            monotonic=last_mono,
            age_sec=age,
            fresh=fresh,
            stable=stable,
            sample_count=count,
            spread=spread,
            coverage_sec=coverage,
            window_sec=self.window_sec,
            reason=reason,
        )


def constant_sample(value: Optional[float], *, window_sec: float = 0.0,
                    wallclock: Callable[[], float] = time.time) -> ScaleSample:
    """A perfectly steady reading — used by the mock scale.

    A fixed value genuinely is fresh and stable, which is what makes the mock
    useful for exercising the PASS/FAIL path without hardware.
    """
    if value is None:
        return ScaleSample(window_sec=window_sec, reason=REASON_NO_READING)
    return ScaleSample(
        value=float(value),
        timestamp=wallclock(),
        monotonic=time.monotonic(),
        age_sec=0.0,
        fresh=True,
        stable=True,
        sample_count=1,
        spread=0.0,
        coverage_sec=window_sec,
        window_sec=window_sec,
        reason=REASON_OK,
    )

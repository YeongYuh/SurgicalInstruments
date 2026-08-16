"""Weight verification with a stability gate.

Standard weight comes from the ACTIVE PACKAGE's ``class_weight.json`` (grams
per instrument) multiplied by the operator-editable standard quantities:

    expected = Σ ( standards[cls] × class_weights[cls] )

A PASS/FAIL verdict is only issued once the scale reading is both *fresh* and
*stable*.  Before then the result is "pending" (``passed = None``), never FAIL —
an unsettled scale is a measurement that has not finished, not a failed
inventory.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from app.scale_sample import (
    REASON_NO_READING,
    REASON_OK,
    REASON_STALE,
    REASON_UNSTABLE,
    REASON_WARMING_UP,
    ScaleSample,
)

logger = logging.getLogger(__name__)

# reason codes
REASON_READY = "ok"
REASON_NO_WEIGHT = "no_weight"
REASON_WEIGHT_STALE = "weight_stale"
REASON_WEIGHT_UNSTABLE = "weight_unstable"
REASON_NO_STANDARD_WEIGHT = "no_standard_weight"
REASON_MISSING_CLASS_WEIGHTS = "missing_class_weights"

# UI state tokens
STATE_WAITING = "waiting_weight"
STATE_STABILIZING = "stabilizing"
STATE_NO_STANDARD = "no_standard_weight"
STATE_PASSED = "passed"
STATE_FAILED = "failed"

_MESSAGES = {
    STATE_WAITING: "等待重量讀值",
    STATE_STABILIZING: "重量穩定中",
    STATE_NO_STANDARD: "尚未設定標準重量",
    STATE_PASSED: "重量在容許範圍內",
    STATE_FAILED: "重量超出容許範圍",
}


def _usable_weight(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def expected_weight_detail(standards: Mapping[str, Any],
                           class_weights: Mapping[str, Any]) -> Tuple[float, List[str]]:
    """Σ standards × class_weights, plus the classes that have no usable weight.

    A missing unit weight is NOT treated as 0 g.  Doing so silently lowers the
    expected total, which is exactly the direction that lets an incomplete tray
    pass: standards A=2 (100 g each) and B=1 with B's weight missing would
    expect 200 g, and a tray holding only the A instruments would be declared
    correct.  The caller must refuse to issue a verdict while anything is
    missing.
    """
    total = 0.0
    missing: List[str] = []
    for cls, std_qty in standards.items():
        try:
            qty = int(std_qty or 0)
        except (TypeError, ValueError):
            missing.append(str(cls))
            continue
        if qty <= 0:
            continue
        weight = class_weights.get(cls)
        if not _usable_weight(weight):
            logger.warning(
                "[weight_verification] class '%s' (std=%d) has no usable class weight (%r)",
                cls, qty, weight)
            missing.append(str(cls))
            continue
        total += qty * float(weight)
    return total, sorted(set(missing))


def expected_weight(standards: Mapping[str, Any],
                    class_weights: Mapping[str, Any]) -> float:
    """Σ standards × class_weights, ignoring classes with no usable weight.

    Only meaningful together with the missing-class list; use
    ``expected_weight_detail`` when the number is going to be compared against
    a real reading.
    """
    total, _missing = expected_weight_detail(standards, class_weights)
    return total


def _coerce_sample(sample: Union[ScaleSample, float, int, None]) -> ScaleSample:
    if isinstance(sample, ScaleSample):
        return sample
    if sample is None:
        return ScaleSample(reason=REASON_NO_READING)
    # A bare number carries no timestamp; the caller is asserting it is current.
    return ScaleSample(
        value=float(sample), age_sec=0.0, fresh=True, stable=True,
        sample_count=1, spread=0.0, reason=REASON_OK,
    )


def compute_weight_verification(
    standards: Mapping[str, Any],
    class_weights: Mapping[str, Any],
    sample: Union[ScaleSample, float, int, None],
    tolerance: float,
) -> Dict[str, Any]:
    """Compare a scale sample against the expected weight for ``standards``.

    ``passed`` is ``None`` whenever ``ready`` is False.  Consumers must check
    ``ready`` before rendering a verdict.
    """
    sample = _coerce_sample(sample)
    expected, missing_class_weights = expected_weight_detail(standards, class_weights)

    result: Dict[str, Any] = {
        "ready": False,
        "passed": None,
        "expected": round(expected, 4),
        "actual": (round(sample.value, 4) if sample.value is not None else None),
        "difference": None,
        "tolerance": tolerance,
        "stable": bool(sample.stable),
        "fresh": bool(sample.fresh),
        "sample_age": (round(sample.age_sec, 3) if sample.age_sec is not None else None),
        "sample_reason": sample.reason,
        "sample_count": sample.sample_count,
        "spread": (round(sample.spread, 4) if sample.spread is not None else None),
        "missing_class_weights": missing_class_weights,
        "reason": REASON_NO_WEIGHT,
        "state": STATE_WAITING,
        "message": _MESSAGES[STATE_WAITING],
    }

    # 1. Is the standard weight fully defined?  Checked first because it is a
    #    configuration fault, not a transient measurement state: the expected
    #    total is wrong (too low) for as long as any class weight is missing,
    #    so no reading of any quality may be turned into a verdict.
    if missing_class_weights:
        result["reason"] = REASON_MISSING_CLASS_WEIGHTS
        result["state"] = STATE_NO_STANDARD
        result["message"] = "標準重量未完整設定：缺少 %s 的單重" % "、".join(
            missing_class_weights[:5]) + ("…" if len(missing_class_weights) > 5 else "")
        return result

    # 2. Do we have a usable measurement at all?
    if sample.value is None or sample.reason == REASON_NO_READING:
        return result
    if not sample.fresh or sample.reason == REASON_STALE:
        result["reason"] = REASON_WEIGHT_STALE
        result["state"] = STATE_WAITING
        result["message"] = _MESSAGES[STATE_WAITING]
        return result
    if not sample.stable or sample.reason in (REASON_WARMING_UP, REASON_UNSTABLE):
        result["reason"] = REASON_WEIGHT_UNSTABLE
        result["state"] = STATE_STABILIZING
        result["message"] = _MESSAGES[STATE_STABILIZING]
        return result

    # 3. Is there anything to compare against?
    if expected <= 0.0:
        result["reason"] = REASON_NO_STANDARD_WEIGHT
        result["state"] = STATE_NO_STANDARD
        result["message"] = _MESSAGES[STATE_NO_STANDARD]
        return result

    # 4. Stable, fresh, and configured — issue a verdict.
    difference = abs(float(sample.value) - expected)
    passed = difference <= float(tolerance)
    state = STATE_PASSED if passed else STATE_FAILED
    result.update({
        "ready": True,
        "passed": passed,
        "difference": round(difference, 4),
        "reason": REASON_READY,
        "state": state,
        "message": _MESSAGES[state],
    })
    return result

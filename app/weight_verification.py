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
from typing import Any, Dict, Mapping, Optional, Union

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


def expected_weight(standards: Mapping[str, Any],
                    class_weights: Mapping[str, Any]) -> float:
    """Σ standards × class_weights.

    Classes present in standards but missing from class_weights contribute 0 g
    and are logged — a missing unit weight must never crash a盤點.
    """
    total = 0.0
    for cls, std_qty in standards.items():
        try:
            qty = int(std_qty or 0)
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue
        if cls not in class_weights:
            logger.warning(
                "[weight_verification] class '%s' (std=%d) not in class weights — 0 g",
                cls, qty)
            continue
        try:
            total += qty * float(class_weights[cls])
        except (TypeError, ValueError):
            logger.warning("[weight_verification] non-numeric class weight for '%s'", cls)
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
    expected = expected_weight(standards, class_weights)

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
        "reason": REASON_NO_WEIGHT,
        "state": STATE_WAITING,
        "message": _MESSAGES[STATE_WAITING],
    }

    # 1. Do we have a usable measurement at all?
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

    # 2. Is there anything to compare against?
    if expected <= 0.0:
        result["reason"] = REASON_NO_STANDARD_WEIGHT
        result["state"] = STATE_NO_STANDARD
        result["message"] = _MESSAGES[STATE_NO_STANDARD]
        return result

    # 3. Stable, fresh, and configured — issue a verdict.
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

from typing import Optional


def check_weight(
    actual_weight: Optional[float],
    expected_weight: float,
    tolerance: float,
) -> dict:
    if actual_weight is None:
        return {
            "passed": False,
            "actual_weight": None,
            "expected_weight": expected_weight,
            "difference": None,
            "tolerance": tolerance,
            "message": "Failed to read scale weight.",
        }

    difference = abs(actual_weight - expected_weight)
    passed = difference <= tolerance
    message = "Weight within tolerance." if passed else "Weight out of tolerance!"

    return {
        "passed": passed,
        "actual_weight": actual_weight,
        "expected_weight": expected_weight,
        "difference": difference,
        "tolerance": tolerance,
        "message": message,
    }

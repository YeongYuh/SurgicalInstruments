"""Strict value validation for operator-supplied inventory numbers.

The package manifest validator already refuses fractional quantities and
non-finite numbers.  Runtime edits go through the same rules here, because a
value that would be rejected in a manifest must not be quietly accepted (and
silently rounded) just because it arrived over HTTP.

Nothing here clamps or truncates.  ``1.7`` instruments is not ``1``; it is a
mistake, and the operator has to see it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple


class ValueValidationError(ValueError):
    """One bad field in an operator-supplied mapping."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__("invalid value for '%s': %r — %s" % (field, value, reason))


def _as_number(field: str, value: Any) -> float:
    # bool is an int subclass; True would otherwise silently become 1.
    if isinstance(value, bool):
        raise ValueValidationError(field, value, "must be a number, not a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueValidationError(field, value, "must not be empty")
        try:
            number = float(text)
        except ValueError:
            raise ValueValidationError(field, value, "must be a number")
    else:
        raise ValueValidationError(field, value, "must be a number")
    if not math.isfinite(number):
        raise ValueValidationError(field, value, "must be a finite number")
    return number


def parse_standard_quantity(field: str, value: Any) -> int:
    """Expected quantity: a whole, non-negative count."""
    number = _as_number(field, value)
    if number < 0:
        raise ValueValidationError(field, value, "must not be negative")
    if number != int(number):
        raise ValueValidationError(field, value, "must be a whole number of instruments")
    return int(number)


def parse_unit_weight(field: str, value: Any) -> float:
    """Grams per instrument: non-negative and finite."""
    number = _as_number(field, value)
    if number < 0:
        raise ValueValidationError(field, value, "must not be negative")
    return number


def _parse_map(values: Mapping[str, Any], parser) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueValidationError("<body>", values, "expected a JSON object")
    parsed: Dict[str, Any] = {}
    for key, value in values.items():
        name = str(key).strip()
        if not name:
            raise ValueValidationError(str(key), value, "class name must not be empty")
        parsed[name] = parser(name, value)
    return parsed


def parse_standards_map(values: Mapping[str, Any]) -> Dict[str, int]:
    return _parse_map(values, parse_standard_quantity)


def parse_unit_weights_map(values: Mapping[str, Any]) -> Dict[str, float]:
    return _parse_map(values, parse_unit_weight)


def is_usable_class_weight(value: Any) -> bool:
    """A class weight only counts if it is a finite, strictly positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) > 0


def clean_loaded_quantity(field: str, value: Any) -> Tuple[bool, int]:
    """Parse a value read from an existing profile file on disk.

    Returns ``(ok, quantity)``.  Bad values are dropped by the caller with a
    warning rather than rounded — a legacy 1.7 is corrupt data, and turning it
    into 1 would invent an expectation nobody set.
    """
    try:
        return True, parse_standard_quantity(field, value)
    except ValueValidationError:
        return False, 0


def clean_loaded_weight(field: str, value: Any) -> Tuple[bool, float]:
    try:
        return True, parse_unit_weight(field, value)
    except ValueValidationError:
        return False, 0.0

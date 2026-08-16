"""Inventory comparison — detected counts versus expected standards.

The one rule that matters here: iterate the UNION of expected and detected
classes.  An instrument the model missed entirely has no entry in ``counts``,
so walking ``counts`` alone silently drops the most important row an inventory
system can produce — the one that is missing.

The kiosk UI mirrors this logic in JavaScript; both sides must agree.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, NamedTuple

STATUS_NORMAL = "正常"
STATUS_MISSING = "缺少"
STATUS_EXTRA = "多出"


class InventoryRow(NamedTuple):
    class_name: str
    detected: int
    standard: int
    status: str


class InventoryTally(NamedTuple):
    normal: int
    missing: int
    extra: int

    @property
    def total(self) -> int:
        return self.normal + self.missing + self.extra


def status_for(detected: int, standard: int) -> str:
    if detected == standard:
        return STATUS_NORMAL
    return STATUS_MISSING if detected < standard else STATUS_EXTRA


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def compare_inventory(counts: Mapping[str, Any],
                      standards: Mapping[str, Any]) -> List[InventoryRow]:
    """One row per class that is either expected or detected, sorted by name.

    A class with neither an expectation nor a detection is dropped: those are
    auto-registered classes the model can emit but this tray does not use, and
    listing them as 正常 would bury the real rows.
    """
    counts = counts or {}
    standards = standards or {}
    rows: List[InventoryRow] = []
    for class_name in sorted(set(standards) | set(counts)):
        detected = _as_int(counts.get(class_name))
        standard = _as_int(standards.get(class_name))
        if detected == 0 and standard == 0:
            continue
        rows.append(InventoryRow(class_name, detected, standard,
                                 status_for(detected, standard)))
    return rows


def tally(rows: List[InventoryRow]) -> InventoryTally:
    normal = sum(1 for r in rows if r.status == STATUS_NORMAL)
    missing = sum(1 for r in rows if r.status == STATUS_MISSING)
    extra = sum(1 for r in rows if r.status == STATUS_EXTRA)
    return InventoryTally(normal, missing, extra)


def summarise(counts: Mapping[str, Any],
              standards: Mapping[str, Any]) -> Dict[str, Any]:
    rows = compare_inventory(counts, standards)
    counts_tally = tally(rows)
    return {
        "rows": [row._asdict() for row in rows],
        "normal": counts_tally.normal,
        "missing": counts_tally.missing,
        "extra": counts_tally.extra,
    }

from __future__ import annotations

import csv
import io
from typing import Any, Mapping, Optional, Sequence

from app.web.history import record_standards


def _status(detected: int, standard: int) -> str:
    if detected == standard:
        return "正常"
    return "缺少" if detected < standard else "多出"


def generate_csv(records: Sequence[Mapping[str, Any]],
                 standards: Optional[Mapping[str, int]] = None) -> str:
    """History export as a UTF-8-with-BOM CSV (opens correctly in Chinese Excel).

    Each row is judged against the standards stored *with that record*, not the
    currently active package's standards.  Exporting yesterday's obstetric
    inventory must not compare it against today's orthopaedic expectations.
    ``standards`` is only a fallback for pre-package records that carry no
    snapshot.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "timestamp", "source", "package_id", "package_name",
        "class_name", "detected", "standard", "status", "weight_g",
    ])

    for rec in records:
        ts = rec.get("timestamp", "")
        source = rec.get("source", "")
        counts = rec.get("counts", {}) or {}
        weight = rec.get("weight", "")
        weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else ""
        pkg_id = rec.get("package_id") or ""
        pkg_name = rec.get("package_display_name") or ""
        rec_standards = record_standards(rec, standards)

        if not counts:
            writer.writerow([ts, source, pkg_id, pkg_name, "", 0, "", "", weight_str])
            continue

        for class_name, detected in sorted(counts.items()):
            std = rec_standards.get(class_name, 0)
            writer.writerow([
                ts, source, pkg_id, pkg_name,
                class_name, detected, std, _status(detected, std), weight_str,
            ])

    return buf.getvalue()


def generate_bom_csv(
    counts: Mapping[str, int],
    standards: Mapping[str, int],
    unit_weights: Mapping[str, float],
    actual_weight: Optional[float],
    tolerance: float,
    timestamp: str,
    *,
    package_id: str = "",
    package_display_name: str = "",
    weight_verification: Optional[Mapping[str, Any]] = None,
) -> str:
    """BOM report for the current runtime state of the ACTIVE package."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    if package_id or package_display_name:
        writer.writerow(["【器械套件】"])
        writer.writerow(["套件代號", package_id])
        writer.writerow(["套件名稱", package_display_name])
        writer.writerow([])

    # Section 1: instrument breakdown
    writer.writerow(["器械類別", "標準數量", "實際數量", "單重(g)", "預期小計(g)"])
    all_classes = sorted(
        set(list(standards.keys()) + list(counts.keys()) + list(unit_weights.keys()))
    )
    total_expected = 0.0
    for cls in all_classes:
        std = standards.get(cls, 0)
        cnt = counts.get(cls, 0)
        uw = unit_weights.get(cls, 0.0)
        subtotal = std * uw
        total_expected += subtotal
        writer.writerow([cls, std, cnt, f"{uw:.4f}", f"{subtotal:.4f}"])

    # Section 2: weight summary
    writer.writerow([])
    writer.writerow(["【重量驗證彙總】"])
    # Prefer the platform's verdict (it knows about the stability gate) and fall
    # back to a direct comparison for callers that do not pass one.
    if weight_verification is not None:
        expected = weight_verification.get("expected", total_expected)
        writer.writerow(["總預期重量(g)", f"{float(expected):.4f}"])
        if actual_weight is not None:
            writer.writerow(["實際秤重(g)", f"{actual_weight:.4f}"])
        else:
            writer.writerow(["實際秤重(g)", "無法讀取"])
        diff = weight_verification.get("difference")
        writer.writerow(["差異(g)", f"{float(diff):.4f}" if diff is not None else "—"])
        writer.writerow(["容許範圍(g)", f"±{tolerance:.2f}"])
        writer.writerow(["重量穩定", "是" if weight_verification.get("stable") else "否"])
        passed = weight_verification.get("passed")
        if not weight_verification.get("ready"):
            verdict = "尚未完成量測（%s）" % weight_verification.get("message", "")
        else:
            verdict = "PASSED" if passed else "FAILED"
        writer.writerow(["驗證結果", verdict])
    else:
        writer.writerow(["總預期重量(g)", f"{total_expected:.4f}"])
        if actual_weight is not None:
            diff = abs(actual_weight - total_expected)
            writer.writerow(["實際秤重(g)", f"{actual_weight:.4f}"])
            writer.writerow(["差異(g)", f"{diff:.4f}"])
            writer.writerow(["容許範圍(g)", f"±{tolerance:.2f}"])
            writer.writerow(["驗證結果", "PASSED" if diff <= tolerance else "FAILED"])
        else:
            writer.writerow(["實際秤重(g)", "無法讀取"])
            writer.writerow(["差異(g)", "—"])
            writer.writerow(["容許範圍(g)", f"±{tolerance:.2f}"])
            writer.writerow(["驗證結果", "無法驗證"])
    writer.writerow([])
    writer.writerow(["產生時間", timestamp])

    return buf.getvalue()

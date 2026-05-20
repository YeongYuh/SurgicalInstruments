from __future__ import annotations

import csv
import io
from typing import Optional


def generate_csv(records: list[dict], standards: dict[str, int]) -> str:
    """Return a UTF-8-with-BOM CSV string (opens correctly in Chinese Excel)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "source", "class_name", "detected", "standard", "status", "weight_g"])

    for rec in records:
        ts = rec.get("timestamp", "")
        source = rec.get("source", "")
        counts = rec.get("counts", {})
        weight = rec.get("weight", "")
        weight_str = f"{weight:.2f}" if isinstance(weight, (int, float)) else ""

        if not counts:
            writer.writerow([ts, source, "", 0, "", "", weight_str])
            continue

        for class_name, detected in sorted(counts.items()):
            std = standards.get(class_name, 0)
            if detected == std:
                status = "正常"
            elif detected < std:
                status = "缺少"
            else:
                status = "多出"
            writer.writerow([ts, source, class_name, detected, std, status, weight_str])

    return buf.getvalue()


def generate_bom_csv(
    counts: dict,
    standards: dict,
    unit_weights: dict,
    actual_weight: Optional[float],
    tolerance: float,
    timestamp: str,
) -> str:
    """BOM report: per-class breakdown + weight verification summary."""
    buf = io.StringIO()
    writer = csv.writer(buf)

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
    writer.writerow(["總預期重量(g)", f"{total_expected:.4f}"])
    if actual_weight is not None:
        diff = abs(actual_weight - total_expected)
        passed = diff <= tolerance
        writer.writerow(["實際秤重(g)", f"{actual_weight:.4f}"])
        writer.writerow(["差異(g)", f"{diff:.4f}"])
        writer.writerow(["容許範圍(g)", f"±{tolerance:.2f}"])
        writer.writerow(["驗證結果", "PASSED" if passed else "FAILED"])
    else:
        writer.writerow(["實際秤重(g)", "無法讀取"])
        writer.writerow(["差異(g)", "—"])
        writer.writerow(["容許範圍(g)", f"±{tolerance:.2f}"])
        writer.writerow(["驗證結果", "無法驗證"])
    writer.writerow([])
    writer.writerow(["產生時間", timestamp])

    return buf.getvalue()

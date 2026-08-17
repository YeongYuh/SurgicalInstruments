"""Score model predictions as an INVENTORY result, not as a detection benchmark.

The system counts instruments, so the questions that matter are "is the tray
total right" and "is each class count right".  A box in the right place with
the wrong label still sends a nurse looking for an instrument that is on the
table, so IoU is deliberately not part of the verdict.

Usage:
    python3 tools/score_inventory.py RESULTS.json GROUND_TRUTH.json --out MATRIX.json

GROUND_TRUTH.json maps a frame filename (or a scene prefix) to
    {"scene": "...", "preset": "...", "condition": "...", "counts": {cls: n}}
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def score(truth: dict, pred: dict) -> dict:
    classes = set(truth) | set(pred)
    fn = fp = 0
    per_class = {}
    for cls in sorted(classes):
        t = int(truth.get(cls, 0))
        p = int(pred.get(cls, 0))
        per_class[cls] = {"truth": t, "pred": p, "exact": t == p}
        if p < t:
            fn += t - p
        elif p > t:
            fp += p - t
    exact_classes = sum(1 for v in per_class.values() if v["exact"])
    return {
        "exact_tray_match": all(v["exact"] for v in per_class.values()),
        "per_class": per_class,
        "per_class_exact": exact_classes,
        "per_class_total": len(per_class),
        # Against the tray: what the operator would be told to go and find,
        # and what they would be told to remove.
        "missing_units": fn,
        "extra_units": fp,
        "false_negative_units": fn,
        "false_positive_units": fp,
        "total_ground_truth_units": sum(truth.values()),
        "total_predicted_units": sum(pred.values()),
        "total_units_match": sum(truth.values()) == sum(pred.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    truth_map = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    rows = []
    for results_path in args.results:
        data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        for frame, variants in sorted(data["frames"].items()):
            entry = truth_map.get(frame)
            if entry is None:  # allow one truth entry per scene prefix
                key = next((k for k in truth_map if frame.startswith(k)), None)
                entry = truth_map.get(key)
            if entry is None:
                print("  !! no ground truth for %s — skipped" % frame)
                continue
            row = {"frame": frame, "scene": entry["scene"],
                   "preset": entry.get("preset", ""),
                   "condition": entry.get("condition", ""),
                   "ground_truth": entry["counts"]}
            for tag in ("640", "448"):
                if tag not in variants:
                    continue
                v = variants[tag]
                row[tag] = score(entry["counts"], v["counts"])
                row[tag]["counts"] = v["counts"]
                row[tag]["inference_ms"] = v["inference_ms"]
                row[tag]["min_confidence"] = (
                    round(min((d["confidence"] for d in v["detections"]), default=0.0), 4))
            rows.append(row)

    summary = {}
    for tag in ("640", "448"):
        got = [r[tag] for r in rows if tag in r]
        if not got:
            continue
        summary[tag] = {
            "frames": len(got),
            "exact_tray_rate": round(
                sum(1 for g in got if g["exact_tray_match"]) / len(got), 4),
            "per_class_exact_rate": round(
                sum(g["per_class_exact"] for g in got)
                / max(1, sum(g["per_class_total"] for g in got)), 4),
            "total_units_match_rate": round(
                sum(1 for g in got if g["total_units_match"]) / len(got), 4),
            "false_negative_units": sum(g["false_negative_units"] for g in got),
            "false_positive_units": sum(g["false_positive_units"] for g in got),
            "median_inference_ms": round(
                statistics.median(g["inference_ms"] for g in got), 1),
        }

    out = {"rows": rows, "summary": summary}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")

    print("%-30s %-14s %-5s %-5s %-4s %-4s %-7s" %
          ("frame", "condition", "640ex", "448ex", "640u", "448u", "ms"))
    for r in rows:
        print("%-30s %-14s %-5s %-5s %-4s %-4s %s/%s" % (
            r["frame"][:30], r["condition"],
            r.get("640", {}).get("exact_tray_match"),
            r.get("448", {}).get("exact_tray_match"),
            r.get("640", {}).get("total_predicted_units"),
            r.get("448", {}).get("total_predicted_units"),
            r.get("640", {}).get("inference_ms"),
            r.get("448", {}).get("inference_ms")))
    print()
    for tag, s in summary.items():
        print("  %s: exact_tray=%s  per_class_exact=%s  FN=%s  FP=%s  median=%sms"
              % (tag, s["exact_tray_rate"], s["per_class_exact_rate"],
                 s["false_negative_units"], s["false_positive_units"],
                 s["median_inference_ms"]))


if __name__ == "__main__":
    main()

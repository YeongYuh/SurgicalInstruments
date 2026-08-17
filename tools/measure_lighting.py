"""Quantify what the camera actually sees, using production capture settings.

A single frame says almost nothing: USB cameras ship black frames while the
sensor settles, and auto-exposure keeps moving for a second or two afterwards.
This grabs a batch, discards the warm-up, and reports the distribution — so
"the scene is too dark" becomes a number instead of an impression.

Usage:
    python3 tools/measure_lighting.py OUT.json [--frames 30] [--save-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

import app.config as config

DARK_LEVEL = 16       # below this a pixel carries no usable detail
SATURATED_LEVEL = 245  # above this the sensor has clipped


def open_camera():
    source = config.CAMERA_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("cannot open camera source %r" % (source,))
    fourcc = getattr(config, "CAMERA_FOURCC", "YUYV")
    if fourcc and fourcc != "AUTO":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=15,
                    help="frames discarded so auto-exposure can settle")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cap = open_camera()
    try:
        for _ in range(args.warmup):
            cap.read()
            time.sleep(0.03)

        grays, per_frame = [], []
        collected = 0
        deadline = time.monotonic() + 60.0
        while collected < args.frames and time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grays.append(gray)
            per_frame.append({
                "index": collected,
                "mean": round(float(gray.mean()), 3),
                "median": float(np.median(gray)),
                "dark_ratio": round(float((gray < DARK_LEVEL).mean()), 5),
                "saturated_ratio": round(float((gray > SATURATED_LEVEL).mean()), 5),
            })
            if args.save_dir:
                cv2.imwrite("%s/frame_%02d.jpg" % (args.save_dir, collected), frame)
            collected += 1
            time.sleep(0.06)
    finally:
        cap.release()

    if not grays:
        raise SystemExit("no frames captured")

    allpix = np.concatenate([g.reshape(-1) for g in grays])
    frame_means = [f["mean"] for f in per_frame]
    report = {
        "label": args.label,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_count": len(grays),
        "resolution": [int(grays[0].shape[1]), int(grays[0].shape[0])],
        "fourcc": getattr(config, "CAMERA_FOURCC", "YUYV"),
        "source": str(config.CAMERA_SOURCE),
        "mean": round(float(allpix.mean()), 3),
        "median": float(np.median(allpix)),
        "p5": float(np.percentile(allpix, 5)),
        "p50": float(np.percentile(allpix, 50)),
        "p95": float(np.percentile(allpix, 95)),
        "std": round(float(allpix.std()), 3),
        "dark_ratio": round(float((allpix < DARK_LEVEL).mean()), 5),
        "saturated_ratio": round(float((allpix > SATURATED_LEVEL).mean()), 5),
        "frame_mean_min": round(min(frame_means), 3),
        "frame_mean_max": round(max(frame_means), 3),
        "frame_mean_spread": round(max(frame_means) - min(frame_means), 3),
        "dark_level": DARK_LEVEL,
        "saturated_level": SATURATED_LEVEL,
        "per_frame": per_frame,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)

    keys = ("frame_count", "mean", "median", "p5", "p50", "p95",
            "dark_ratio", "saturated_ratio",
            "frame_mean_min", "frame_mean_max", "frame_mean_spread")
    for k in keys:
        print("  %-18s %s" % (k, report[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

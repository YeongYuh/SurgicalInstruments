"""Capture raw camera frames to disk, using production capture settings.

Accuracy comparisons must replay one saved frame into every model, so the
frame is written once here and never re-captured per model.  Warm-up frames are
discarded because a USB camera's first frames are black and its auto-exposure
is still moving.

Usage:
    python3 tools/capture_frame.py OUT_PREFIX [--count 3] [--gap 1.0]
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

import app.config as config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="e.g. /path/scene_b_complete  -> _01.jpg ...")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--gap", type=float, default=1.0, help="seconds between frames")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not cap.isOpened():
        raise SystemExit("cannot open %r" % (config.CAMERA_SOURCE,))
    fourcc = getattr(config, "CAMERA_FOURCC", "YUYV")
    if fourcc and fourcc != "AUTO":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

    written = []
    try:
        for _ in range(args.warmup):
            cap.read()
            time.sleep(0.03)
        for i in range(1, args.count + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                print("  !! frame %d failed" % i)
                continue
            path = "%s_%02d.jpg" % (args.prefix, i)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("  wrote %s  %dx%d  mean=%.1f"
                  % (path, frame.shape[1], frame.shape[0], gray.mean()))
            written.append(path)
            if i < args.count:
                time.sleep(args.gap)
    finally:
        cap.release()
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

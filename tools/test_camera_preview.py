#!/usr/bin/env python3
"""Standalone camera capture test — run before the web app to isolate hardware issues.

Usage:
    python3 tools/test_camera_preview.py
    CAMERA_SOURCE=/dev/video1 python3 tools/test_camera_preview.py
    WEBCAM_WIDTH=1280 WEBCAM_HEIGHT=720 python3 tools/test_camera_preview.py

Output:
    - Per-second stats: ret ok/fail, frame shape, JPEG size
    - Summary at end
    - Sample frame saved to tmp/camera_test.jpg
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CAMERA_SOURCE_RAW = os.environ.get("CAMERA_SOURCE", "/dev/video0")
try:
    CAMERA_SOURCE = int(CAMERA_SOURCE_RAW)
except ValueError:
    CAMERA_SOURCE = CAMERA_SOURCE_RAW

WIDTH   = int(os.environ.get("WEBCAM_WIDTH",  "640"))
HEIGHT  = int(os.environ.get("WEBCAM_HEIGHT", "480"))
FPS_REQ = int(os.environ.get("WEBCAM_FPS",    "15"))
DURATION = int(os.environ.get("TEST_DURATION", "30"))

OUT_DIR = Path("tmp")
OUT_DIR.mkdir(exist_ok=True)
SAMPLE_PATH = OUT_DIR / "camera_test.jpg"

print("=" * 56)
print(" Camera Preview Test")
print("=" * 56)
print(f"  Source   : {CAMERA_SOURCE!r}")
print(f"  Requested: {WIDTH}x{HEIGHT} @ {FPS_REQ} fps")
print(f"  Duration : {DURATION} s")
print(f"  Sample   : {SAMPLE_PATH}")
print("=" * 56)

# Open with explicit V4L2 backend
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"[ERROR] Failed to open {CAMERA_SOURCE!r} with CAP_V4L2")
    sys.exit(1)

# Configure
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          FPS_REQ)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
cc_int     = int(cap.get(cv2.CAP_PROP_FOURCC))
cc_str     = "".join(chr((cc_int >> i) & 0xFF) for i in (0, 8, 16, 24))

print(f"  Actual   : {actual_w}x{actual_h} @ {actual_fps:.0f} fps  fourcc={cc_str!r}")
print()

total_ok   = 0
total_fail = 0
sample_saved = False
interval_ok = interval_fail = 0
interval_start = time.time()
test_start = time.time()
last_shape = None
last_jpeg_size = 0

try:
    while time.time() - test_start < DURATION:
        ret, frame = cap.read()
        now = time.time()

        if not ret:
            total_fail += 1
            interval_fail += 1
            continue

        total_ok += 1
        interval_ok += 1
        last_shape = frame.shape

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            last_jpeg_size = len(buf)
            if not sample_saved:
                cv2.imwrite(str(SAMPLE_PATH), frame)
                sample_saved = True
                print(f"  [sample]  Saved first frame → {SAMPLE_PATH}")

        # Print per-second stats
        if now - interval_start >= 1.0:
            elapsed = now - test_start
            fps_actual = interval_ok / (now - interval_start)
            mean_val   = float(frame.mean()) if frame is not None else 0.0
            print(
                f"  t={elapsed:5.1f}s  "
                f"ok={interval_ok:3d}  fail={interval_fail:2d}  "
                f"fps={fps_actual:4.1f}  "
                f"shape={last_shape}  "
                f"jpeg={last_jpeg_size:,} B  "
                f"mean={mean_val:.1f}"
            )
            interval_ok = interval_fail = 0
            interval_start = now

finally:
    cap.release()

total_elapsed = time.time() - test_start
print()
print("=" * 56)
print(" Summary")
print("=" * 56)
print(f"  Duration : {total_elapsed:.1f} s")
print(f"  Frames ok: {total_ok}")
print(f"  Frames fail: {total_fail}")
print(f"  Avg FPS  : {total_ok / total_elapsed:.1f}")
print(f"  Last JPEG: {last_jpeg_size:,} bytes")
print(f"  Last shape: {last_shape}")
if not sample_saved:
    print("  [WARNING] No valid frame captured!")
else:
    print(f"  Sample   : {SAMPLE_PATH}")
print("=" * 56)

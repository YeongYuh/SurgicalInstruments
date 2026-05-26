#!/usr/bin/env python3
"""Standalone camera capture test — run before the web app to isolate hardware issues.

Usage:
    python3 tools/test_camera_preview.py
    CAMERA_SOURCE=/dev/video1 python3 tools/test_camera_preview.py
    CAMERA_FOURCC=YUYV python3 tools/test_camera_preview.py
    WEBCAM_WIDTH=1280 WEBCAM_HEIGHT=720 python3 tools/test_camera_preview.py

Output:
    - Per-second stats: ret ok/fail, frame shape, JPEG size, mean pixel value
    - Summary and recommendation at end
    - Sample frame saved to tmp/camera_test.jpg
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

# ── Config from env vars ────────────────────────────────────────────────────
_src_raw = os.environ.get("CAMERA_SOURCE", "/dev/video0")
try:
    CAMERA_SOURCE: int | str = int(_src_raw)
except ValueError:
    CAMERA_SOURCE = _src_raw

WIDTH    = int(os.environ.get("WEBCAM_WIDTH",  "640"))
HEIGHT   = int(os.environ.get("WEBCAM_HEIGHT", "480"))
FPS_REQ  = int(os.environ.get("WEBCAM_FPS",    "15"))
FOURCC   = os.environ.get("CAMERA_FOURCC", "MJPG").upper()
DURATION = int(os.environ.get("TEST_DURATION", "30"))

OUT_DIR     = Path("tmp")
OUT_DIR.mkdir(exist_ok=True)
SAMPLE_PATH = OUT_DIR / "camera_test.jpg"

print("=" * 60)
print(" Camera Preview Test")
print("=" * 60)
print(f"  Source     : {CAMERA_SOURCE!r}")
print(f"  Requested  : {WIDTH}x{HEIGHT} @ {FPS_REQ} fps  fourcc={FOURCC}")
print(f"  Duration   : {DURATION} s")
print(f"  Sample     : {SAMPLE_PATH}")
print("=" * 60)

# ── Open with explicit V4L2 backend ─────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
if not cap.isOpened():
    print(f"[ERROR] Cannot open {CAMERA_SOURCE!r} with CAP_V4L2")
    sys.exit(1)

# ── Apply settings ───────────────────────────────────────────────────────────
if FOURCC in ("MJPG", "YUYV"):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
# AUTO: leave driver default

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS,          FPS_REQ)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
cc_int     = int(cap.get(cv2.CAP_PROP_FOURCC))
cc_str     = "".join(chr((cc_int >> i) & 0xFF) for i in (0, 8, 16, 24))

print(f"  Negotiated : {actual_w}x{actual_h} @ {actual_fps:.0f} fps  fourcc={cc_str!r}")
print()

# ── Capture loop ─────────────────────────────────────────────────────────────
total_ok       = 0
total_fail     = 0
sample_saved   = False
interval_ok    = 0
interval_fail  = 0
interval_start = time.time()
test_start     = time.time()
last_shape     = None
last_jpeg_bytes = 0

try:
    while time.time() - test_start < DURATION:
        ret, frame = cap.read()
        now = time.time()

        if not ret:
            total_fail    += 1
            interval_fail += 1
            continue

        total_ok    += 1
        interval_ok += 1
        last_shape  = frame.shape

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            last_jpeg_bytes = len(buf)
            if not sample_saved:
                cv2.imwrite(str(SAMPLE_PATH), frame)
                sample_saved = True
                print(f"  [sample]  First frame saved → {SAMPLE_PATH}")

        if now - interval_start >= 1.0:
            elapsed     = now - test_start
            fps_meas    = interval_ok / max(now - interval_start, 1e-9)
            mean_val    = float(frame.mean()) if frame is not None else 0.0
            print(
                f"  t={elapsed:5.1f}s  "
                f"ok={interval_ok:3d}  fail={interval_fail:2d}  "
                f"fps={fps_meas:5.1f}  "
                f"shape={last_shape}  "
                f"jpeg={last_jpeg_bytes:,} B  "
                f"mean={mean_val:.1f}"
            )
            interval_ok = interval_fail = 0
            interval_start = now

finally:
    cap.release()

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed_total = time.time() - test_start
avg_fps       = total_ok / max(elapsed_total, 1e-9)

print()
print("=" * 60)
print(" Summary")
print("=" * 60)
print(f"  Source      : {CAMERA_SOURCE!r}")
print(f"  Requested   : {WIDTH}x{HEIGHT} @ {FPS_REQ} fps  fourcc={FOURCC}")
print(f"  Negotiated  : {actual_w}x{actual_h} @ {actual_fps:.0f} fps  fourcc={cc_str!r}")
print(f"  Duration    : {elapsed_total:.1f} s")
print(f"  Frames ok   : {total_ok}")
print(f"  Frames fail : {total_fail}")
print(f"  Avg FPS     : {avg_fps:.1f}")
print(f"  Last JPEG   : {last_jpeg_bytes:,} bytes")
print(f"  Last shape  : {last_shape}")
if sample_saved:
    print(f"  Sample      : {SAMPLE_PATH}")
else:
    print("  [WARNING]   No valid frame captured!")

# ── Recommendation ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(" Recommendation")
print("=" * 60)
if not sample_saved:
    print("  Camera did not produce valid frames.")
    print("  Check: ls /dev/video*   dmesg | grep usb")
elif FOURCC == "MJPG":
    # "Corrupt JPEG data" warnings come from libjpeg inside OpenCV during MJPG
    # decoding.  They do not affect frame validity (shape/mean are fine) but
    # they clutter the log and can occasionally produce partially-decoded frames.
    print("  MJPG selected.")
    print(f"  Measured FPS: {avg_fps:.1f}  (requested {FPS_REQ})")
    if avg_fps < FPS_REQ * 0.7:
        print(f"  [NOTE] FPS lower than requested — USB bandwidth or driver limit.")
    print()
    print("  If you see 'Corrupt JPEG data' log lines during normal operation:")
    print("    CAMERA_FOURCC=YUYV ./run_jetson.sh")
    print("  YUYV avoids libjpeg decoding entirely (raw planar YUV from driver).")
    print("  It uses slightly more CPU for JPEG re-encode but no decode warnings.")
    print()
    print("  If MJPG preview is stable and warnings are only in the log (not")
    print("  visible as corrupted frames), keeping MJPG is fine.")
elif FOURCC == "YUYV":
    print("  YUYV selected — no libjpeg decode warnings expected.")
    print(f"  Measured FPS: {avg_fps:.1f}")
    if avg_fps < FPS_REQ * 0.7:
        print("  [NOTE] FPS lower than requested — YUYV raw transfer is bandwidth-heavy.")
        print("  Try: WEBCAM_FPS=10 CAMERA_FOURCC=YUYV ./run_jetson.sh")
    else:
        print("  Preview should be stable.  Use with:")
        print("    CAMERA_FOURCC=YUYV ./run_jetson.sh")
else:
    print(f"  AUTO fourcc — negotiated {cc_str!r}.")
    print(f"  Measured FPS: {avg_fps:.1f}")
print("=" * 60)

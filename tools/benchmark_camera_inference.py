"""
Benchmark detector inference on a live camera frame.

Captures one valid frame from the camera, then runs detector.predict() with a
detailed timing breakdown per step.  Does NOT start Flask, the scale reader, or
any background thread — purely a camera + detector benchmark.

Usage:
    cd /home/ncut/projects/instrument
    source venv/bin/activate

    # Default: ONNX backend, imgsz 640 / 416 / 320
    python3 tools/benchmark_camera_inference.py

    # Single imgsz
    python3 tools/benchmark_camera_inference.py --imgsz 416

    # Specific backend
    DETECTOR_BACKEND=onnx python3 tools/benchmark_camera_inference.py --imgsz 640

    # Override camera source
    python3 tools/benchmark_camera_inference.py --source /dev/video0
"""
from __future__ import annotations

import argparse
import base64
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

WARMUP = 2
RUNS   = 5


# ── Camera capture ────────────────────────────────────────────────────────────

def capture_camera_frame(source) -> np.ndarray:
    """Open camera, discard black warmup frames, return first valid frame."""
    import app.config as config

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, int):
        cap.release()
        path = f"/dev/video{source}"
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            print(f"  Integer index {source} failed; opened via {path}")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source={source!r}")

    fourcc = config.CAMERA_FOURCC
    if fourcc in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cc_int     = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc_str     = "".join(chr((cc_int >> i) & 0xFF) for i in (0, 8, 16, 24))
    print(f"  Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} fps  fourcc={cc_str!r}")

    frame = None
    for i in range(30):
        ret, f = cap.read()
        if ret and f is not None and float(f.mean()) > 5.0:
            frame = f
            print(f"  Frame captured on attempt {i+1}: "
                  f"{f.shape[1]}x{f.shape[0]}  mean={float(f.mean()):.1f}")
            break

    cap.release()

    if frame is None:
        raise RuntimeError("Could not capture a valid (non-black) frame from camera")
    return frame


# ── Per-step timing ───────────────────────────────────────────────────────────

def _run_one(detector, frame: np.ndarray, imgsz: int, conf: float) -> dict:
    from app.visualizer import draw_detections

    t0 = time.perf_counter()
    _, detections = detector.predict(frame, conf=conf, imgsz=imgsz)
    t_predict = time.perf_counter() - t0

    t0 = time.perf_counter()
    counts = detector.count_instruments(detections)
    t_count = time.perf_counter() - t0

    # Annotated image — included here to measure its cost even though
    # camera mode skips it.  Compare with/without to quantify the saving.
    t0 = time.perf_counter()
    annotated = draw_detections(frame, detections)
    t_draw = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _ = base64.b64encode(buf).decode()
    t_encode = time.perf_counter() - t0

    return {
        "predict_ms": t_predict * 1000,
        "count_ms":   t_count   * 1000,
        "draw_ms":    t_draw    * 1000,
        "encode_ms":  t_encode  * 1000,
        "total_ms":   (t_predict + t_count + t_draw + t_encode) * 1000,
        "dets":       len(detections),
        "counts":     counts,
    }


# ── Backend runner ────────────────────────────────────────────────────────────

def benchmark_backend(backend: str, frame: np.ndarray,
                       imgsz: int, conf: float) -> dict:
    from app.detector import SurgicalInstrumentDetector
    import app.config as config

    det = SurgicalInstrumentDetector(
        model_path=str(config.MODEL_PATH),
        backend=backend,
        onnx_path=str(config.ONNX_MODEL_PATH),
        onnx_task=config.ONNX_TASK,
    )

    t_load = time.perf_counter()
    det._load_model()
    load_ms   = (time.perf_counter() - t_load) * 1000
    effective = det._effective_backend

    print(f"\n  backend={backend}  effective={effective}  "
          f"load={load_ms:.0f} ms  imgsz={imgsz}")
    if backend == "onnx" and effective == "pt":
        print("  !!! ONNX fell back to PT — check ONNX_MODEL_PATH / ONNX_TASK !!!")

    print(f"  Warmup ({WARMUP} runs)...", end=" ", flush=True)
    for _ in range(WARMUP):
        _run_one(det, frame, imgsz, conf)
    print("done")

    rows = []
    print(f"  Benchmark ({RUNS} runs)...", end=" ", flush=True)
    for _ in range(RUNS):
        rows.append(_run_one(det, frame, imgsz, conf))
    print("done")

    def _stats(key: str) -> tuple[float, float, float]:
        vals = [r[key] for r in rows]
        return statistics.mean(vals), statistics.median(vals), min(vals)

    p_mean, p_med, p_min = _stats("predict_ms")
    d_mean, _,    d_min  = _stats("draw_ms")
    e_mean, _,    e_min  = _stats("encode_ms")
    t_mean, t_med, t_min = _stats("total_ms")
    dets = rows[-1]["dets"]
    cnts = rows[-1]["counts"]

    print(f"  ┌─ predict (model call) : mean={p_mean:6.0f} ms  med={p_med:6.0f} ms  min={p_min:6.0f} ms")
    print(f"  ├─ draw_detections      : mean={d_mean:6.0f} ms  min={d_min:6.0f} ms  (skipped in camera mode)")
    print(f"  ├─ imencode+base64      : mean={e_mean:6.0f} ms  min={e_min:6.0f} ms  (skipped in camera mode)")
    print(f"  └─ TOTAL                : mean={t_mean:6.0f} ms  med={t_med:6.0f} ms  min={t_min:6.0f} ms")
    print(f"     detections={dets}  counts={cnts}")

    return {
        "backend": backend, "effective": effective,
        "imgsz": imgsz, "load_ms": load_ms,
        "predict_mean_ms": p_mean, "predict_min_ms": p_min,
        "total_mean_ms": t_mean,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imgsz",  type=int,   default=None,
                        help="Single imgsz to test (default: sweep 640, 416, 320)")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--source", default=None,
                        help="Camera source (int index or /dev/videoN path)")
    args = parser.parse_args()

    import app.config as config

    # Resolve camera source
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
    else:
        raw = os.environ.get("CAMERA_SOURCE",
              os.environ.get("WEBCAM_INDEX", "0"))
        try:
            source = int(raw)
        except ValueError:
            source = raw

    print(f"\n{'='*62}")
    print(f"  Camera Inference Benchmark")
    print(f"  source={source!r}  conf={args.conf}")
    print(f"  model    = {config.MODEL_PATH}")
    print(f"  onnx     = {config.ONNX_MODEL_PATH}")
    print(f"  fourcc   = {config.CAMERA_FOURCC}")
    print(f"  warmup={WARMUP}  runs={RUNS}")
    print(f"{'='*62}")

    frame = capture_camera_frame(source)

    force_backend = os.environ.get("DETECTOR_BACKEND", "")
    backends      = [force_backend] if force_backend else ["onnx", "pt"]
    imgsz_list    = [args.imgsz] if args.imgsz else [640, 416, 320]

    summary: list[dict] = []
    for backend in backends:
        for imgsz in imgsz_list:
            print(f"\n{'─'*62}")
            try:
                r = benchmark_backend(backend, frame, imgsz, args.conf)
                summary.append(r)
            except Exception as exc:
                print(f"  [{backend} imgsz={imgsz}] FAILED: {exc}")

    # Summary table
    print(f"\n{'='*62}")
    print(f"  {'Backend':<8} {'Eff':<6} {'imgsz':>6}  {'predict':>8}  {'total':>8}")
    print(f"  {'─'*54}")
    for r in summary:
        print(f"  {r['backend']:<8} {r['effective']:<6} {r['imgsz']:>6}  "
              f"{r['predict_mean_ms']:>7.0f}ms  {r['total_mean_ms']:>7.0f}ms")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()

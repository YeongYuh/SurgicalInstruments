"""
Benchmark detector.py inference across pt and onnx backends.

Usage:
    cd /home/ncut/projects/instrument
    source venv/bin/activate

    # Run full benchmark
    python3 tools/benchmark_detector.py

    # Specific image
    python3 tools/benchmark_detector.py --image path/to/image.jpg

    # Specific backend only
    DETECTOR_BACKEND=onnx python3 tools/benchmark_detector.py
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

# ── Ensure project root on path ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

DEFAULT_IMAGE = ROOT / "input" / "test_image.jpg"
WARMUP  = 3
RUNS    = 10


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _run_backend(backend: str, image: np.ndarray, conf: float = 0.25) -> dict:
    """Load detector with the given backend and benchmark it."""
    from app.detector import SurgicalInstrumentDetector
    import app.config as config

    det = SurgicalInstrumentDetector(
        model_path=str(config.MODEL_PATH),
        backend=backend,
        onnx_path=str(config.ONNX_MODEL_PATH),
        onnx_task=config.ONNX_TASK,
    )

    # Force model load and record load time
    t_load = time.perf_counter()
    det._load_model()
    load_ms = (time.perf_counter() - t_load) * 1000
    effective = det._effective_backend

    print(f"\n  backend={backend}  effective={effective}  load={load_ms:.0f} ms")
    if backend == "onnx" and effective == "pt":
        print("  *** ONNX FELL BACK TO PT — check ONNX_MODEL_PATH ***")

    # Warmup
    print(f"  Warmup ({WARMUP} runs)...", end=" ", flush=True)
    for _ in range(WARMUP):
        det.predict(image, conf=conf)
    print("done")

    # Benchmark
    times = []
    detections_count = 0
    print(f"  Benchmark ({RUNS} runs)...", end=" ", flush=True)
    for i in range(RUNS):
        t0 = time.perf_counter()
        _, dets = det.predict(image, conf=conf)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        if i == RUNS - 1:
            detections_count = len(dets)
    print("done")

    return {
        "backend": backend,
        "effective": effective,
        "load_ms": load_ms,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "detections": detections_count,
        "runs": RUNS,
    }


def _benchmark_onnx_threads(image: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Sweep intra_op_num_threads for ONNX RT to find optimal setting."""
    import onnxruntime as ort
    import app.config as config
    from ultralytics import YOLO

    results = []
    for n_threads in [1, 2, 4]:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        session = ort.InferenceSession(
            str(config.ONNX_MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        # Prepare input: letterbox → float32 NCHW
        h, w = image.shape[:2]
        scale = 640 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))
        canvas = np.zeros((640, 640, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        inp = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, norm
        inp = np.ascontiguousarray(inp.transpose(2, 0, 1))[np.newaxis]  # NCHW

        # Warmup
        for _ in range(WARMUP):
            session.run(output_names, {input_name: inp})

        # Benchmark raw session.run() only
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            session.run(output_names, {input_name: inp})
            times.append((time.perf_counter() - t0) * 1000)

        results.append({
            "threads": n_threads,
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
        })
        print(f"    intra_op_threads={n_threads}: mean={statistics.mean(times):.0f} ms  "
              f"median={statistics.median(times):.0f} ms  min={min(times):.0f} ms")

    return results


def _benchmark_onnx_raw_vs_app(image: np.ndarray, conf: float = 0.25) -> None:
    """Compare raw session.run() vs full ultralytics wrapper."""
    import onnxruntime as ort
    import app.config as config

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0  # same as ultralytics default

    session = ort.InferenceSession(
        str(config.ONNX_MODEL_PATH),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    h, w = image.shape[:2]
    scale = 640 / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((640, 640, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    inp = canvas[:, :, ::-1].astype(np.float32) / 255.0
    inp = np.ascontiguousarray(inp.transpose(2, 0, 1))[np.newaxis]

    for _ in range(WARMUP):
        session.run(output_names, {input_name: inp})

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: inp})
        times.append((time.perf_counter() - t0) * 1000)

    print(f"  Raw session.run() (no pre/postprocess): mean={statistics.mean(times):.0f} ms  "
          f"median={statistics.median(times):.0f} ms  min={min(times):.0f} ms")


def _print_table(results: list[dict]) -> None:
    print()
    header = f"{'Backend':<10} {'Effective':<10} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Dets':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['backend']:<10} {r['effective']:<10} "
              f"{r['mean_ms']:>7.0f}ms {r['median_ms']:>7.0f}ms "
              f"{r['min_ms']:>7.0f}ms {r['max_ms']:>7.0f}ms "
              f"{r['detections']:>5}")

    if len(results) == 2:
        a, b = results[0]["mean_ms"], results[1]["mean_ms"]
        faster = results[0]["backend"] if a < b else results[1]["backend"]
        ratio = max(a, b) / min(a, b)
        print(f"\n  {faster} is {ratio:.2f}x faster (mean)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        print("  Provide --image path/to/image.jpg  or place an image at input/test_image.jpg")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Benchmark: {image_path.name}")
    image = _load_image(image_path)
    print(f"  Image size: {image.shape[1]}x{image.shape[0]} px")
    print(f"  Warmup={WARMUP}  Runs={RUNS}  conf={args.conf}")
    print(f"{'='*60}")

    force_backend = os.environ.get("DETECTOR_BACKEND", "")
    backends_to_test = [force_backend] if force_backend else ["pt", "onnx"]

    results = []
    for backend in backends_to_test:
        try:
            r = _run_backend(backend, image, conf=args.conf)
            results.append(r)
        except Exception as e:
            print(f"  [{backend}] FAILED: {e}")

    print(f"\n{'='*60}")
    print("  RESULTS — Full app path (detector.predict)")
    _print_table(results)

    # Only do deep ONNX analysis if ONNX was tested
    if any(r["effective"] == "onnx" for r in results):
        print(f"\n{'='*60}")
        print("  ONNX raw session.run() (default threads, no pre/postprocess)")
        _benchmark_onnx_raw_vs_app(image, conf=args.conf)

        print(f"\n{'='*60}")
        print("  ONNX thread sweep — raw session.run() only")
        _benchmark_onnx_threads(image, conf=args.conf)

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

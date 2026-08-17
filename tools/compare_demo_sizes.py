"""Run the SAME raw frame through the demo 640 and 448 ONNX models.

Comparing two live captures would compare two different scenes, so every
scene here is captured once to disk and replayed into both models.  Nothing is
shared between the two adapters except that file.

The headline number is inventory COUNT correctness, not detection overlap:
this is a counting system, so a box in roughly the right place is worth
nothing if it makes the tray total wrong.

Usage:
    python3 tools/compare_demo_sizes.py FRAME.jpg [FRAME2.jpg ...] --out RESULT.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

import app.config as config
from app.inference.package import load_manifest
from app.inference.registry import get_adapter_class

DEMO_DIR = Path(config.PROJECT_ROOT) / "model_packages" / "demo"

VARIANTS = {
    "640": {"model_file": "models/demo/best.onnx", "image_size": 640},
    "448": {"model_file": "models/demo/best_448.onnx", "image_size": 448},
}


def build_adapter(tmpdir: Path, tag: str, spec: dict):
    """A package identical to demo except for the model file and input size."""
    manifest = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["id"] = "demo"
    manifest["model_file"] = spec["model_file"]
    manifest["inference"]["image_size"] = spec["image_size"]
    root = tmpdir / tag
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    package = load_manifest(root / "manifest.json",
                            project_root=Path(config.PROJECT_ROOT))
    adapter = get_adapter_class(package.adapter)(package)
    adapter.load()
    adapter.warmup()
    return package, adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/tmp/si007-variants")
    args = ap.parse_args()

    tmpdir = Path(args.tmp)
    tmpdir.mkdir(parents=True, exist_ok=True)

    results = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": {}}

    for tag, spec in VARIANTS.items():
        package, adapter = build_adapter(tmpdir, tag, spec)
        conf = package.confidence
        print("[%s] loaded %s imgsz=%s conf=%s"
              % (tag, spec["model_file"], spec["image_size"], conf))
        try:
            for path in args.frames:
                image = cv2.imread(path)
                if image is None:
                    print("  !! cannot read %s" % path)
                    continue
                t0 = time.monotonic()
                result = adapter.infer(image)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                counts = dict(Counter(d.class_name for d in result.detections))
                entry = results["frames"].setdefault(Path(path).name, {})
                entry[tag] = {
                    "counts": counts,
                    "total_units": sum(counts.values()),
                    "inference_ms": round(elapsed_ms, 1),
                    "image_size": spec["image_size"],
                    "confidence": conf,
                    "detections": [
                        {"class": d.class_name,
                         "confidence": round(float(d.confidence), 4),
                         "bbox": ([round(float(v), 1) for v in d.bbox_xyxy]
                                  if d.has_bbox else None)}
                        for d in result.detections
                    ],
                }
                print("  %-28s %-4s %6.0f ms  units=%d  %s"
                      % (Path(path).name, tag, elapsed_ms,
                         sum(counts.values()), counts or "{}"))
        finally:
            adapter.unload()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()

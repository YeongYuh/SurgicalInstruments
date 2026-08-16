"""Shared test scaffolding.

Everything here runs without a real model binary, without a camera, and
without a scale.  The architecture is exercised through a FakeAdapter, so the
suite stays green on a development machine where models/best.onnx does not
exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before anything imports app.config: it keeps importing app.web
# from loading a model or opening a serial port.
os.environ.setdefault("INSTRUMENT_DISABLE_BOOTSTRAP", "true")
os.environ.setdefault("SCALE_READER_MODE", "mock")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from app.inference.base import AdapterError, ModelAdapter  # noqa: E402
from app.inference.registry import register_adapter  # noqa: E402
from app.inference.types import Detection, InferenceResult, counts_from_detections  # noqa: E402


class FakeAdapter(ModelAdapter):
    """A model that returns whatever the manifest tells it to.

    ``adapter_options``:
        counts      {"class": n}  what to "detect"
        with_bbox   emit geometry (default True)
        fail_load   raise on load, to test that a failed switch is contained
    """

    name = "fake"

    def __init__(self, package) -> None:
        super().__init__(package)
        self.load_calls = 0
        self.unload_calls = 0
        self.warmup_calls = 0
        self.infer_calls = 0

    def _do_load(self) -> None:
        self.load_calls += 1
        if self.options.get("fail_load"):
            raise AdapterError("simulated load failure for package '%s'" % self.package.id)

    def _do_unload(self) -> None:
        self.unload_calls += 1

    def _do_warmup(self) -> None:
        self.warmup_calls += 1

    def _do_infer(self, image, conf, imgsz) -> InferenceResult:
        self.infer_calls += 1
        with_bbox = self.options.get("with_bbox", True)
        detections: List[Detection] = []
        counts: Dict[str, int] = self.options.get("counts", {"widget": 1})
        for class_id, (name, qty) in enumerate(sorted(counts.items())):
            for i in range(int(qty)):
                detections.append(Detection(
                    class_name=name,
                    class_id=class_id,
                    confidence=0.9,
                    bbox_xyxy=(10.0 + i, 20.0, 30.0 + i, 40.0) if with_bbox else None,
                ))
        return InferenceResult(
            counts=counts_from_detections(detections),
            detections=detections,
            metadata={"conf": conf, "imgsz": imgsz},
        )


class CountsOnlyAdapter(FakeAdapter):
    """A model with no geometry at all — counts are the entire output.

    This is the shape a classifier or a bespoke counting network would take,
    and the platform must handle it end to end.
    """

    name = "counts_only"

    def _do_infer(self, image, conf, imgsz) -> InferenceResult:
        self.infer_calls += 1
        counts: Dict[str, int] = self.options.get("counts", {"widget": 2})
        detections = [
            Detection(class_name=name, class_id=idx, confidence=0.8, bbox_xyxy=None)
            for idx, (name, qty) in enumerate(sorted(counts.items()))
            for _ in range(int(qty))
        ]
        return InferenceResult(counts=dict(counts), detections=detections)


register_adapter("fake", FakeAdapter, replace=True)
register_adapter("counts_only", CountsOnlyAdapter, replace=True)


# ── package fixtures on disk ────────────────────────────────────────────────

def write_package(
    packages_dir: Path,
    package_id: str,
    *,
    adapter: str = "fake",
    display_name: Optional[str] = None,
    department: str = "test",
    class_weights: Optional[Dict[str, float]] = None,
    standards: Optional[Dict[str, int]] = None,
    adapter_options: Optional[Dict[str, Any]] = None,
    confidence: float = 0.25,
    image_size: Optional[int] = 320,
    model_file: Optional[str] = "model.bin",
    create_model_file: bool = True,
    template: bool = False,
    schema_version: int = 1,
    manifest_overrides: Optional[Dict[str, Any]] = None,
    omit: Optional[List[str]] = None,
) -> Path:
    """Create model_packages/<id>/ with a manifest and its data files."""
    root = Path(packages_dir) / package_id
    root.mkdir(parents=True, exist_ok=True)

    if class_weights is None:
        class_weights = {"widget": 10.0, "gizmo": 5.0}
    if standards is None:
        standards = {"widget": 2, "gizmo": 1}

    (root / "class_weight.json").write_text(
        json.dumps(class_weights, ensure_ascii=False), encoding="utf-8")
    (root / "standards.json").write_text(
        json.dumps(standards, ensure_ascii=False), encoding="utf-8")
    if model_file and create_model_file:
        (root / model_file).write_bytes(b"fake-model")

    manifest: Dict[str, Any] = {
        "schema_version": schema_version,
        "id": package_id,
        "display_name": display_name or ("Package " + package_id),
        "department": department,
        "adapter": adapter,
        "adapter_options": adapter_options or {},
        "inference": {"confidence": confidence},
        "inventory": {
            "class_weights": "class_weight.json",
            "default_standards": "standards.json",
        },
    }
    if model_file:
        manifest["model_file"] = model_file
    if image_size is not None:
        manifest["inference"]["image_size"] = image_size
    if template:
        manifest["template"] = True
    if manifest_overrides:
        manifest.update(manifest_overrides)
    for key in (omit or []):
        manifest.pop(key, None)

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def write_raw_manifest(packages_dir: Path, package_id: str, payload: Any) -> Path:
    """Write an arbitrary (possibly invalid) manifest for negative tests."""
    root = Path(packages_dir) / package_id
    root.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    (root / "manifest.json").write_text(text, encoding="utf-8")
    return root


def make_manager(tmp_path: Path, **kwargs):
    """A ModelManager rooted entirely inside tmp_path."""
    from app.inference.manager import ModelManager

    packages_dir = Path(tmp_path) / "model_packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = Path(tmp_path) / "profiles"
    return ModelManager(
        packages_dir=packages_dir,
        profiles_dir=profiles_dir,
        project_root=Path(tmp_path),
        **kwargs,
    )

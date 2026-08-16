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
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import pytest  # noqa: E402

from app.inference.base import AdapterError, ModelAdapter  # noqa: E402
from app.inference.registry import register_adapter  # noqa: E402
from app.inference.types import Detection, InferenceResult, counts_from_detections  # noqa: E402


class ResidentModels:
    """Process-wide count of adapters currently holding a loaded model.

    ``peak`` is what proves the Jetson RAM guarantee: a package switch must
    never push it above 1, no matter how the load and unload interleave.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def acquire(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def release(self) -> None:
        with self._lock:
            self.current = max(0, self.current - 1)

    def reset(self) -> None:
        with self._lock:
            self.current = 0
            self.peak = 0


RESIDENT = ResidentModels()


class InferenceControl:
    """Handles for pausing a FakeAdapter inside _do_infer, deterministically."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entered = threading.Event()   # set once an inference is inside
        self.release = threading.Event()   # set to let it finish
        self.active = 0
        self.max_active = 0


#: package id -> InferenceControl, so a manifest (JSON) can still reach one.
INFERENCE_CONTROL: Dict[str, InferenceControl] = {}


def control_for(package_id: str) -> InferenceControl:
    ctl = InferenceControl()
    INFERENCE_CONTROL[package_id] = ctl
    return ctl


class FakeAdapter(ModelAdapter):
    """A model that returns whatever the manifest tells it to.

    ``adapter_options``:
        counts        {"class": n}  what to "detect"
        with_bbox     emit geometry (default True)
        fail_load     raise on load, to test that a failed switch is contained
        fail_warmup   raise on warmup, to test the switch verification gate
        load_delay    seconds to spend inside load, to widen races
        blocking      pause inside infer until the package's control releases it
    """

    name = "fake"

    def __init__(self, package) -> None:
        super().__init__(package)
        self.load_calls = 0
        self.unload_calls = 0
        self.warmup_calls = 0
        self.infer_calls = 0
        # Stands in for the real model handle: if inference ever runs while
        # this is None, an unload tore the model out mid-call.
        self._model_handle = None

    def _do_load(self) -> None:
        self.load_calls += 1
        if self.options.get("fail_load"):
            raise AdapterError("simulated load failure for package '%s'" % self.package.id)
        delay = self.options.get("load_delay")
        if delay:
            time.sleep(float(delay))
        self._model_handle = object()
        RESIDENT.acquire()

    def _do_unload(self) -> None:
        self.unload_calls += 1
        if self.options.get("fail_unload"):
            # Raise BEFORE releasing: a teardown that failed has, by
            # definition, not freed the model, and the resident counter must
            # reflect that or the test would not prove anything.
            raise AdapterError("simulated unload failure for package '%s'" % self.package.id)
        self._model_handle = None
        RESIDENT.release()

    def _do_warmup(self) -> None:
        self.warmup_calls += 1
        if self.options.get("fail_warmup"):
            raise AdapterError("simulated warmup failure for package '%s'" % self.package.id)

    def _do_infer(self, image, conf, imgsz) -> InferenceResult:
        self.infer_calls += 1
        if self._model_handle is None:
            raise AssertionError(
                "inference ran with no model loaded for package '%s'" % self.package.id)

        ctl = INFERENCE_CONTROL.get(self.package.id) if self.options.get("blocking") else None
        if ctl is not None:
            with ctl.lock:
                ctl.active += 1
                ctl.max_active = max(ctl.max_active, ctl.active)
            try:
                ctl.entered.set()
                ctl.release.wait(timeout=15.0)
                if self._model_handle is None:
                    raise AssertionError(
                        "model was unloaded while inference was still running")
            finally:
                with ctl.lock:
                    ctl.active -= 1

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


class ModellessAdapter(FakeAdapter):
    """An adapter that needs no model file — e.g. a remote or rule-based one."""

    name = "modelless"
    requires_model_file = False

    def _do_load(self) -> None:
        self.load_calls += 1
        self._model_handle = object()
        RESIDENT.acquire()


class ClassListingAdapter(FakeAdapter):
    """Publishes a fixed model class list, like a real YOLO model does.

    Needed to exercise the compatibility check, which is deliberately skipped
    for adapters that cannot say what their model can recognise.
    """

    name = "class_listing"

    def _describe(self):
        info = super()._describe()
        info.class_names = [str(c) for c in self.options.get("model_classes", [])]
        return info


register_adapter("fake", FakeAdapter, replace=True)
register_adapter("counts_only", CountsOnlyAdapter, replace=True)
register_adapter("modelless", ModellessAdapter, replace=True, requires_model_file=False)
register_adapter("class_listing", ClassListingAdapter, replace=True)


@pytest.fixture(autouse=True)
def _reset_resident_counters():
    """Every test starts from a clean resident-model count."""
    RESIDENT.reset()
    INFERENCE_CONTROL.clear()
    yield
    INFERENCE_CONTROL.clear()


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
        # Default to something the class weights can actually price — a package
        # whose default standards reference a class with no weight is invalid,
        # which is the behaviour under test elsewhere, not a fixture default.
        standards = {cls: 1 for cls in class_weights}

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

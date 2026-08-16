"""Ultralytics YOLO adapter.

This is the ONLY module in the project allowed to import ultralytics or to know
what a YOLO ``Results`` object looks like.  Everything above it sees plain
``Detection`` / ``InferenceResult`` values.

Manifest options (``adapter_options``)::

    "task"            "detect" | "segment" | "classify"  (required for .onnx)
    "strict"          true  -> never fall back to fallback_model_file
    "include_masks"   true  -> attach segmentation masks to Detection.mask
                              (off by default: masks are large and the
                               inventory platform does not use them)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, List, Optional

from app.inference.base import AdapterError, ModelAdapter
from app.inference.types import Detection, InferenceResult, ModelInfo, counts_from_detections

logger = logging.getLogger(__name__)


class UltralyticsAdapter(ModelAdapter):
    name = "ultralytics"

    def __init__(self, package) -> None:
        super().__init__(package)
        self._model = None
        self._backend = ""          # "onnx" | "pt" | other suffix
        self._model_path: Optional[Path] = None
        self._class_names: List[str] = []

    # ── loading ───────────────────────────────────────────────────────────

    def _select_model_file(self) -> Path:
        pkg = self.package
        strict = bool(self.options.get("strict", False))
        primary = pkg.model_file
        fallback = pkg.fallback_model_file

        if primary is not None and primary.exists():
            return primary
        if primary is not None:
            if strict:
                raise AdapterError(
                    "model file not found: %s (strict mode — no fallback)" % primary)
            logger.warning("[ultralytics] model file not found: %s", primary)
        if fallback is not None and fallback.exists():
            logger.warning("[ultralytics] falling back to %s", fallback)
            return fallback
        raise AdapterError(
            "no usable model file for package '%s' — tried %s%s"
            % (pkg.id, primary,
               (" and %s" % fallback) if fallback is not None else ""))

    def _do_load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("ultralytics is not installed: %s" % exc)

        path = self._select_model_file()
        suffix = path.suffix.lower().lstrip(".")
        task = self.options.get("task")

        kwargs = {}
        if task:
            # Ultralytics cannot infer the task from an exported .onnx graph;
            # for .pt it can, but an explicit task is still honoured.
            kwargs["task"] = str(task)

        try:
            self._model = YOLO(str(path), **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("YOLO(%s) failed: %s" % (path, exc))

        self._model_path = path
        self._backend = "onnx" if suffix == "onnx" else ("pt" if suffix == "pt" else suffix)
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            self._class_names = [str(names[k]) for k in sorted(names)]
        elif isinstance(names, (list, tuple)):
            self._class_names = [str(n) for n in names]
        logger.info("[ultralytics] loaded %s  backend=%s  task=%s  classes=%d",
                    path, self._backend, task or "auto", len(self._class_names))

    def _do_unload(self) -> None:
        self._model = None
        self._model_path = None
        self._class_names = []

    def _do_warmup(self) -> None:
        import numpy as np

        # Warm up at the real inference size.  A tiny dummy frame is cheaper but
        # useless: it does not exercise the same kernels, and a segmentation
        # graph exported at 640 fails outright on a 64x64 input.
        size = self.package.image_size or 640
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        kwargs = {"conf": 0.25, "verbose": False}
        if self.package.image_size:
            kwargs["imgsz"] = self.package.image_size
        self._model(dummy, **kwargs)

    # ── inference ─────────────────────────────────────────────────────────

    def _do_infer(self, image: Any, conf: float, imgsz: Optional[int]) -> InferenceResult:
        if self._model is None:
            raise AdapterError("model is not loaded")

        kwargs = {"conf": conf, "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz

        t0 = time.perf_counter()
        results = self._model(image, **kwargs)
        infer_ms = (time.perf_counter() - t0) * 1000

        detections = self._extract(results)
        return InferenceResult(
            counts=counts_from_detections(detections),
            detections=detections,
            inference_ms=infer_ms,
            model_info=self.model_info,
            metadata={"backend": self._backend},
        )

    def _extract(self, results) -> List[Detection]:
        """Translate an ultralytics Results object into plain Detections.

        Every YOLO-specific attribute access is confined to this method.
        """
        detections: List[Detection] = []
        if not results:
            return detections
        result = results[0]

        names = getattr(result, "names", None) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        include_masks = bool(self.options.get("include_masks", False))
        masks = getattr(result, "masks", None) if include_masks else None
        mask_data = getattr(masks, "data", None) if masks is not None else None

        for index, box in enumerate(boxes):
            try:
                class_id = int(box.cls.item())
            except Exception:  # noqa: BLE001 - a malformed row must not kill the frame
                continue
            class_name = str(names.get(class_id, class_id)) if isinstance(names, dict) \
                else str(class_id)
            try:
                confidence = float(box.conf.item())
            except Exception:  # noqa: BLE001
                confidence = 0.0

            bbox = None
            try:
                xyxy = box.xyxy.squeeze().tolist()
                if isinstance(xyxy, float):
                    xyxy = box.xyxy.flatten().tolist()
                if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
                    bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
            except Exception:  # noqa: BLE001 - geometry is optional by contract
                bbox = None

            mask = None
            if mask_data is not None and index < len(mask_data):
                mask = mask_data[index]

            detections.append(Detection(
                class_name=class_name,
                class_id=class_id,
                confidence=confidence,
                bbox_xyxy=bbox,
                mask=mask,
            ))
        return detections

    # ── identity ──────────────────────────────────────────────────────────

    def _describe(self) -> ModelInfo:
        info = super()._describe()
        info.backend = self._backend
        info.class_names = list(self._class_names)
        if self._model_path is not None:
            info.model_file = str(self._model_path)
            info.model_file_exists = self._model_path.exists()
        info.extra = {
            "task": self.options.get("task", ""),
            "strict": bool(self.options.get("strict", False)),
        }
        return info

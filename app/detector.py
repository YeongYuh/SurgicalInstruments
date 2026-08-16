"""Backward-compatible detector facade.

The web platform no longer uses this class — it goes through ModelManager and a
ModelAdapter.  ``app/main.py`` (the standalone image/webcam CLI) still does, so
this is kept as a thin wrapper over UltralyticsAdapter.

All ultralytics-specific behaviour lives in
``app/inference/adapters/ultralytics_adapter.py``; nothing here imports YOLO.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.inference.adapters.ultralytics_adapter import UltralyticsAdapter
from app.inference.package import ModelPackage
from app.inference.types import counts_from_detections

logger = logging.getLogger(__name__)

_DEBUG_TIMING = os.environ.get("DEBUG_INFERENCE_TIMING", "false").lower() == "true"


def _synthetic_package(
    model_path: Path,
    backend: str,
    onnx_path: Optional[Path],
    onnx_task: str,
) -> ModelPackage:
    """Wrap loose CLI arguments in the ModelPackage shape the adapter expects."""
    if backend == "onnx" and onnx_path is not None:
        primary, fallback = onnx_path, model_path
    else:
        primary, fallback = model_path, None
    return ModelPackage(
        package_id="cli",
        display_name="CLI",
        department="",
        adapter="ultralytics",
        root=model_path.parent,
        manifest_path=model_path.parent / "manifest.json",
        project_root=Path.cwd(),
        model_file=primary,
        fallback_model_file=fallback,
        adapter_options={
            "task": onnx_task,
            "strict": os.environ.get("STRICT_DETECTOR_BACKEND", "false").lower() == "true",
        },
    )


class SurgicalInstrumentDetector:
    """Legacy API: ``predict()`` -> (raw_result, list_of_detection_dicts)."""

    def __init__(
        self,
        model_path: str,
        backend: str = "pt",
        onnx_path: Optional[str] = None,
        onnx_task: str = "segment",
    ):
        self.model_path = Path(model_path)
        self.backend = backend
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.onnx_task = onnx_task
        self._adapter = UltralyticsAdapter(
            _synthetic_package(self.model_path, backend, self.onnx_path, onnx_task)
        )

    @property
    def _effective_backend(self) -> Optional[str]:
        info = self._adapter.model_info
        return info.backend or None

    def _load_model(self) -> None:
        self._adapter.load()

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        imgsz: Optional[int] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Accept a file path (str/Path) or a BGR numpy frame.

        The first element of the tuple is None — the raw runtime result object
        is deliberately not exposed any more, and no caller used it.
        """
        t_start = time.perf_counter()
        if isinstance(source, Path):
            source = str(source)
        result = self._adapter.infer(source, conf=conf, imgsz=imgsz)
        detections = result.detection_dicts()

        if _DEBUG_TIMING:
            logger.info(
                "[Timing/%s] total=%.0f ms  model_call=%.0f ms  dets=%d",
                self._effective_backend,
                (time.perf_counter() - t_start) * 1000,
                result.inference_ms,
                len(detections),
            )
        return None, detections

    def count_instruments(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in detections:
            name = det["class_name"] if isinstance(det, dict) else det.class_name
            counts[name] = counts.get(name, 0) + 1
        return counts


__all__ = ["SurgicalInstrumentDetector", "counts_from_detections"]

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SurgicalInstrumentDetector:
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
        self._model = None
        self._effective_backend: Optional[str] = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        if self.backend == "onnx":
            if self.onnx_path and self.onnx_path.exists():
                try:
                    self._model = YOLO(str(self.onnx_path), task=self.onnx_task)
                    self._effective_backend = "onnx"
                    logger.info("[Detector] Loaded ONNX model: %s (task=%s)", self.onnx_path, self.onnx_task)
                    return
                except Exception as exc:
                    strict = os.environ.get("STRICT_DETECTOR_BACKEND", "false").lower() == "true"
                    if strict:
                        raise
                    logger.warning("[Detector] ONNX load failed (%s) — falling back to .pt", exc)
            else:
                logger.warning("[Detector] ONNX model not found at %s — falling back to .pt", self.onnx_path)

        self._model = YOLO(str(self.model_path))
        self._effective_backend = "pt"
        logger.info("[Detector] Loaded PT model: %s", self.model_path)

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = 0.25,
    ) -> tuple[Any, list[dict]]:
        """Accept a file path (str/Path) or a BGR numpy frame."""
        self._load_model()
        results = self._model(source, conf=conf, verbose=False)
        result = results[0]

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = float(box.conf.item())
                xyxy = box.xyxy.squeeze().tolist()
                if isinstance(xyxy, float):
                    xyxy = box.xyxy.flatten().tolist()
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": xyxy,
                })

        return result, detections

    def count_instruments(self, detections: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for det in detections:
            name = det["class_name"]
            counts[name] = counts.get(name, 0) + 1
        return counts

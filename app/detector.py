from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np


class SurgicalInstrumentDetector:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))

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
                # Ensure xyxy is always a flat list of 4 values
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

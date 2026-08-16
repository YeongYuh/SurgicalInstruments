"""Model-agnostic inference contract.

Nothing in this module may import a specific ML runtime (ultralytics, torch,
onnxruntime, tensorrt, ...).  These types are the *only* thing the inventory
platform (camera, routes, history, reports, weight verification) is allowed to
see.  Everything runtime-specific stays behind a ModelAdapter.

Design notes
------------
* ``counts`` is the primary result.  The platform counts instruments; it does
  not require boxes, masks, or any spatial output.
* ``bbox_xyxy`` is OPTIONAL.  A classifier or a counts-only model produces
  detections without geometry, and the platform must keep working.
* ``mask`` is a free-form slot reserved for segmentation models.  The platform
  never interprets it; only a renderer that understands the adapter would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# (x1, y1, x2, y2) in pixels of the frame handed to infer()
BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """One recognised instrument instance.

    Only ``class_name`` is required for inventory counting; ``class_id`` and
    ``confidence`` are informational, and geometry is optional.
    """

    class_name: str
    class_id: int = -1
    confidence: float = 0.0
    bbox_xyxy: Optional[BBox] = None
    mask: Optional[Any] = None

    @property
    def has_bbox(self) -> bool:
        return self.bbox_xyxy is not None

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form.

        ``xyxy`` is emitted only when geometry exists, so that consumers which
        probe for the key behave correctly for counts-only models.
        """
        out: Dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
        }
        if self.bbox_xyxy is not None:
            out["bbox_xyxy"] = list(self.bbox_xyxy)
            out["xyxy"] = list(self.bbox_xyxy)  # legacy key
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        bbox = data.get("bbox_xyxy", data.get("xyxy"))
        if bbox is not None:
            seq = list(bbox)
            bbox = (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
        return cls(
            class_name=str(data.get("class_name", "")),
            class_id=int(data.get("class_id", -1)),
            confidence=float(data.get("confidence", 0.0)),
            bbox_xyxy=bbox,
            mask=data.get("mask"),
        )


@dataclass
class ModelInfo:
    """Identity of whatever produced an InferenceResult.

    Stored in history records so a past count can always be traced back to the
    model package that produced it, even after the active package changed.
    """

    package_id: str
    display_name: str = ""
    department: str = ""
    adapter: str = ""
    model_file: str = ""
    model_file_exists: bool = False
    loaded: bool = False
    backend: str = ""            # adapter-reported effective backend (e.g. "onnx")
    class_names: List[str] = field(default_factory=list)
    warmup_error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "department": self.department,
            "adapter": self.adapter,
            "model_file": self.model_file,
            "model_file_exists": self.model_file_exists,
            "loaded": self.loaded,
            "backend": self.backend,
            "class_names": list(self.class_names),
            "warmup_error": self.warmup_error,
            "extra": dict(self.extra),
        }

    def identity(self) -> Dict[str, Any]:
        """Compact identity suitable for embedding in a history record."""
        return {
            "package_id": self.package_id,
            "display_name": self.display_name,
            "department": self.department,
            "adapter": self.adapter,
            "model_file": self.model_file,
            "backend": self.backend,
        }


@dataclass
class InferenceResult:
    """What every adapter returns.  The platform consumes nothing else."""

    counts: Dict[str, int] = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    model_info: Optional[ModelInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_geometry(self) -> bool:
        return any(d.has_bbox for d in self.detections)

    def detection_dicts(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self.detections]


def counts_from_detections(detections: Sequence[Detection]) -> Dict[str, int]:
    """Tally detections by class name.  Shared by every adapter."""
    counts: Dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts

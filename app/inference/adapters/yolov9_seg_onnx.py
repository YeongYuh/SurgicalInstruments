"""YOLOv9 GELAN segmentation, exported to ONNX.

Why this exists
---------------
The demo instrument model is a WongKinYiu/yolov9 ``gelan-c-seg`` checkpoint.
Its ``.pt`` pickles ``models.common.*`` / ``models.yolo.*`` from that repository,
so ``ultralytics.YOLO()`` cannot open it at all — and no amount of guessing from
the ``.onnx`` extension would have made the exported graph work either, because
its output layout is not Ultralytics'.  This adapter owns that layout, and
nothing above it changes.

Graph contract (produced by yolov9 ``export.py --include onnx``)::

    input   images   (1, 3, S, S)      float32, RGB, 0-1, letterboxed
    output0          (1, 4+nc+nm, N)   xywh (input pixels) + class scores + mask coeffs
    output1          (1, nm, S/4, S/4) mask prototypes

Class scores are already activated and there is no separate objectness term —
the confidence of a candidate is simply its highest class score, matching the
reference implementation's ``prediction[:, 4:4+nc].amax(1)``.

Mask prototypes are read but not decoded: the inventory platform counts
instances, and turning 8400 prototype combinations into full-resolution masks
on a Jetson CPU would cost more than the detection itself.  ``include_masks``
exists for a future renderer.

``adapter_options``::

    conf         default confidence floor (manifest inference.confidence wins)
    iou          NMS IoU threshold                       (default 0.45)
    max_det      cap on detections per frame             (default 300)
    providers    onnxruntime execution providers         (default CPU)
    intra_op_threads / inter_op_threads                  (default: let ORT decide)
    include_masks  keep the raw prototypes on Detection.mask (default False)
    class_names  explicit list, only if the graph carries no names metadata
"""

from __future__ import annotations

import ast
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.inference.base import AdapterError, ModelAdapter
from app.inference.types import Detection, InferenceResult, ModelInfo, counts_from_detections

logger = logging.getLogger(__name__)

DEFAULT_IOU = 0.45
DEFAULT_MAX_DET = 300
#: Grey used by the reference letterbox; the model was trained with it.
LETTERBOX_PAD = 114


def _letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to a square, as the training did.

    Returns the padded image plus the scale and offsets needed to map boxes
    back onto the original frame.
    """
    import cv2

    height, width = image.shape[:2]
    scale = min(size / float(height), size / float(width))
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Plain greedy NMS on xyxy boxes.  numpy only — no torch at runtime."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


class Yolov9SegOnnxAdapter(ModelAdapter):
    name = "yolov9_seg_onnx"
    requires_model_file = True

    def __init__(self, package) -> None:
        super().__init__(package)
        self._session = None
        self._input_name = ""
        self._output_names: List[str] = []
        self._imgsz = 640
        self._class_names: List[str] = []
        self._num_masks = 0

    # ── loading ───────────────────────────────────────────────────────────

    def _do_load(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("onnxruntime is not installed: %s" % exc)

        path = self.package.resolved_model_file()
        if path is None:
            raise AdapterError(
                "no usable model file for package '%s' — tried %s"
                % (self.package.id, self.package.model_file))
        if path.suffix.lower() != ".onnx":
            raise AdapterError(
                "%s expects an .onnx graph, got %s. Export the yolov9 checkpoint "
                "first: python export.py --weights <best.pt> --include onnx "
                "--imgsz 640 640 --batch-size 1" % (self.name, path.name))

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        intra = self.options.get("intra_op_threads")
        if intra:
            options.intra_op_num_threads = int(intra)
        inter = self.options.get("inter_op_threads")
        if inter:
            options.inter_op_num_threads = int(inter)

        providers = self.options.get("providers") or ["CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(str(path), options, providers=list(providers))
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("cannot open ONNX graph %s: %s" % (path, exc))

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise AdapterError("expected exactly one graph input, got %d" % len(inputs))
        self._input_name = inputs[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        if len(self._output_names) < 1:
            raise AdapterError("ONNX graph produces no outputs")

        shape = inputs[0].shape
        # (1, 3, S, S) — a fixed square is what the reference export produces.
        size = shape[2] if len(shape) == 4 else None
        if isinstance(size, int) and size > 0:
            self._imgsz = size
        elif self.package.image_size:
            self._imgsz = int(self.package.image_size)
        else:
            raise AdapterError(
                "graph has a dynamic input size and the manifest sets no "
                "inference.image_size — cannot decide what to letterbox to")

        self._class_names = self._resolve_class_names()
        if not self._class_names:
            raise AdapterError(
                "no class names: the graph carries no 'names' metadata and the "
                "manifest sets no adapter_options.class_names")

        # Derive the mask-coefficient count from the graph itself so the shape
        # contract is checked once at load rather than per frame.
        out_shape = self._session.get_outputs()[0].shape
        if len(out_shape) == 3 and isinstance(out_shape[1], int):
            self._num_masks = int(out_shape[1]) - 4 - len(self._class_names)
            if self._num_masks < 0:
                raise AdapterError(
                    "graph output0 has %d channels, too few for 4 box + %d classes"
                    % (out_shape[1], len(self._class_names)))

        logger.info("[yolov9_seg_onnx] loaded %s  imgsz=%d  classes=%d  masks=%d  "
                    "providers=%s", path.name, self._imgsz, len(self._class_names),
                    self._num_masks, self._session.get_providers())

    def _resolve_class_names(self) -> List[str]:
        """Prefer the graph's own metadata; the model is the source of truth."""
        override = self.options.get("class_names")
        if override:
            return [str(n) for n in override]

        try:
            meta = self._session.get_modelmeta().custom_metadata_map or {}
        except Exception:  # noqa: BLE001
            meta = {}
        raw = meta.get("names")
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError("graph 'names' metadata is not readable: %s" % exc)
        if isinstance(parsed, dict):
            return [str(parsed[key]) for key in sorted(parsed)]
        if isinstance(parsed, (list, tuple)):
            return [str(n) for n in parsed]
        raise AdapterError("graph 'names' metadata has unexpected type %s" % type(parsed))

    def _do_unload(self) -> None:
        self._session = None
        self._input_name = ""
        self._output_names = []
        self._class_names = []

    def _do_warmup(self) -> None:
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._do_infer(blank, conf=0.25, imgsz=None)

    # ── inference ─────────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        padded, scale, dx, dy = _letterbox(image, self._imgsz)
        # BGR -> RGB, HWC -> CHW, 0-1, batched.
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1])
        tensor = tensor.astype(np.float32) / 255.0
        return tensor[None], scale, dx, dy

    def _do_infer(self, image: Any, conf: float, imgsz: Optional[int]) -> InferenceResult:
        if self._session is None:
            raise AdapterError("model is not loaded")
        if isinstance(image, (str, bytes)):
            import cv2

            decoded = cv2.imread(str(image))
            if decoded is None:
                raise AdapterError("cannot read image: %s" % image)
            image = decoded
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise AdapterError("expected a BGR numpy frame, got %r" % type(image))
        if imgsz is not None and int(imgsz) != self._imgsz:
            # The graph's input is fixed at export time; silently letterboxing to
            # a different size would feed the model something it never saw.
            logger.debug("[yolov9_seg_onnx] ignoring imgsz=%s; graph is fixed at %d",
                         imgsz, self._imgsz)

        tensor, scale, dx, dy = self._preprocess(image)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})

        detections = self._postprocess(
            outputs, conf=conf, height=image.shape[0], width=image.shape[1],
            scale=scale, dx=dx, dy=dy)
        return InferenceResult(
            counts=counts_from_detections(detections),
            detections=detections,
            model_info=self.model_info,
            metadata={"backend": "onnx", "imgsz": self._imgsz},
        )

    def _postprocess(self, outputs, *, conf: float, height: int, width: int,
                     scale: float, dx: int, dy: int) -> List[Detection]:
        predictions = np.asarray(outputs[0])
        if predictions.ndim != 3:
            raise AdapterError("unexpected output0 rank %d" % predictions.ndim)
        # (1, 4+nc+nm, N) -> (N, 4+nc+nm)
        predictions = predictions[0].T

        num_classes = len(self._class_names)
        channels = predictions.shape[1]
        num_masks = channels - 4 - num_classes
        if num_masks < 0:
            raise AdapterError(
                "graph output has %d channels, too few for 4 box + %d classes"
                % (channels, num_classes))
        self._num_masks = num_masks

        scores_all = predictions[:, 4:4 + num_classes]
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

        keep = scores >= float(conf)
        if not np.any(keep):
            return []
        boxes_xywh = predictions[keep, :4]
        scores = scores[keep]
        class_ids = class_ids[keep]
        coeffs = predictions[keep, 4 + num_classes:] if num_masks else None

        # xywh (letterboxed pixels) -> xyxy in the ORIGINAL frame
        cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = (cx - w / 2.0 - dx) / scale
        y1 = (cy - h / 2.0 - dy) / scale
        x2 = (cx + w / 2.0 - dx) / scale
        y2 = (cy + h / 2.0 - dy) / scale
        boxes = np.stack([
            np.clip(x1, 0, width), np.clip(y1, 0, height),
            np.clip(x2, 0, width), np.clip(y2, 0, height),
        ], axis=1)

        iou = float(self.options.get("iou", DEFAULT_IOU))
        max_det = int(self.options.get("max_det", DEFAULT_MAX_DET))
        include_masks = bool(self.options.get("include_masks", False))
        protos = np.asarray(outputs[1]) if (include_masks and len(outputs) > 1) else None

        # Class-aware NMS: offsetting by class keeps two different instruments
        # that genuinely overlap from suppressing one another.
        offsets = class_ids.astype(np.float32) * 8192.0
        shifted = boxes + offsets[:, None]
        order = _nms(shifted, scores, iou)[:max_det]

        detections: List[Detection] = []
        for index in order:
            class_id = int(class_ids[index])
            mask = None
            if protos is not None and coeffs is not None:
                mask = {"coeffs": coeffs[index], "protos": protos}
            detections.append(Detection(
                class_name=self._class_names[class_id],
                class_id=class_id,
                confidence=float(scores[index]),
                bbox_xyxy=(float(boxes[index, 0]), float(boxes[index, 1]),
                           float(boxes[index, 2]), float(boxes[index, 3])),
                mask=mask,
            ))
        return detections

    # ── identity ──────────────────────────────────────────────────────────

    def _describe(self) -> ModelInfo:
        info = super()._describe()
        info.backend = "onnx"
        info.class_names = list(self._class_names)
        resolved = self.package.resolved_model_file()
        if resolved is not None:
            info.model_file = str(resolved)
            info.model_file_exists = resolved.exists()
        info.extra = {
            "family": "yolov9-gelan-seg",
            "imgsz": self._imgsz,
            "mask_channels": self._num_masks,
            "iou": float(self.options.get("iou", DEFAULT_IOU)),
            "providers": (list(self._session.get_providers())
                          if self._session is not None else []),
        }
        return info

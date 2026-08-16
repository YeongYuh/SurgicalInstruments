"""The YOLOv9 GELAN segmentation adapter.

The demo checkpoint is a WongKinYiu/yolov9 model whose ONNX output layout is not
Ultralytics'.  The pure functions (letterbox mapping, NMS) are tested directly;
the end-to-end path is tested against the real exported graph when it is present
on the machine, and skipped otherwise so the suite stays green on a dev box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import app.config as config
from app.inference.adapters.yolov9_seg_onnx import (
    LETTERBOX_PAD,
    Yolov9SegOnnxAdapter,
    _letterbox,
    _nms,
)
from app.inference.base import AdapterError
from app.inference.package import ModelPackage
from app.inference.registry import adapter_requires_model_file, get_adapter_class, has_adapter

DEMO_ONNX = Path(config.PROJECT_ROOT) / "models" / "demo" / "best.onnx"


# ── registry ────────────────────────────────────────────────────────────────

def test_adapter_is_registered():
    assert has_adapter("yolov9_seg_onnx")
    assert get_adapter_class("yolov9_seg_onnx") is Yolov9SegOnnxAdapter
    assert adapter_requires_model_file("yolov9_seg_onnx") is True


# ── letterbox: the mapping boxes are scaled back through ────────────────────

@pytest.mark.parametrize("shape", [(480, 640), (640, 480), (640, 640), (100, 900)])
def test_letterbox_preserves_aspect_ratio_and_pads(shape):
    image = np.full((shape[0], shape[1], 3), 200, dtype=np.uint8)
    padded, scale, dx, dy = _letterbox(image, 640)

    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(min(640 / shape[0], 640 / shape[1]))
    # The content sits centred, with the training pad colour around it.
    assert dx >= 0 and dy >= 0
    if dy > 0:
        assert (padded[0, :, :] == LETTERBOX_PAD).all()
    if dx > 0:
        assert (padded[:, 0, :] == LETTERBOX_PAD).all()


def test_letterbox_of_a_square_image_is_a_plain_resize():
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    _padded, scale, dx, dy = _letterbox(image, 640)
    assert (scale, dx, dy) == (2.0, 0, 0)


def test_box_round_trip_through_the_letterbox_mapping():
    """A box drawn in the padded frame must map back onto the original pixels."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _padded, scale, dx, dy = _letterbox(image, 640)

    original = (100.0, 50.0, 300.0, 400.0)
    padded_box = [original[0] * scale + dx, original[1] * scale + dy,
                  original[2] * scale + dx, original[3] * scale + dy]
    back = [(padded_box[0] - dx) / scale, (padded_box[1] - dy) / scale,
            (padded_box[2] - dx) / scale, (padded_box[3] - dy) / scale]

    assert back == pytest.approx(list(original))


# ── NMS ─────────────────────────────────────────────────────────────────────

def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([
        [0, 0, 100, 100],
        [5, 5, 105, 105],      # heavy overlap with the first
        [500, 500, 600, 600],  # disjoint
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = _nms(boxes, scores, 0.45)

    assert keep == [0, 2]


def test_nms_keeps_everything_when_nothing_overlaps():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.5, 0.9], dtype=np.float32)
    assert sorted(_nms(boxes, scores, 0.45)) == [0, 1]


def test_nms_on_an_empty_set():
    assert _nms(np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32), 0.45) == []


def test_nms_returns_boxes_in_descending_score_order():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110], [200, 200, 210, 210]],
                     dtype=np.float32)
    scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)
    assert _nms(boxes, scores, 0.45) == [1, 2, 0]


# ── the adapter refuses what it cannot decode ───────────────────────────────

def _package(tmp_path, model_name="model.pt", options=None):
    (tmp_path / model_name).write_bytes(b"x")
    return ModelPackage(
        package_id="demo", display_name="Demo", department="demo",
        adapter="yolov9_seg_onnx", root=tmp_path,
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path,
        model_file=tmp_path / model_name, adapter_options=options or {},
        confidence=0.25, image_size=640)


def test_a_torch_checkpoint_is_refused_with_the_export_command(tmp_path):
    """The .pt cannot be read here — say how to produce what can."""
    adapter = Yolov9SegOnnxAdapter(_package(tmp_path, "best.pt"))
    with pytest.raises(AdapterError) as excinfo:
        adapter.load()
    message = str(excinfo.value)
    assert "expects an .onnx graph" in message
    assert "export.py" in message


def test_a_missing_model_file_is_refused(tmp_path):
    package = ModelPackage(
        package_id="demo", display_name="Demo", department="demo",
        adapter="yolov9_seg_onnx", root=tmp_path,
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path,
        model_file=tmp_path / "absent.onnx", confidence=0.25, image_size=640)
    with pytest.raises(AdapterError) as excinfo:
        Yolov9SegOnnxAdapter(package).load()
    assert "no usable model file" in str(excinfo.value)


# ── end to end against the real exported graph ──────────────────────────────

requires_graph = pytest.mark.skipif(
    not DEMO_ONNX.exists(),
    reason="demo ONNX not present; export it with yolov9 export.py")


@pytest.fixture(scope="module")
def demo_adapter():
    if not DEMO_ONNX.exists():
        pytest.skip("demo ONNX not present")
    package = ModelPackage(
        package_id="demo", display_name="Demo 手術器械組", department="demo",
        adapter="yolov9_seg_onnx", root=DEMO_ONNX.parent,
        manifest_path=DEMO_ONNX.parent / "manifest.json",
        project_root=Path(config.PROJECT_ROOT),
        model_file=DEMO_ONNX, adapter_options={"iou": 0.45},
        confidence=0.25, image_size=640)
    adapter = Yolov9SegOnnxAdapter(package)
    adapter.load()
    yield adapter
    adapter.unload()


@requires_graph
def test_class_names_come_from_the_graph_metadata(demo_adapter):
    """The model is the source of truth for its own class list."""
    info = demo_adapter.model_info
    assert len(info.class_names) == 17
    assert info.class_names[0] == "Adson-Smooth-Tissue-Forceps"
    assert "Towel-Clamp" in info.class_names
    assert info.backend == "onnx"
    assert info.extra["family"] == "yolov9-gelan-seg"
    assert info.extra["imgsz"] == 640
    assert info.extra["mask_channels"] == 32


@requires_graph
def test_class_names_match_the_package_class_weights(demo_adapter):
    import json

    weights = json.loads(
        (Path(config.PROJECT_ROOT) / "model_packages/demo/class_weight.json")
        .read_text(encoding="utf-8"))
    assert set(demo_adapter.model_info.class_names) == set(weights)


@requires_graph
def test_inference_returns_detections_with_geometry(demo_adapter):
    import cv2

    image = cv2.imread(str(Path(config.PROJECT_ROOT) / "input" / "test_image.jpg"))
    assert image is not None

    result = demo_adapter.infer(image, conf=0.25)

    assert result.counts == {k: v for k, v in result.counts.items() if v > 0}
    assert sum(result.counts.values()) == len(result.detections)
    for det in result.detections:
        assert det.class_name in demo_adapter.model_info.class_names
        assert 0.25 <= det.confidence <= 1.0
        assert det.has_bbox
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 < x2 <= image.shape[1]
        assert 0 <= y1 < y2 <= image.shape[0]
    assert result.inference_ms > 0
    assert result.metadata["backend"] == "onnx"


@requires_graph
def test_a_high_threshold_suppresses_everything(demo_adapter):
    import cv2

    image = cv2.imread(str(Path(config.PROJECT_ROOT) / "input" / "test_image.jpg"))
    result = demo_adapter.infer(image, conf=0.999)
    assert result.counts == {}
    assert result.detections == []


@requires_graph
def test_a_blank_frame_is_a_valid_zero_detection_result(demo_adapter):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = demo_adapter.infer(blank, conf=0.5)
    assert isinstance(result.counts, dict)
    assert len(result.detections) == sum(result.counts.values())


@requires_graph
def test_boxes_land_inside_a_non_square_frame(demo_adapter):
    """Letterbox offsets are undone, not just the scale."""
    import cv2

    image = cv2.imread(str(Path(config.PROJECT_ROOT) / "input" / "test_image.jpg"))
    tall = cv2.resize(image, (480, 720))

    result = demo_adapter.infer(tall, conf=0.25)

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 <= 480 and 0 <= x2 <= 480
        assert 0 <= y1 <= 720 and 0 <= y2 <= 720

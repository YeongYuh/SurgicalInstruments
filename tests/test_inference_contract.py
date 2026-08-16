"""The model-agnostic contract: counts are mandatory, geometry is not."""

from __future__ import annotations

import numpy as np
import pytest

from app.inference.types import Detection, InferenceResult, counts_from_detections
from app.visualizer import draw_detections
from conftest import make_manager, write_package


# ── 13. a counts-only model is a first-class citizen ────────────────────────

def test_counts_only_model_produces_a_usable_result(tmp_path):
    manager = make_manager(tmp_path)
    write_package(tmp_path / "model_packages", "classifier",
                  adapter="counts_only",
                  class_weights={"forceps": 4.0, "clamp": 2.0},
                  standards={"forceps": 3, "clamp": 1},
                  adapter_options={"counts": {"forceps": 3, "clamp": 1}})
    manager.activate("classifier")

    result = manager.infer(object())

    assert result.counts == {"forceps": 3, "clamp": 1}
    assert result.has_geometry is False
    assert all(d.bbox_xyxy is None for d in result.detections)


def test_counts_only_result_still_drives_inventory_and_weight(tmp_path):
    """No boxes must not mean no inventory: the whole platform still works."""
    from app.weight_verification import compute_weight_verification
    from app.scale_sample import constant_sample

    manager = make_manager(tmp_path)
    write_package(tmp_path / "model_packages", "classifier",
                  adapter="counts_only",
                  class_weights={"forceps": 4.0},
                  standards={"forceps": 3},
                  adapter_options={"counts": {"forceps": 3}})
    manager.activate("classifier")
    state = manager.state()
    result = manager.infer(object())

    # inventory comparison
    missing = {cls: state.standards[cls] - result.counts.get(cls, 0)
               for cls in state.standards}
    assert missing == {"forceps": 0}

    # weight verification: 3 × 4 g = 12 g
    wv = compute_weight_verification(state.standards, state.class_weights,
                                     constant_sample(12.0), tolerance=0.5)
    assert wv["expected"] == 12.0
    assert wv["ready"] is True
    assert wv["passed"] is True


# ── 14. Detection without geometry ──────────────────────────────────────────

def test_detection_without_bbox_is_valid():
    det = Detection(class_name="forceps", class_id=2, confidence=0.7)

    assert det.has_bbox is False
    payload = det.to_dict()
    assert payload["class_name"] == "forceps"
    # The legacy 'xyxy' key must be ABSENT, not None — consumers probe for it.
    assert "xyxy" not in payload
    assert "bbox_xyxy" not in payload


def test_detection_with_bbox_emits_both_keys():
    det = Detection(class_name="forceps", class_id=2, confidence=0.7,
                    bbox_xyxy=(1.0, 2.0, 3.0, 4.0))
    payload = det.to_dict()
    assert payload["xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]


def test_detection_roundtrips_through_dict():
    original = Detection("forceps", 2, 0.7, (1.0, 2.0, 3.0, 4.0))
    assert Detection.from_dict(original.to_dict()) == original

    bare = Detection("forceps", 2, 0.7)
    assert Detection.from_dict(bare.to_dict()) == bare


def test_counts_from_detections_tallies_by_name():
    detections = [
        Detection("forceps", 0, 0.9),
        Detection("forceps", 0, 0.8),
        Detection("clamp", 1, 0.7),
    ]
    assert counts_from_detections(detections) == {"forceps": 2, "clamp": 1}


def test_inference_result_defaults_are_empty_not_none():
    result = InferenceResult()
    assert result.counts == {}
    assert result.detections == []
    assert result.has_geometry is False


# ── 15. the visualizer must survive missing geometry ────────────────────────

def _blank_image():
    return np.zeros((80, 120, 3), dtype=np.uint8)


def test_draw_detections_skips_detections_without_bbox():
    image = _blank_image()
    annotated = draw_detections(image, [Detection("forceps", 0, 0.9)])

    assert annotated.shape == image.shape
    # Nothing was drawn, but nothing crashed either.
    assert not annotated.any()


def test_draw_detections_draws_when_geometry_exists():
    image = _blank_image()
    annotated = draw_detections(
        image, [Detection("forceps", 0, 0.9, (10.0, 20.0, 60.0, 70.0))])
    assert annotated.any()


def test_draw_detections_handles_mixed_and_legacy_shapes():
    image = _blank_image()
    detections = [
        Detection("forceps", 0, 0.9, (10.0, 20.0, 60.0, 70.0)),   # dataclass, boxed
        Detection("clamp", 1, 0.8),                                # dataclass, no box
        {"class_name": "gauze", "class_id": 2, "confidence": 0.6,
         "xyxy": [5, 5, 40, 40]},                                  # legacy dict
        {"class_name": "swab", "class_id": 3, "confidence": 0.5},  # legacy, no box
    ]
    annotated = draw_detections(image, detections)
    assert annotated.shape == image.shape
    assert annotated.any()


@pytest.mark.parametrize("detections", [None, [], [{}]])
def test_draw_detections_tolerates_degenerate_input(detections):
    image = _blank_image()
    assert draw_detections(image, detections).shape == image.shape


def test_draw_detections_tolerates_malformed_bbox():
    image = _blank_image()
    detections = [{"class_name": "x", "class_id": 0, "confidence": 0.5, "xyxy": [1, 2]}]
    assert draw_detections(image, detections).shape == image.shape


def test_overlay_runtime_info_renders_pending_without_crashing():
    """passed=None is 'not finished', and must not be painted as a failure."""
    from app.visualizer import overlay_runtime_info

    image = _blank_image()
    pending = {"passed": None, "ready": False, "expected": 10.0, "actual": None,
               "difference": None, "tolerance": 0.5}
    out = overlay_runtime_info(image, {"forceps": 1}, None, pending, "2026-01-01 00:00:00")
    assert out.shape == image.shape

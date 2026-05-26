from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# BGR palette — cycles through colours for different class ids
_PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (0, 128, 255),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes, class name, and confidence on a copy of image."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        colour = _PALETTE[det["class_id"] % len(_PALETTE)]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, 0.55, 1)
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            colour,
            -1,
        )
        cv2.putText(
            annotated, label, (x1, y1 - baseline - 2),
            _FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated


def overlay_runtime_info(
    image: np.ndarray,
    counts: dict,
    actual_weight: Optional[float],
    weight_result: dict,
    timestamp_text: str,
) -> np.ndarray:
    """Overlay instrument counts, weight, PASS/FAIL, and timestamp on frame."""
    result = image.copy()

    # Build text lines
    lines: list[tuple[str, tuple]] = []  # (text, bgr_colour)
    white = (255, 255, 255)
    green = (0, 230, 0)
    red = (0, 60, 230)
    yellow = (0, 220, 220)

    if counts:
        for name, cnt in sorted(counts.items()):
            lines.append((f"{name}: {cnt}", white))
    else:
        lines.append(("No instruments detected", yellow))

    lines.append(("", white))  # spacer

    weight_str = f"{actual_weight:.2f} g" if actual_weight is not None else "N/A"
    lines.append((f"Weight: {weight_str}", white))

    status = "PASS" if weight_result["passed"] else "FAIL"
    lines.append((f"Status: {status}", green if weight_result["passed"] else red))

    lines.append(("", white))  # spacer
    lines.append((timestamp_text, (180, 180, 180)))

    # Measure panel size
    font_scale = 0.55
    thickness = 1
    line_h = 22
    padding = 8

    max_text_w = max(
        (cv2.getTextSize(txt, _FONT, font_scale, thickness)[0][0] for txt, _ in lines if txt),
        default=100,
    )
    panel_w = max_text_w + padding * 2
    panel_h = len(lines) * line_h + padding * 2

    # Semi-transparent dark background
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, result, 0.45, 0, result)

    # Draw text
    y = padding + line_h - 4
    for text, colour in lines:
        if text:
            cv2.putText(result, text, (padding, y), _FONT, font_scale, colour, thickness, cv2.LINE_AA)
        y += line_h

    return result


def save_annotated_image(
    annotated: np.ndarray,
    source_path: str,
    output_dir: str,
) -> Path:
    """Save image-mode annotated result to output/annotated/."""
    out_dir = Path(output_dir) / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source_path).stem
    out_path = out_dir / f"{stem}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    return out_path


# Backward-compatible alias used by older callers
save_annotated = save_annotated_image


def save_frame(
    frame: np.ndarray,
    output_dir: str,
    timestamp_text: str,
) -> Path:
    """Save a webcam annotated frame to output/frames/ with timestamp filename."""
    out_dir = Path(output_dir) / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp_text.replace(":", "-").replace(" ", "_")
    out_path = out_dir / f"frame_{ts}.jpg"
    cv2.imwrite(str(out_path), frame)
    return out_path


def show_image(annotated: np.ndarray, window_title: str = "Result"):
    cv2.imshow(window_title, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

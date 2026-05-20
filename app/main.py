from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2

import app.config as config
from app.detector import SurgicalInstrumentDetector
from app.scale_reader import create_scale_reader
from app.weight_checker import check_weight
from app.visualizer import (
    draw_detections,
    overlay_runtime_info,
    save_annotated_image,
    save_frame,
    show_image,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Surgical Instrument Detection + Scale Weight Verification"
    )

    # Source
    parser.add_argument(
        "--source-type",
        choices=["image", "webcam"],
        default=config.SOURCE_TYPE,
        help="Input source type: 'image' (file) or 'webcam'",
    )
    parser.add_argument(
        "--source",
        default=config.INPUT_PATH,
        help="Path to input image (image mode only)",
    )

    # Model / detection
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to YOLO best.pt")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, help="Detection confidence threshold")

    # Weight verification
    parser.add_argument("--expected-weight", type=float, default=config.EXPECTED_WEIGHT, help="Expected total weight (g)")
    parser.add_argument("--tolerance", type=float, default=config.WEIGHT_TOLERANCE, help="Acceptable weight difference (g)")

    # Scale reader
    parser.add_argument("--scale-mode", choices=["mock", "serial"], default=config.SCALE_READER_MODE, help="Scale reader mode")
    parser.add_argument("--mock-weight", type=float, default=config.MOCK_WEIGHT, help="Weight returned in mock mode")
    parser.add_argument("--serial-port", default=config.SERIAL_PORT, help="Serial port (e.g. /dev/ttyUSB0)")
    parser.add_argument("--baudrate", type=int, default=config.SERIAL_BAUDRATE, help="Serial baud rate")
    parser.add_argument("--serial-timeout", type=float, default=config.SERIAL_TIMEOUT, help="Serial read timeout in seconds")
    parser.add_argument("--serial-retries", type=int, default=config.SERIAL_READ_RETRIES, help="Serial read retry count")

    # Webcam
    parser.add_argument("--webcam-index", type=int, default=config.WEBCAM_INDEX, help="Webcam device index")
    parser.add_argument("--webcam-width", type=int, default=config.WEBCAM_WIDTH, help="Webcam capture width")
    parser.add_argument("--webcam-height", type=int, default=config.WEBCAM_HEIGHT, help="Webcam capture height")
    parser.add_argument("--webcam-interval", type=float, default=config.WEBCAM_DETECTION_INTERVAL, help="Seconds between YOLO inferences in webcam mode")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames to output/frames (webcam mode)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N inferences (webcam mode, for testing)")

    # Display
    parser.add_argument("--show", action="store_true", help="Display annotated image/frame in a GUI window")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_scale_info(args: argparse.Namespace) -> dict:
    return {
        "mode": args.scale_mode,
        "mock_weight": args.mock_weight,
        "port": args.serial_port,
        "baudrate": args.baudrate,
    }


def print_detection_summary(counts: dict):
    print("\nYOLO detected instruments:")
    if counts:
        for name, cnt in sorted(counts.items()):
            print(f"  - {name}: {cnt}")
    else:
        print("  (none detected)")


def print_weight_summary(scale_info: dict, weight_result: dict):
    print("\nScale reader:")
    if scale_info["mode"] == "mock":
        print(f"  Mode: mock")
        print(f"  Mock weight: {scale_info['mock_weight']} g")
    else:
        print(f"  Mode: serial")
        print(f"  Port: {scale_info['port']}")
        print(f"  Baudrate: {scale_info['baudrate']}")

    if weight_result["actual_weight"] is not None:
        print(f"\nScale weight: {weight_result['actual_weight']} g")
    else:
        print("\nScale weight: (failed to read)")

    print("\nWeight verification:")
    print(f"  {'PASSED' if weight_result['passed'] else 'FAILED'}")
    print(f"  Expected:   {weight_result['expected_weight']} g")
    if weight_result["actual_weight"] is not None:
        print(f"  Actual:     {weight_result['actual_weight']} g")
        print(f"  Difference: {weight_result['difference']:.4f} g")
    else:
        print(f"  Actual:     (unavailable)")
    print(f"  Tolerance:  ±{weight_result['tolerance']} g")
    print(f"  Message:    {weight_result['message']}")


def _print_webcam_inference_line(frame_num: int, counts: dict, weight_result: dict, ts: str):
    """Compact one-liner for webcam mode so the console doesn't scroll too fast."""
    status = "PASS" if weight_result["passed"] else "FAIL"
    w = weight_result["actual_weight"]
    weight_str = f"{w:.2f} g" if w is not None else "N/A"
    instruments = ", ".join(f"{n}:{c}" for n, c in sorted(counts.items())) or "none"
    print(f"[{ts}] #{frame_num:04d} | {instruments} | Weight: {weight_str} | {status}")


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def run_image_mode(args: argparse.Namespace):
    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print(f"  Place your trained YOLO11 best.pt at: {model_path}")
        sys.exit(1)

    if not source_path.exists():
        print(f"[ERROR] Input image not found: {source_path}")
        print(f"  Place a test image at: {source_path}")
        sys.exit(1)

    # YOLO detection
    print("Loading YOLO model...")
    detector = SurgicalInstrumentDetector(str(model_path))

    print(f"Running inference on: {source_path}")
    _, detections = detector.predict(source_path, conf=args.conf)
    counts = detector.count_instruments(detections)
    print_detection_summary(counts)

    # Scale reading
    reader = create_scale_reader(
        mode=args.scale_mode,
        mock_weight=args.mock_weight,
        port=args.serial_port,
        baudrate=args.baudrate,
        timeout=args.serial_timeout,
        retries=args.serial_retries,
    )
    actual_weight = reader.read_weight()
    weight_result = check_weight(actual_weight, args.expected_weight, args.tolerance)

    scale_info = _build_scale_info(args)
    print_weight_summary(scale_info, weight_result)

    # Visualisation
    image = cv2.imread(str(source_path))
    if image is None:
        print(f"\n[WARNING] Could not read image with OpenCV: {source_path}")
        return

    annotated = draw_detections(image, detections)
    saved_path = save_annotated_image(annotated, str(source_path), config.OUTPUT_DIR)
    print(f"\nAnnotated image saved to:\n  {saved_path}")

    if args.show:
        show_image(annotated, window_title="Surgical Instrument Detection")


# ---------------------------------------------------------------------------
# Webcam mode
# ---------------------------------------------------------------------------

def run_webcam_mode(args: argparse.Namespace):
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        print(f"  Place your trained YOLO11 best.pt at: {model_path}")
        sys.exit(1)

    print("Loading YOLO model...")
    detector = SurgicalInstrumentDetector(str(model_path))

    reader = create_scale_reader(
        mode=args.scale_mode,
        mock_weight=args.mock_weight,
        port=args.serial_port,
        baudrate=args.baudrate,
        timeout=args.serial_timeout,
        retries=args.serial_retries,
    )

    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {args.webcam_index}")
        print("  Check: ls /dev/video*")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: index={args.webcam_index}, resolution={actual_w}x{actual_h}")
    print(f"YOLO inference interval: {args.webcam_interval}s")
    if args.show:
        print("Press 'q' in the window or Ctrl+C to stop.")
    else:
        print("Running headless. Press Ctrl+C to stop.")

    last_inference_time = 0.0
    last_annotated: cv2.Mat | None = None
    inference_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame from webcam. Retrying...")
                time.sleep(0.1)
                continue

            now = time.time()
            do_inference = (now - last_inference_time) >= args.webcam_interval

            if do_inference:
                _, detections = detector.predict(frame, conf=args.conf)
                counts = detector.count_instruments(detections)

                actual_weight = reader.read_weight()
                weight_result = check_weight(actual_weight, args.expected_weight, args.tolerance)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                annotated = draw_detections(frame, detections)
                annotated = overlay_runtime_info(annotated, counts, actual_weight, weight_result, ts)

                last_annotated = annotated
                last_inference_time = now
                inference_count += 1

                _print_webcam_inference_line(inference_count, counts, weight_result, ts)

                if args.save_frames:
                    saved = save_frame(annotated, config.OUTPUT_DIR, ts)
                    print(f"  Frame saved: {saved}")

                if args.max_frames is not None and inference_count >= args.max_frames:
                    print(f"\nReached --max-frames={args.max_frames}. Stopping.")
                    break

            if args.show:
                display = last_annotated if last_annotated is not None else frame
                cv2.imshow("Surgical Instrument Detection", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nUser pressed 'q'. Stopping.")
                    break
            elif not do_inference:
                # Throttle reads in headless mode to avoid busy-looping
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Stopping.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.source_type == "image":
        run_image_mode(args)
    elif args.source_type == "webcam":
        run_webcam_mode(args)
    else:
        print(f"[ERROR] Unknown source type: {args.source_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()

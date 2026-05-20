# Surgical Instrument Detection + Weight Verification

YOLO11-based surgical instrument detection with direct weight reading from an electronic scale or Arduino via serial port. Supports image-file inference and continuous webcam inference. Replaces the previous PaddleOCR workflow.

---

## Project Purpose

- Detect surgical instruments using a trained YOLO11 model.
- Image mode: run inference on a single image file.
- Webcam mode: capture frames from a webcam and run YOLO inference approximately once per second.
- Read weight directly from an electronic scale or Arduino UNO + HX711 via USB/Serial.
- Verify measured weight against an expected value within a configurable tolerance.

---

## Environment

| Item | Detail |
|---|---|
| Development | WSL Ubuntu 20.04 |
| Runtime | Python 3.8+ inside `venv` |
| Target deployment | Jetson Nano Ubuntu |
| GPU | Optional (CPU fallback supported) |

---

## Project Structure

```
instrument/
├── app/
│   ├── __init__.py
│   ├── config.py          # Central configuration
│   ├── detector.py        # YOLO11 inference (image path or numpy frame)
│   ├── scale_reader.py    # Mock / Serial scale reader
│   ├── weight_checker.py  # Weight verification logic
│   ├── visualizer.py      # OpenCV annotation, overlay, save
│   └── main.py            # Entry point (argparse, image + webcam modes)
├── models/
│   └── best.pt            # ← place your trained model here
├── input/
│   └── test_image.jpg     # ← place your test image here (image mode)
├── output/
│   ├── annotated/         # image mode results
│   └── frames/            # webcam mode saved frames
├── requirements.txt
├── .env.example           # all environment variables with defaults
├── run.sh                 # quick-start: image mode + mock scale
├── run_webcam.sh          # quick-start: webcam mode + mock scale
├── run_web.sh             # web server (generic)
├── run_jetson.sh          # web server with Jetson Nano defaults + diagnostics
├── jetson_setup.sh        # one-time Jetson Nano setup (Python 3.8, venv, deps)
└── README.md
```

> **Important:** `models/best.pt` and `input/test_image.jpg` are **not included**.
> You must place them manually before running.

---

## Installation

```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

## File Placement

| File | Location |
|---|---|
| Trained YOLO11 model | `models/best.pt` |
| Test image | `input/test_image.jpg` |

---

## Execution Modes

### Image mode

Run inference on a single image file, then save an annotated result.

```bash
# Quick start
bash run.sh

# Manual
source venv/bin/activate
python -m app.main \
  --source-type image \
  --model models/best.pt \
  --source input/test_image.jpg \
  --conf 0.25 \
  --expected-weight 1520.35 \
  --tolerance 0.5 \
  --scale-mode mock \
  --mock-weight 1520.35
```

Add `--show` to display the annotated image in a GUI window (requires desktop).

### Webcam mode — mock scale

```bash
# Quick start
bash run_webcam.sh

# Manual
source venv/bin/activate
python -m app.main \
  --source-type webcam \
  --model models/best.pt \
  --conf 0.25 \
  --expected-weight 1520.35 \
  --tolerance 0.5 \
  --scale-mode mock \
  --mock-weight 1520.35 \
  --webcam-index 0 \
  --webcam-width 1280 \
  --webcam-height 720 \
  --webcam-interval 1.0 \
  --show
```

### Webcam mode — serial scale (Arduino / USB scale)

```bash
source venv/bin/activate
python -m app.main \
  --source-type webcam \
  --model models/best.pt \
  --conf 0.25 \
  --expected-weight 1520.35 \
  --tolerance 0.5 \
  --scale-mode serial \
  --serial-port /dev/ttyUSB0 \
  --baudrate 9600 \
  --serial-timeout 0.3 \
  --serial-retries 2 \
  --webcam-index 0 \
  --webcam-width 1280 \
  --webcam-height 720 \
  --webcam-interval 1.0 \
  --show
```

Use `--serial-timeout 0.3` and `--serial-retries 2` to keep serial reads short so the webcam loop is not blocked.

### Save frames during webcam mode

Add `--save-frames` to save each annotated inference frame to `output/frames/`.

### Limit inferences for testing

Add `--max-frames N` to stop after N YOLO inferences (useful for CI or quick tests without pressing `q`).

```bash
python -m app.main --source-type webcam --model models/best.pt \
  --scale-mode mock --mock-weight 1520.35 \
  --webcam-interval 1.0 --max-frames 5 --save-frames
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--source-type` | `image` | `image` or `webcam` |
| `--source` | `input/test_image.jpg` | Input image path (image mode) |
| `--model` | `models/best.pt` | Path to YOLO model |
| `--conf` | `0.25` | Detection confidence threshold |
| `--expected-weight` | `1520.35` | Expected weight in grams |
| `--tolerance` | `0.5` | Acceptable deviation in grams |
| `--scale-mode` | `mock` | `mock` or `serial` |
| `--mock-weight` | `1520.35` | Weight returned in mock mode |
| `--serial-port` | `/dev/ttyUSB0` | Serial device path |
| `--baudrate` | `9600` | Serial baud rate |
| `--serial-timeout` | `2.0` | Serial read timeout (s) |
| `--serial-retries` | `5` | Serial read retry count |
| `--webcam-index` | `0` | Webcam device index |
| `--webcam-width` | `1280` | Capture width |
| `--webcam-height` | `720` | Capture height |
| `--webcam-interval` | `1.0` | Seconds between YOLO inferences |
| `--save-frames` | _(flag)_ | Save annotated frames to output/frames |
| `--max-frames` | _(none)_ | Stop after N inferences |
| `--show` | _(flag)_ | Display result in GUI window |

---

## Scale Reader Modes

### `mock`

Returns a fixed weight value. Use when:
- Physical hardware is not connected.
- Running in WSL without USB passthrough.
- Writing automated tests.

### `serial`

Reads weight from an Arduino UNO + HX711 or a USB electronic scale.
Accepts these output formats (and more):

```
1520.35
1520.35 g
Weight: 1520.35 g
ST,GS,+001520.35g
weight=1520.35
```

Python extracts the numeric value using regex.

---

## Serial Connection Notes

### Common device paths

| Platform | Arduino UNO | USB Scale |
|---|---|---|
| Linux / Jetson Nano | `/dev/ttyACM0` | `/dev/ttyUSB0` |
| WSL (with usbipd) | `/dev/ttyACM0` | `/dev/ttyUSB0` |

### Check connected devices

```bash
ls /dev/ttyUSB*
ls /dev/ttyACM*
dmesg | grep tty
```

### Fix permission denied

```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

### WSL USB passthrough

USB serial devices require [usbipd-win](https://github.com/dorssel/usbipd-win):

```powershell
# In Windows PowerShell (admin)
usbipd list
usbipd attach --wsl --busid <BUSID>
```

---

## Recommended Arduino UNO Output Format

```cpp
// Option A — bare number
Serial.println(weight_grams);

// Option B — labeled
Serial.print("Weight: ");
Serial.print(weight_grams, 2);
Serial.println(" g");
```

---

## WSL Webcam Notes

Webcam access in WSL 2 requires USB passthrough via usbipd-win.

```powershell
# Windows PowerShell (admin)
usbipd attach --wsl --busid <BUSID>
```

Check available video devices in WSL:

```bash
ls /dev/video*
```

Quick webcam test:

```bash
python -c "import cv2; cap=cv2.VideoCapture(0); print('opened:', cap.isOpened()); cap.release()"
```

If webcam does not work in WSL, test on native Ubuntu or Jetson Nano instead.

---

## Jetson Nano — Setup

Run the one-time setup script (requires internet and `sudo`):

```bash
bash jetson_setup.sh
```

This installs Python 3.8, creates a native ARM64 venv, and installs all dependencies.

### Start the web server (standard command)

```bash
./run_jetson.sh
```

This sets the scale defaults (`serial` / `/dev/ttyUSB0` / `9600 baud`) and prints a
diagnostic block before starting Flask.  Override any value inline:

```bash
SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh          # Arduino on ACM0
SCALE_READER_MODE=mock ./run_jetson.sh             # no hardware connected
WEBCAM_INDEX=1 BACKEND_PORT=8080 ./run_jetson.sh   # custom camera / port
DETECTOR_BACKEND=pt ./run_jetson.sh                # force PyTorch backend
```

### Detector backend — ONNX vs PyTorch

The default backend is **ONNX** (`CPUExecutionProvider`), benchmarked at ~3.9× faster
than PyTorch on the Jetson Nano CPU:

| Backend | `detector.predict()` | Notes |
|---------|----------------------|-------|
| `onnx` (default) | ~1.6 s/frame | ONNX Runtime, `CPUExecutionProvider` |
| `pt` | ~6.3 s/frame | PyTorch / Ultralytics |

`best.onnx` must be generated once before use (already present if setup was followed):

```bash
source venv/bin/activate
python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', opset=12)"
```

If `models/best.onnx` is missing at startup, the app falls back to `models/best.pt`
automatically with a log warning.  Set `STRICT_DETECTOR_BACKEND=true` to disable fallback.

To force PyTorch:

```bash
DETECTOR_BACKEND=pt ./run_jetson.sh
```

TensorRT (`.engine`) remains a future optimization path for additional speedup.

### Verified Jetson Nano scale configuration

| Setting | Value | Notes |
|---------|-------|-------|
| `SCALE_READER_MODE` | `serial` | use `mock` if no hardware |
| `SERIAL_PORT` | `/dev/ttyUSB0` | USB scale / CH340 adapter |
| `SERIAL_BAUDRATE` | `9600` | match Arduino sketch baud rate |
| `SERIAL_TIMEOUT` | `2.0` | seconds per readline attempt |
| `SERIAL_READ_RETRIES` | `5` | lines drained per `read_weight()` call |

These are the defaults in `run_jetson.sh` and `.env.example`.

### Serial scale troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| Weight field shows `--` on every upload | Is the device present? | `ls /dev/ttyUSB*` `ls /dev/ttyACM*` |
| `[WARNING] /dev/ttyUSB0 not found` in diagnostics | USB cable connected? | plug in cable, re-run |
| `Permission denied` opening serial port | User not in `dialout` group | `sudo usermod -aG dialout $USER` then **log out and back in** |
| Wrong port (device exists but no data) | Identify the correct path | `dmesg \| grep tty` right after plugging in |
| Weight reads 0 or garbage on first upload | Arduino startup delay | wait 5 s after `./run_jetson.sh` before first upload |
| Weight is always the same wrong value | Wrong baud rate | `SERIAL_BAUDRATE=115200 ./run_jetson.sh` (match your Arduino sketch) |
| No `/dev/ttyUSB*` at all | Driver missing or cable fault | `dmesg \| grep usb` to diagnose |

### Other Jetson Nano notes

- USB webcam is `/dev/video0` → `WEBCAM_INDEX=0` (default).
- CSI camera requires a GStreamer pipeline string instead of a device index.
- Use lower resolution if inference is slow: `WEBCAM_WIDTH=640 WEBCAM_HEIGHT=480`.
- YOLO runs once per second (`WEBCAM_DETECTION_INTERVAL=1.0`) by design.
- Optional max-performance mode:
  ```bash
  sudo nvpmodel -m 0 && sudo jetson_clocks
  ```
- TensorRT export for faster inference (optional):
  ```bash
  yolo export model=models/best.pt format=engine
  ```

---

## Common Issues

| Problem | Solution |
|---|---|
| `best.pt not found` | Place model at `models/best.pt` |
| `Input image not found` | Place image at `input/test_image.jpg` |
| `Cannot open webcam index 0` | Check `ls /dev/video*`; try `--webcam-index 1` |
| No `/dev/video*` in WSL | Use usbipd-win to attach the USB webcam |
| `cv2.imshow` crashes | Omit `--show` in headless environments |
| Serial port permission denied | `sudo usermod -aG dialout $USER`, re-login |
| Serial weight read timeout | Reduce `SERIAL_TIMEOUT=0.5` or use `--serial-timeout 0.5` |
| WSL USB device not found | Attach via usbipd-win |
| No objects detected | Lower `--conf`, verify model classes match image content |
| Weight shows `--` after startup | Normal — Arduino initialises for ~3 s; wait before first upload |
| Weight shows `--` always | Check `ls /dev/ttyUSB*`; verify `SERIAL_PORT`; check dialout group |
| Wrong weight value | Verify `SERIAL_BAUDRATE` matches Arduino sketch |

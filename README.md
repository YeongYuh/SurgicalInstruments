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
│   ├── inference/         # Model-agnostic layer — no ML runtime leaks above this
│   │   ├── types.py       #   Detection / InferenceResult / ModelInfo
│   │   ├── base.py        #   ModelAdapter contract
│   │   ├── registry.py    #   adapter name -> class
│   │   ├── package.py     #   manifest parsing + validation
│   │   ├── profile.py     #   per-package standards / unit weights
│   │   ├── manager.py     #   ModelManager: one active model, atomic switching
│   │   └── adapters/
│   │       └── ultralytics_adapter.py   # the only file that imports YOLO
│   ├── scale_sample.py    # Sample timestamps, freshness, stability
│   ├── scale_reader.py    # Mock / Serial scale reader
│   ├── weight_verification.py  # Standard weight + stability gate
│   ├── detector.py        # Legacy facade over UltralyticsAdapter (CLI only)
│   ├── weight_checker.py  # Weight verification logic (CLI only)
│   ├── visualizer.py      # OpenCV annotation, overlay, save
│   ├── web/               # Flask platform: camera, routes, history, reports
│   └── main.py            # Entry point (argparse, image + webcam modes)
├── model_packages/
│   ├── ortho_tka/         # production package: manifest + factory standards
│   └── demo/              # template — fill in and set "template": false
├── tests/                 # pytest suite (FakeAdapter, no hardware required)
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

### Camera FOURCC — YUYV vs MJPG

The default capture format is **YUYV**, verified stable on the Jetson Nano USB webcam:

| Setting | FPS (measured) | Notes |
|---------|---------------|-------|
| `YUYV` (default) | ~7.5 fps | Raw planar YUV — no libjpeg decode, clean logs |
| `MJPG` | ~7.5 fps | Native MJPEG from camera — may print harmless `Corrupt JPEG data` warnings |
| `AUTO` | driver default | Let V4L2 negotiate; result varies by camera |

Measured FPS is ~7.5 on this camera regardless of fourcc — USB bandwidth and the V4L2
driver on the Jetson 4.9 kernel do not honour the `fps=15` hint at this resolution.
This is acceptable: live preview updates at the camera's natural rate and YOLO inference
runs in a background thread every 5 seconds independently.

To test a specific fourcc before starting the app:

```bash
python3 tools/test_camera_preview.py                # uses CAMERA_FOURCC env var (default YUYV)
CAMERA_FOURCC=MJPG python3 tools/test_camera_preview.py
```

Override at runtime:

```bash
CAMERA_FOURCC=MJPG ./run_jetson.sh   # if your camera works better with MJPG
CAMERA_FOURCC=AUTO ./run_jetson.sh   # let the driver decide
```

### Model packages — swapping the instrument family

The platform (camera, scale, weight verification, inventory comparison, UI,
history, reports, hardware recovery) is fixed.  The **model** is pluggable:

```
model_packages/<id>/manifest.json    which adapter, which model file, defaults
model_packages/<id>/standards.json   factory-default expected quantities
```

The manifest is the source of truth — not environment variables:

```json
{
  "schema_version": 1,
  "id": "ortho_tka",
  "display_name": "骨科 TKA 器械組",
  "department": "orthopedics",
  "adapter": "ultralytics",
  "model_file": "models/best.onnx",
  "fallback_model_file": "models/best.pt",
  "adapter_options": { "task": "segment" },
  "inference": { "confidence": 0.25, "image_size": 640 },
  "inventory": {
    "class_weights": "models/class_weight.json",
    "default_standards": "model_packages/ortho_tka/standards.json"
  }
}
```

Select one at startup, or switch at runtime from the header dropdown.

`./run_jetson.sh` with no overrides starts the `demo` package on its `SurgeryB`
tray. `STARTUP_INVENTORY_PRESET` forces that tray on every boot — without it the
unit resumes whichever surgery the last operator selected. Set it to the empty
string to keep that resume behaviour instead:

```bash
./run_jetson.sh                                            # demo + SurgeryB
ACTIVE_MODEL_PACKAGE=ortho_tka ./run_jetson.sh             # ortho, no preset forced
STARTUP_INVENTORY_PRESET=SurgeryC ./run_jetson.sh          # demo + SurgeryC
STARTUP_INVENTORY_PRESET= ./run_jetson.sh                  # demo, resume last preset

curl -s localhost:5000/api/model-packages                       # list
curl -s -X POST localhost:5000/api/model-package \
     -H 'Content-Type: application/json' -d '{"id":"ortho_tka"}' # switch
```

Switching is an all-or-nothing transaction:

1. recognition pauses, and any in-flight inference finishes first
2. the **old model is released before the new one loads** — the Jetson has 4 GB,
   so the two are never resident at the same time
3. the new model must load, warm up, and be able to recognise every expected
   instrument; failing any of those rolls the old model back into service
4. results from the old package are cleared rather than reinterpreted under the
   new one's standards
5. recognition resumes if it was running — on success *and* on failure

`ok: true` therefore means the model is loaded and usable, not merely accepted.

Startup applies the same bar: load → compatibility → warmup. Until all three
pass, `/status` reports `model_ready: false` (with `model_loading: true` while
it is still working), and `/upload`, `/recognize` and recognition start return
**503** rather than lazily loading an unverified model.

`model_error` and `last_switch_error` are separate: after a failed switch is
rolled back, the restored model is healthy (`model_error: null`) even though the
switch failed (`last_switch_error` set).

Results carry the package and generation that produced them. Any reader that
finds a result from a superseded activation reports `result_stale: true` and
serves no counts, so no API can combine one package's detections with another's
standards.

`has_result` separates "nothing has been recognised yet" (`result_status:
"no_result"`) from "the model looked and found nothing" — the second is a real
inventory with legitimately empty counts. `/bom_report` returns **409** for the
first and a normal report for the second, so a tray that was never scanned can
never produce a file that reads as every instrument missing.

If a model cannot be *released* — teardown raised — the manager refuses to load
anything else (`recovery_required: true`, HTTP 503) rather than risk two models
in 4 GB. That state needs a service restart; it is the one failure the platform
cannot recover from in-process.

Profile edits carry the package they were made against (`X-Model-Package`), so a
debounced standards edit that arrives after a switch is refused with **409**
rather than written onto the wrong department's tray.

`adapter` must always be named explicitly.  **`.onnx` is a serialization format,
not an inference contract** — two ONNX files can need entirely different
post-processing, so the adapter is never guessed from the file extension.  A
model that is not an Ultralytics export needs a new `ModelAdapter` subclass
registered with `register_adapter()`; no platform code changes.

Per-package operator configuration is isolated under
`output/profiles/<id>/standards.json` — orthopaedic standards can never be
applied to an obstetric tray.  The package's own `standards.json` stays a
read-only factory default.

### Inference backend

`ortho_tka` runs **ONNX** (`CPUExecutionProvider`), benchmarked ~3.9× faster than
PyTorch on the Jetson Nano CPU:

| Backend | per frame | Notes |
|---------|-----------|-------|
| `onnx` (default) | ~1.6 s | ONNX Runtime, `CPUExecutionProvider` |
| `pt` | ~6.3 s | PyTorch / Ultralytics |

`best.onnx` must be generated once before use (already present if setup was followed):

```bash
source venv/bin/activate
python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', opset=12)"
```

If `model_file` is missing at startup the adapter falls back to
`fallback_model_file` with a log warning; set `"strict": true` in
`adapter_options` to fail instead.

> The legacy `DETECTOR_BACKEND` / `ONNX_MODEL_PATH` / `ONNX_TASK` env vars now
> apply only to the standalone `app/main.py` CLI.  The web platform reads the
> manifest.

TensorRT (`.engine`) remains a future optimization path — it would be a new
adapter plus a manifest change, not a core rewrite.

### Weight stability gate

A PASS/FAIL verdict is only issued once the scale reading is both **fresh** and
**stable**.  Until then the UI shows 量測中 and `weight_verification.passed` is
`null` — an unsettled scale is an unfinished measurement, not a failed
inventory.  Tunable via `SCALE_STABLE_*` (see `.env.example`).

Stability is measured over the span the *samples* cover, not how much wall time
has passed, so a burst of readings followed by silence never counts as a settled
second.  Once the scale stops reporting (cable pulled, board reset) the cached
value is returned with `fresh: false` and can no longer be verified.

A missing or non-positive class weight for an expected instrument also blocks
any verdict (`reason: "missing_class_weights"`).  Treating it as 0 g would lower
the expected total — exactly the direction that lets an incomplete tray pass.

### Inventory rows are a union

The results table and the CSV export list every class that is *expected or*
detected.  An instrument the model missed entirely has no entry in the counts,
so iterating detections alone would drop the most important row there is: the
one showing 缺少.  See `app/inventory.py`; the kiosk UI mirrors it.

### Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest
```

The suite runs entirely on a FakeAdapter and temporary packages, so it needs no
model binary, camera, or scale.

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

### 7-inch HDMI LCD / kiosk display

The UI is responsive and targets 1024×600 and 800×480 screens.
Launch Chromium in kiosk mode after starting the Flask server:

```bash
chromium-browser --kiosk --app=http://127.0.0.1:5000 --force-device-scale-factor=1
```

If the UI is still too large for the physical display, reduce the scale factor:

```bash
chromium-browser --kiosk --app=http://127.0.0.1:5000 --force-device-scale-factor=0.9
chromium-browser --kiosk --app=http://127.0.0.1:5000 --force-device-scale-factor=0.85
chromium-browser --kiosk --app=http://127.0.0.1:5000 --force-device-scale-factor=0.8
```

The CSS compact mode (`@media (max-width: 1100px), (max-height: 700px)`) activates
automatically on smaller screens and reduces header, padding, font sizes, and button
heights while keeping all sections visible.

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

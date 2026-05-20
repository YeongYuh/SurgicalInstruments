#!/bin/bash
set -e
cd "$(dirname "$0")"

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

#!/bin/bash
set -e
cd "$(dirname "$0")"

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

#!/usr/bin/env bash
set -euo pipefail

cd /home/camlion/projects/instrument
source venv/bin/activate

export ENABLE_SYSTEM_SHUTDOWN=true

echo "[instrument-server] starting at $(date)"
echo "[instrument-server] pwd=$(pwd)"
echo "[instrument-server] python=$(which python3)"
python3 --version

exec ./run_jetson.sh

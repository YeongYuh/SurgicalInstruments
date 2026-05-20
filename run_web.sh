#!/bin/bash
set -e

cd "$(dirname "$0")"
source venv/bin/activate

mkdir -p output/uploads output/annotated output/frames

export FLASK_APP=app.web
export PYTHONPATH="$(pwd)"

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
PORT="${BACKEND_PORT:-5000}"
echo "Starting web server..."
echo "  Local:   http://localhost:${PORT}"
if [ -n "$LOCAL_IP" ]; then
  echo "  Network: http://${LOCAL_IP}:${PORT}"
fi
echo "Press Ctrl+C to stop."

python -m flask run \
  --host="${BACKEND_HOST:-0.0.0.0}" \
  --port="${PORT}" \
  --no-debugger \
  --no-reload

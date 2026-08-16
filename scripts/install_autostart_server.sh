#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_SRC="$PROJECT_DIR/deploy/systemd/instrument-server.service"
SERVICE_DST="/etc/systemd/system/instrument-server.service"

echo "[install-server] copying service file..."
sudo cp "$SERVICE_SRC" "$SERVICE_DST"

echo "[install-server] reloading systemd..."
sudo systemctl daemon-reload

echo "[install-server] enabling instrument-server.service..."
sudo systemctl enable instrument-server.service

echo "[install-server] restarting instrument-server.service..."
sudo systemctl restart instrument-server.service

echo ""
echo "[install-server] done. current status:"
systemctl status instrument-server.service --no-pager

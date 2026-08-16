#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DESKTOP_SRC="$PROJECT_DIR/deploy/autostart/instrument-kiosk.desktop"
AUTOSTART_DIR="/home/camlion/.config/autostart"
CHROMIUM_DATA_DIR="/home/camlion/.config/instrument-chromium"

echo "[install-kiosk] creating autostart directory..."
mkdir -p "$AUTOSTART_DIR"

echo "[install-kiosk] copying .desktop file..."
cp "$DESKTOP_SRC" "$AUTOSTART_DIR/instrument-kiosk.desktop"

echo "[install-kiosk] creating Chromium user data directory..."
mkdir -p "$CHROMIUM_DATA_DIR"

echo "[install-kiosk] setting ownership..."
chown -R camlion:camlion "$AUTOSTART_DIR" "$CHROMIUM_DATA_DIR"
chmod 644 "$AUTOSTART_DIR/instrument-kiosk.desktop"

echo ""
echo "[install-kiosk] done."
echo "  autostart entry : $AUTOSTART_DIR/instrument-kiosk.desktop"
echo "  chromium data   : $CHROMIUM_DATA_DIR"
echo ""
echo "Chromium kiosk will launch automatically on next graphical desktop login."
echo "Ensure auto-login is enabled for user 'camlion' in your display manager settings."

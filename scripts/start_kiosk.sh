#!/usr/bin/env bash
set -euo pipefail

# Disable screen blanking and DPMS power management
xset s off
xset s noblank
xset -dpms

# Hide mouse cursor after 1 second of inactivity (if unclutter is installed)
command -v unclutter >/dev/null && unclutter -idle 1 -root &

# Wait until the Flask server is reachable (up to 120 seconds)
echo "[kiosk] waiting for Flask server on http://127.0.0.1:5000 ..."
timeout=120
elapsed=0
until curl -sf http://127.0.0.1:5000 >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$timeout" ]; then
        echo "[kiosk] ERROR: server did not start within ${timeout}s — aborting"
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "[kiosk] server is up (${elapsed}s). launching Chromium..."

exec chromium-browser \
    --kiosk \
    --app=http://127.0.0.1:5000 \
    --force-device-scale-factor=1 \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-features=TranslateUI \
    --check-for-update-interval=31536000 \
    --user-data-dir=/home/camlion/.config/instrument-chromium

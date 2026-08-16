#!/usr/bin/env bash
# Disable screen sleep, blanking, and screensaver.
# All commands are safe to run on a live desktop session.
# Errors are silently ignored because desktop environments differ.

echo "[disable-sleep] applying X11 xset settings..."
xset s off     || true   # disable screen saver timer
xset s noblank || true   # do not blank screen
xset -dpms     || true   # disable DPMS (display power management)

echo "[disable-sleep] applying GNOME settings (if installed)..."
gsettings set org.gnome.desktop.session idle-delay 0                                         || true
gsettings set org.gnome.desktop.screensaver lock-enabled false                               || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'      || true
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' || true

echo "[disable-sleep] done."
echo ""
echo "To verify X11 settings:"
echo "  xset q"
echo ""
echo "Optional persistent systemd-logind tweak (requires root, edit manually):"
echo "  sudo nano /etc/systemd/logind.conf"
echo "  # Set: HandleLidSwitch=ignore  IdleAction=ignore  IdleActionSec=0"
echo "  sudo systemctl restart systemd-logind"

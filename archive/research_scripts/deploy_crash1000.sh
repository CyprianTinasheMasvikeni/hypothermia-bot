#!/bin/bash
# =============================================================================
# deploy_crash1000.sh — deploy CRASH1000 bot to Oracle and (re)start service
#
# First-time run: creates the crash1000 systemd service
# Subsequent runs: just syncs the file and restarts
#
# Usage: bash deploy_crash1000.sh
# =============================================================================

KEY="$HOME/Desktop/Keys/ssh-key-2026-04-21.key"
SERVER="ubuntu@92.4.135.209"
REMOTE_DIR="/home/ubuntu/hypothermia"
LOCAL_BOT="$(dirname "$0")/deriv_crash1000_bot.py"

echo ""
echo "============================================================"
echo "  CRASH1000 Bot — Deploy to Oracle"
echo "============================================================"

# ── 1. Copy bot file ──────────────────────────────────────────────────────────
echo "[1/3] Copying deriv_crash1000_bot.py..."
scp -i "$KEY" "$LOCAL_BOT" "$SERVER:$REMOTE_DIR/deriv_crash1000_bot.py"
echo "  Done."

# ── 2. Create service if it doesn't exist ─────────────────────────────────────
echo "[2/3] Setting up crash1000 systemd service..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
SERVICE=/etc/systemd/system/crash1000.service
BOTDIR=/home/ubuntu/hypothermia

if [ ! -f "$SERVICE" ]; then
    echo "  Creating service file..."
    sudo tee "$SERVICE" > /dev/null << EOF
[Unit]
Description=CRASH1000 Spike Reversion Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/hypothermia
EnvironmentFile=/home/ubuntu/hypothermia/.env
ExecStart=/home/ubuntu/hypothermia/venv/bin/python deriv_crash1000_bot.py
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/hypothermia/crash1000_bot.log
StandardError=append:/home/ubuntu/hypothermia/crash1000_bot.log

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable crash1000
    echo "  Service created and enabled."
else
    echo "  Service already exists."
fi
REMOTE

# ── 3. Restart ────────────────────────────────────────────────────────────────
echo "[3/3] Restarting crash1000 service..."
ssh -i "$KEY" "$SERVER" "sudo systemctl restart crash1000 && sleep 2 && sudo systemctl status crash1000 --no-pager"

echo ""
echo "  Watch CRASH1000 logs:"
echo "  ssh -i \"$KEY\" $SERVER 'tail -f $REMOTE_DIR/crash1000_bot.log'"
echo ""
echo "  Both bots running:"
echo "  sudo systemctl status hypothermia crash1000"
echo ""

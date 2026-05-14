#!/bin/bash
# =============================================================================
# deploy_all.sh — deploy CRASH1000 + BOOM1000 + XAUUSD bots + dashboard to Oracle Cloud
#
# Services managed:
#   crash1000    — Crash 1000 spike reversion bot  (deriv_crash1000_bot.py)
#   boom1000     — Boom  1000 spike reversion bot  (deriv_boom1000_bot.py)
#   xauusd       — XAUUSD M5 EMA Pullback bot      (xau_bot.py)
#   dashboard    — Streamlit UI                    (dashboard.py)  → port 8501
#
# Usage: bash deploy_all.sh
# =============================================================================

KEY="$HOME/Desktop/Keys/ssh-key-2026-04-21.key"
SERVER="ubuntu@92.4.135.209"
REMOTE_DIR="/home/ubuntu/hypothermia"
LOCAL_DIR="$(dirname "$0")"

SCP="scp -i \"$KEY\""

echo ""
echo "============================================================"
echo "  Hypothermia — Deploy to Oracle Cloud"
echo "  2 spike bots + 1 gold bot + dashboard"
echo "============================================================"

# ── 1. Copy bot files ─────────────────────────────────────────────────────────
echo ""
echo "[1/4] Copying bot files..."
$SCP "$LOCAL_DIR/deriv_crash1000_bot.py" "$SERVER:$REMOTE_DIR/deriv_crash1000_bot.py"
$SCP "$LOCAL_DIR/deriv_boom1000_bot.py"  "$SERVER:$REMOTE_DIR/deriv_boom1000_bot.py"
$SCP "$LOCAL_DIR/xau_bot.py"             "$SERVER:$REMOTE_DIR/xau_bot.py"
$SCP "$LOCAL_DIR/xau_config.py"          "$SERVER:$REMOTE_DIR/xau_config.py"
$SCP "$LOCAL_DIR/xau_strategy.py"        "$SERVER:$REMOTE_DIR/xau_strategy.py"
$SCP "$LOCAL_DIR/portfolio_risk.py"      "$SERVER:$REMOTE_DIR/portfolio_risk.py"
$SCP "$LOCAL_DIR/dashboard.py"           "$SERVER:$REMOTE_DIR/dashboard.py"
echo "  Done."

# ── 2. Set up systemd services if they don't exist ────────────────────────────
echo ""
echo "[2/4] Ensuring systemd services exist..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
BOTDIR=/home/ubuntu/hypothermia

# ── crash1000 (Crash 1000) ────────────────────────────────────────────────────
SERVICE=/etc/systemd/system/crash1000.service
if [ ! -f "$SERVICE" ]; then
    echo "  Creating crash1000 service..."
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
StandardOutput=append:/home/ubuntu/hypothermia/bot_crash1000.log
StandardError=append:/home/ubuntu/hypothermia/bot_crash1000.log

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable crash1000
    echo "  crash1000 service created."
else
    echo "  crash1000 service already exists."
fi

# ── boom1000 (Boom 1000) ─────────────────────────────────────────────────────
SERVICE=/etc/systemd/system/boom1000.service
if [ ! -f "$SERVICE" ]; then
    echo "  Creating boom1000 service..."
    sudo tee "$SERVICE" > /dev/null << EOF
[Unit]
Description=BOOM1000 Spike Reversion Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/hypothermia
EnvironmentFile=/home/ubuntu/hypothermia/.env
ExecStart=/home/ubuntu/hypothermia/venv/bin/python deriv_boom1000_bot.py
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/hypothermia/bot_boom1000.log
StandardError=append:/home/ubuntu/hypothermia/bot_boom1000.log

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable boom1000
    echo "  boom1000 service created."
else
    echo "  boom1000 service already exists."
fi

# ── xauusd (XAUUSD M5 EMA Pullback bot) ──────────────────────────────────────
SERVICE=/etc/systemd/system/xauusd.service
if [ ! -f "$SERVICE" ]; then
    echo "  Creating xauusd service..."
    sudo tee "$SERVICE" > /dev/null << EOF
[Unit]
Description=XAUUSD M5 EMA Pullback Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/hypothermia
EnvironmentFile=/home/ubuntu/hypothermia/.env
ExecStart=/home/ubuntu/hypothermia/venv/bin/python xau_bot.py
Restart=always
RestartSec=30
StandardOutput=append:/home/ubuntu/hypothermia/bot_xauusd.log
StandardError=append:/home/ubuntu/hypothermia/bot_xauusd.log

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable xauusd
    echo "  xauusd service created."
else
    echo "  xauusd service already exists."
fi

# ── dashboard (Streamlit on port 8501) ───────────────────────────────────────
SERVICE=/etc/systemd/system/dashboard.service
if [ ! -f "$SERVICE" ]; then
    echo "  Creating dashboard service..."
    sudo tee "$SERVICE" > /dev/null << EOF
[Unit]
Description=Hypothermia Trading Dashboard (Streamlit)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/hypothermia
EnvironmentFile=/home/ubuntu/hypothermia/.env
ExecStart=/home/ubuntu/hypothermia/venv/bin/streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
Restart=always
RestartSec=15
StandardOutput=append:/home/ubuntu/hypothermia/dashboard.log
StandardError=append:/home/ubuntu/hypothermia/dashboard.log

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable dashboard
    echo "  dashboard service created."

    if sudo ufw status | grep -q "Status: active"; then
        sudo ufw allow 8501/tcp
        echo "  Firewall: port 8501 opened."
    fi
else
    echo "  dashboard service already exists."
fi

REMOTE

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "[3/4] Installing/updating Python packages..."
ssh -i "$KEY" "$SERVER" \
    "cd $REMOTE_DIR && venv/bin/pip install --quiet --upgrade streamlit plotly pandas numpy websockets python-dotenv 2>&1 | tail -3"
echo "  Packages up to date."

# ── 4. Restart all services ───────────────────────────────────────────────────
echo ""
echo "[4/4] Restarting all services..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
for SVC in crash1000 boom1000 xauusd dashboard; do
    sudo systemctl restart $SVC
    sleep 1
    STATUS=$(sudo systemctl is-active $SVC)
    echo "  $SVC: $STATUS"
done
REMOTE

echo ""
echo "============================================================"
echo "  Deploy complete!"
echo ""
echo "  Dashboard: http://92.4.135.209:8501"
echo ""
echo "  Watch logs:"
echo "    Crash 1000 : ssh -i \"$KEY\" $SERVER 'tail -f $REMOTE_DIR/bot_crash1000.log'"
echo "    Boom 1000  : ssh -i \"$KEY\" $SERVER 'tail -f $REMOTE_DIR/bot_boom1000.log'"
echo "    XAUUSD     : ssh -i \"$KEY\" $SERVER 'tail -f $REMOTE_DIR/bot_xauusd.log'"
echo "    Dashboard  : ssh -i \"$KEY\" $SERVER 'tail -f $REMOTE_DIR/dashboard.log'"
echo ""
echo "  All services:"
echo "    ssh -i \"$KEY\" $SERVER 'sudo systemctl status crash1000 boom1000 xauusd dashboard --no-pager'"
echo "============================================================"

#!/bin/bash
# =============================================================================
# Hypothermia Bot — Oracle Cloud Ubuntu 22.04 one-time setup
# Run: bash setup_oracle.sh
# =============================================================================
set -e

BOTDIR="$HOME/hypothermia"
PYTHON="$BOTDIR/venv/bin/python"

echo ""
echo "============================================================"
echo "  Hypothermia Bot — Server Setup"
echo "============================================================"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git curl wget unzip \
    xvfb x11vnc fluxbox \
    software-properties-common

# ── 2. Wine (needed to run MT5 terminal on Linux) ────────────────────────────
echo "[2/7] Installing Wine..."
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key \
    https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ \
    https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
sudo apt-get update -qq
sudo apt-get install -y -qq --install-recommends winehq-stable
echo "  Wine version: $(wine --version)"

# ── 3. Download and install MT5 terminal via Wine ────────────────────────────
echo "[3/7] Installing MetaTrader 5..."
MT5_SETUP="$HOME/mt5setup.exe"
wget -q -O "$MT5_SETUP" \
    "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"

# Run MT5 installer silently via Xvfb (no real display needed)
Xvfb :99 -screen 0 1024x768x16 &
XVFB_PID=$!
sleep 2
DISPLAY=:99 wine "$MT5_SETUP" /auto 2>/dev/null || true
sleep 10
kill "$XVFB_PID" 2>/dev/null || true

MT5_EXE=$(find "$HOME/.wine" -name "terminal64.exe" 2>/dev/null | head -1)
if [ -z "$MT5_EXE" ]; then
    echo "  MT5 install path not found — may need manual install via VNC."
    echo "  Run: x11vnc -storepasswd && x11vnc -rfbport 5900 -rfbauth ~/.vnc/passwd"
    echo "  Then connect via VNC viewer and install MT5 manually."
    MT5_EXE="$HOME/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
fi
echo "  MT5 path: $MT5_EXE"

# ── 4. Bot directory + Python venv ───────────────────────────────────────────
echo "[4/7] Setting up bot directory..."
mkdir -p "$BOTDIR"
cd "$BOTDIR"
python3 -m venv venv
"$BOTDIR/venv/bin/pip" install --quiet --upgrade pip
echo "  Python venv ready."

# ── 5. Install Python dependencies ───────────────────────────────────────────
echo "[5/7] Installing Python packages..."
# MetaTrader5 Python lib on Linux uses Wine internally
"$BOTDIR/venv/bin/pip" install --quiet \
    "MetaTrader5>=5.0.45" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "python-dotenv>=1.0.0" \
    "streamlit>=1.35.0" \
    "plotly>=5.18.0"
echo "  Packages installed."

# ── 6. Write .env template (user must fill in credentials) ───────────────────
echo "[6/7] Creating .env template..."
if [ ! -f "$BOTDIR/.env" ]; then
cat > "$BOTDIR/.env" <<'ENVEOF'
# Deriv MT5 credentials — fill these in before starting the bot
MT5_LOGIN=12345678
MT5_PASSWORD=YourPasswordHere
MT5_SERVER=Deriv-Demo
# MT5_SERVER=Deriv-Live   # switch to live when ready
MT5_PATH=
# If MT5_PATH is empty the bot will try to auto-detect
ENVEOF
    echo "  .env created — EDIT IT with your Deriv MT5 credentials."
else
    echo "  .env already exists — skipping."
fi

# ── 7. systemd service ───────────────────────────────────────────────────────
echo "[7/7] Installing systemd service..."
SERVICE_FILE="/etc/systemd/system/hypothermia.service"
sudo tee "$SERVICE_FILE" > /dev/null <<SVCEOF
[Unit]
Description=Hypothermia Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BOTDIR
EnvironmentFile=$BOTDIR/.env
ExecStartPre=/bin/bash -c 'Xvfb :99 -screen 0 1024x768x16 &'
Environment=DISPLAY=:99
ExecStart=$BOTDIR/venv/bin/python live_bot.py
Restart=always
RestartSec=30
StandardOutput=append:$BOTDIR/bot_stdout.log
StandardError=append:$BOTDIR/bot_stderr.log

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable hypothermia
echo "  Service installed and enabled."

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Edit your credentials:"
echo "     nano $BOTDIR/.env"
echo ""
echo "  2. Upload your code from your PC:"
echo "     scp -r Hypothermia_IPW/* ubuntu@YOUR_IP:$BOTDIR/"
echo ""
echo "  3. Start the bot:"
echo "     sudo systemctl start hypothermia"
echo ""
echo "  4. Watch logs:"
echo "     sudo journalctl -fu hypothermia"
echo "     tail -f $BOTDIR/bot.log"
echo ""
echo "  5. Push code updates anytime from your PC:"
echo "     bash deploy.sh ubuntu@YOUR_IP"
echo ""

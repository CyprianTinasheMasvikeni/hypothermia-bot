#!/bin/bash
# =============================================================================
# deploy.sh — push code to Oracle Cloud server and restart bot
# Usage: bash deploy.sh ubuntu@YOUR_SERVER_IP
# =============================================================================

SERVER="${1:-ubuntu@YOUR_SERVER_IP}"
BOTDIR="$HOME/hypothermia"
REMOTE_DIR="/home/ubuntu/hypothermia"

# Files to deploy (skip secrets + test scripts)
EXCLUDE=(
    "--exclude=.env"
    "--exclude=*.pyc"
    "--exclude=__pycache__"
    "--exclude=live_trades.csv"
    "--exclude=bot.log"
    "--exclude=*.log"
)

echo ""
echo "============================================================"
echo "  Deploying to $SERVER"
echo "============================================================"

# ── 1. Sync code ─────────────────────────────────────────────────────────────
echo "[1/3] Syncing code..."
rsync -az --progress "${EXCLUDE[@]}" \
    "$(dirname "$0")/" \
    "$SERVER:$REMOTE_DIR/"
echo "  Files synced."

# ── 2. Install/update Python packages if requirements changed ─────────────────
echo "[2/3] Updating dependencies..."
ssh "$SERVER" "cd $REMOTE_DIR && venv/bin/pip install --quiet -r requirements.txt"
echo "  Dependencies up to date."

# ── 3. Restart bot service ────────────────────────────────────────────────────
echo "[3/3] Restarting bot..."
ssh "$SERVER" "sudo systemctl restart hypothermia && sleep 2 && sudo systemctl status hypothermia --no-pager"

echo ""
echo "  Bot restarted. Watch logs:"
echo "  ssh $SERVER 'tail -f $REMOTE_DIR/bot.log'"
echo ""

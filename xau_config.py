# xau_config.py
# XAUUSD M5 EMA Pullback Bot -- Deriv WebSocket API
# Tune all settings here. Bot reads these on startup.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Deriv connection -------------------------------------------------------
DERIV_WS = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TOKEN     = os.environ.get("DERIV_TOKEN", "")   # set in .env file

# ---- Symbol & contract ------------------------------------------------------
SYMBOL     = "frxXAUUSD"   # Gold on Deriv WebSocket API
MULTIPLIER = 100            # multiplier contract leverage (check your account)

# ---- Timeframes (seconds) ---------------------------------------------------
M5_GRAN           = 300     # 5-minute candles
H1_GRAN           = 3600    # 1-hour candles
HISTORY_BARS      = 300     # M5 bars loaded on startup
H1_HISTORY_BARS   = 100     # H1 bars loaded on startup

# ---- Indicator periods ------------------------------------------------------
M5_EMA_PERIOD = 50          # EMA on M5 close (proximity filter)
H1_EMA_PERIOD = 21          # EMA on H1 close (trend filter)
ATR_PERIOD    = 14          # ATR smoothing window

# ---- Entry filters ----------------------------------------------------------
ZONE_ATR_MULT  = 0.50       # candle close within 0.50 ATR of M5 EMA50
SESSION_START  = 7          # UTC hour, inclusive (07:00)
SESSION_END    = 20         # UTC hour, inclusive (20:59)
COOLDOWN_BARS  = 2          # min M5 bars between entries

# ---- Exit rules -------------------------------------------------------------
# Chandelier trailing stop: tiered ATR distance based on R progress
# (min_R_to_activate, ATR_trail_distance)
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
CHAND_MULT       = 1.6      # multiplier on top of all tier values (best from backtest)
PARTIAL_R        = 2.0      # close partial when price reaches this many R in our favour
PARTIAL_PCT      = 0.50     # fraction of position closed at partial
MAX_HOLD_CANDLES = 6        # force close after 6 M5 bars (30 min)
SL_ATR_MULT      = 1.0      # SL distance = 1 * ATR

# ---- Risk management --------------------------------------------------------
RISK_PCT_BASE    = 0.02     # 2% of balance per trade
MAX_TRADES_PER_DAY = 12
DAILY_DD_LIMIT   = 0.03     # pause day if balance drops 3% below day start
MONTHLY_DD_LIMIT = 0.20     # pause month if balance drops 20% below month start

# ---- Bot internals ----------------------------------------------------------
POLL_SECS = 5               # main loop poll interval (seconds)

# ---- Log / state files ------------------------------------------------------
LOG_CSV    = BASE_DIR / "live_trades_xauusd.csv"
LOG_TXT    = BASE_DIR / "bot_xauusd.log"
STATE_JSON = BASE_DIR / "state_xauusd.json"

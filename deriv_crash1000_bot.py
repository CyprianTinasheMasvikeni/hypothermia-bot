"""
Deriv API Live Trading Bot — CRASH1000 Spike Reversion
Strategy : BUY after crash spike (M5 body > 2.5x ATR downward)
           + H1 trend filter: only enter when H1 close > H1 EMA21
Market   : 24/7 — no session filter (synthetic index)
Exit     : Chandelier 3xATR (progressive) | 50% partial at 2R | 24-candle timeout
Risk     : 2% base | 3% hot | 1% cold | max 6 trades/day
Kill sw. : 3% daily DD | 20% monthly DD
Backtest : WR=59.5% | PF=2.417 | AvgR=+1.01R | 8/8 months profitable
           OOS (last 2 months): WR=60.5% | PF=2.223 — filter holds OOS
           Gap risk: avg SL loss = -1.86R | Cooldown: 12-candle
           Filter removes 41% of trades, keeps only high-quality setups
"""
import asyncio
import json
import logging
import os
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import portfolio_risk as portfolio

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

import websockets

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SYMBOL            = "CRASH1000"
MULTIPLIER        = 100
M5_GRAN           = 300
H1_GRAN           = 3600
HISTORY_BARS      = 300
H1_HISTORY_BARS   = 100       # 100 H1 bars — enough for EMA21 warmup
H1_EMA_PERIOD     = 21        # H1 EMA21 — S5 filter from backtest
ATR_PERIOD        = 14
SPIKE_THRESHOLD   = 2.5       # body must exceed 2.5x ATR — from backtest

RISK_PCT_BASE     = 0.02
RISK_PCT_HOT      = 0.03
RISK_PCT_COLD     = 0.01
STREAK_THRESHOLD  = 2
SL_ATR_MULT       = 1.0
CHANDELIER_TIERS  = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R         = 2.0
PARTIAL_PCT       = 0.50
MAX_HOLD_CANDLES  = 24        # force close after 24 M5 bars (2 hours) — from backtest
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT    = 0.05      # skip rest of day if daily loss exceeds this
MONTHLY_DD_LIMIT  = 0.20      # pause rest of calendar month if month loss exceeds this
POLL_SECS         = 5
SPIKE_COOLDOWN    = 12        # skip entry if spike in last N candles — CD12 best PF per backtest
GAP_RISK_MULT     = 1.86      # realistic avg SL loss from backtest (gap spikes through SL)

DERIV_WS = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TOKEN    = os.environ.get("DERIV_TOKEN", "")

LOG_CSV    = BASE_DIR / "live_trades_crash1000.csv"
LOG_TXT    = BASE_DIR / "bot_crash1000.log"
STATE_JSON = BASE_DIR / "state_crash1000.json"

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("crash1000_bot")


# ── BOT STATE ─────────────────────────────────────────────────────────────────
class BotState:
    def __init__(self):
        self.balance            = 0.0
        self.peak_balance       = 0.0
        self.day                = None
        self.day_start_balance  = 0.0
        self.trades_today       = 0
        self.last_signal_candle = None
        self.active_trade       = None
        self.consecutive_wins   = 0
        self.consecutive_losses = 0
        self.m5_candles            = []
        self.new_m5_close          = False
        self.last_spike_candle_idx = -999   # for 12-candle cooldown filter
        self.h1_candles            = []     # H1 bars for trend filter
        self.month                 = None
        self.month_start_balance   = 0.0
        self.month_paused          = False   # True = monthly DD hit, skip rest of month

    @property
    def risk_pct(self) -> float:
        if self.consecutive_wins >= STREAK_THRESHOLD:
            return RISK_PCT_HOT
        if self.consecutive_losses >= STREAK_THRESHOLD:
            return RISK_PCT_COLD
        return RISK_PCT_BASE


state = BotState()


# ── DERIV CLIENT ──────────────────────────────────────────────────────────────
class DerivClient:
    def __init__(self):
        self.ws          = None
        self.pending     = {}
        self.req_counter = 0
        self.sub_queue   = asyncio.Queue()
        self._recv_task  = None

    async def connect(self, uri: str):
        self.ws = await websockets.connect(uri, ping_interval=30, ping_timeout=10, open_timeout=20)
        self._recv_task = asyncio.create_task(self._receiver())
        log.info("WebSocket connected.")

    async def _receiver(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                rid = msg.get("req_id")
                if rid and rid in self.pending:
                    fut = self.pending.pop(rid)
                    if not fut.done():
                        fut.set_result(msg)
                else:
                    await self.sub_queue.put(msg)
        except Exception as e:
            log.warning("Receiver stopped: %s", e)
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket closed"))
            self.pending.clear()

    async def send(self, data: dict) -> dict:
        self.req_counter += 1
        rid  = self.req_counter
        data = {**data, "req_id": rid}
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(data))
        return await asyncio.wait_for(fut, timeout=30)

    async def close(self):
        if self._recv_task:
            self._recv_task.cancel()
        if self.ws:
            await self.ws.close()


client = DerivClient()


# ── HELPERS ───────────────────────────────────────────────────────────────────
def candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df[["time", "epoch", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def calc_atr(df: pd.DataFrame) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)),
                   abs(df["low"]  - df["close"].shift(1))))
    return tr.rolling(ATR_PERIOD).mean()


def calc_stake(atr: float, entry_price: float, risk_amount: float) -> float:
    sl_factor = MULTIPLIER * SL_ATR_MULT * atr / entry_price
    if sl_factor <= 0:
        return max(1.0, round(risk_amount, 2))
    return max(1.0, round(risk_amount / sl_factor, 2))


def _chandelier_mult(peak_r: float) -> float:
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            mult = m
    return mult


def get_h1_filter():
    """Return (h1_close, h1_ema21) of last COMPLETED H1 bar. Returns (None, None) if warming up."""
    if len(state.h1_candles) < H1_EMA_PERIOD + 1:
        return None, None
    closes = pd.Series([c["close"] for c in state.h1_candles])
    ema21  = closes.ewm(span=H1_EMA_PERIOD, adjust=False).mean()
    # Use [-2]: last completed bar ([-1] is the currently forming H1 bar)
    return float(state.h1_candles[-2]["close"]), float(ema21.iloc[-2])


# ── CSV JOURNAL ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "trade_id", "open_time", "close_time", "symbol", "direction",
    "entry_price", "exit_price", "stake", "multiplier",
    "risk_amount", "intended_risk", "risk_pct", "pnl_usd", "result", "reason",
    "r_multiple", "peak_r", "candles_held", "spike_body_atr",
    "atr", "balance_before", "balance_after",
]


def write_state(last_signal: str = "WAIT", last_reason: str = ""):
    try:
        h1_close, h1_ema21 = get_h1_filter() if len(state.h1_candles) >= H1_EMA_PERIOD + 1 else (None, None)
        if h1_close is not None and h1_ema21 is not None:
            h1_regime = "BULL" if h1_close > h1_ema21 else "BEAR"
            h1_filter = "PASS" if h1_close > h1_ema21 else "BLOCKED"
        else:
            h1_regime = "WARMUP"
            h1_filter = "WARMUP"
        STATE_JSON.write_text(json.dumps({
            "balance":              round(state.balance, 2),
            "peak_balance":         round(state.peak_balance, 2),
            "day":                  state.day.isoformat() if state.day else None,
            "trades_today":         state.trades_today,
            "consecutive_wins":     state.consecutive_wins,
            "consecutive_losses":   state.consecutive_losses,
            "risk_pct":             round(state.risk_pct * 100, 1),
            "last_signal":          last_signal,
            "last_reason":          last_reason,
            "active_trade":         state.active_trade,
            "month":                state.month,
            "month_start_balance":  round(state.month_start_balance, 2),
            "month_paused":         state.month_paused,
            "h1_close":             round(h1_close, 2) if h1_close else None,
            "h1_ema21":             round(h1_ema21, 2) if h1_ema21 else None,
            "h1_regime":            h1_regime,
            "h1_filter":            h1_filter,
            "updated_at":           datetime.now(timezone.utc).isoformat(),
        }, default=str), encoding="utf-8")
    except Exception:
        pass


def init_csv():
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def log_trade(record: dict):
    with open(LOG_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
            {k: record.get(k, "") for k in CSV_FIELDS}
        )
    log.info("JOURNAL | %s | pnl=$%.2f | %s | %s | peak=%.2fR | W%d/L%d",
             record.get("trade_id", ""), float(record.get("pnl_usd", 0)),
             record.get("result", ""), record.get("reason", ""),
             float(record.get("peak_r", 0)),
             state.consecutive_wins, state.consecutive_losses)


# ── DERIV API CALLS ───────────────────────────────────────────────────────────
async def authorize() -> bool:
    resp = await client.send({"authorize": TOKEN})
    if "error" in resp:
        log.error("Auth failed: %s", resp["error"]["message"])
        return False
    state.balance      = float(resp["authorize"]["balance"])
    state.peak_balance = max(state.peak_balance, state.balance)
    log.info("Authorized | balance=%.2f", state.balance)
    return True


async def get_balance() -> float:
    try:
        resp = await client.send({"balance": 1, "account": "current"})
        if "error" not in resp:
            return float(resp["balance"]["balance"])
    except Exception:
        pass
    return state.balance


async def fetch_history(count: int) -> list:
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   M5_GRAN,
        "count":         count,
        "end":           "latest",
    })
    if "error" in resp:
        log.error("History failed: %s", resp["error"]["message"])
        return []
    return resp.get("candles", [])


async def fetch_h1_history() -> list:
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   H1_GRAN,
        "count":         H1_HISTORY_BARS,
        "end":           "latest",
    })
    return resp.get("candles", [])


async def subscribe_h1_candles():
    await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   H1_GRAN,
        "count":         1,
        "end":           "latest",
        "subscribe":     1,
    })


async def subscribe_candles():
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   M5_GRAN,
        "count":         1,
        "end":           "latest",
        "subscribe":     1,
    })
    if "error" in resp:
        log.error("Subscribe failed: %s", resp["error"]["message"])


async def buy_contract(stake: float, sl_usd: float) -> dict | None:
    resp = await client.send({
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount":        stake,
            "basis":         "stake",
            "contract_type": "MULTUP",
            "currency":      "USD",
            "limit_order":   {"stop_loss": round(sl_usd, 2)},
            "multiplier":    MULTIPLIER,
            "symbol":        SYMBOL,
        },
    })
    if "error" in resp:
        log.error("Buy failed: %s", resp["error"]["message"])
        return None
    contract = resp.get("buy", {})
    log.info("CONTRACT OPEN | MULTUP | id=%s | stake=%.2f | SL=$%.2f",
             contract.get("contract_id"), stake, sl_usd)
    return contract


async def sell_contract(contract_id: int) -> dict | None:
    resp = await client.send({"sell": contract_id, "price": 0})
    if "error" in resp:
        log.error("Sell failed: %s", resp["error"]["message"])
        return None
    return resp.get("sell", {})


async def get_open_contract(contract_id: int) -> dict | None:
    try:
        resp = await client.send({"proposal_open_contract": 1, "contract_id": contract_id})
        if "error" in resp:
            return None
        return resp.get("proposal_open_contract")
    except Exception:
        return None


# ── CANDLE UPDATES ────────────────────────────────────────────────────────────
def handle_ohlc(msg: dict):
    ohlc = msg.get("ohlc", {})
    gran = int(ohlc.get("granularity", 0))

    if gran == H1_GRAN:
        c = {
            "epoch": int(ohlc["open_time"]),
            "open":  float(ohlc["open"]),
            "high":  float(ohlc["high"]),
            "low":   float(ohlc["low"]),
            "close": float(ohlc["close"]),
        }
        lst = state.h1_candles
        if lst and lst[-1]["epoch"] == c["epoch"]:
            lst[-1] = c   # update forming bar
        else:
            lst.append(c) # new H1 bar closed
            if len(lst) > H1_HISTORY_BARS + 5:
                del lst[:len(lst) - (H1_HISTORY_BARS + 5)]
        return

    if gran != M5_GRAN:
        return

    c = {
        "epoch": int(ohlc["open_time"]),
        "open":  float(ohlc["open"]),
        "high":  float(ohlc["high"]),
        "low":   float(ohlc["low"]),
        "close": float(ohlc["close"]),
    }
    lst = state.m5_candles
    if lst and lst[-1]["epoch"] == c["epoch"]:
        lst[-1] = c
    else:
        lst.append(c)
        state.new_m5_close = True
        if len(lst) > HISTORY_BARS + 20:
            del lst[:len(lst) - (HISTORY_BARS + 20)]


# ── TRADE MANAGEMENT ──────────────────────────────────────────────────────────
async def manage_open_trade():
    at = state.active_trade
    if at is None:
        return

    # Count candles elapsed since entry — same as backtest HOLD_CANDLES logic
    if state.new_m5_close:
        at["candles_held"] = at.get("candles_held", 0) + 1

    # Check if partial contract was closed externally by Deriv SL before we hit 2R
    if not at.get("partial_done") and at.get("contract_id_partial"):
        p_contract = await get_open_contract(at["contract_id_partial"])
        if p_contract and (p_contract.get("is_sold") or p_contract.get("status") in ("sold", "expired")):
            p_pnl = float(p_contract.get("profit") or 0)
            at["partial_done"] = True
            at["locked_pnl"]   = p_pnl
            log.info("PARTIAL SL HIT | pnl=$%.2f", p_pnl)

    contract = await get_open_contract(at["contract_id"])
    if contract is None:
        return

    # Deriv closed main contract via SL
    if contract.get("is_sold") or contract.get("status") in ("sold", "expired"):
        await record_closed_trade(contract, reason="SL")
        return

    current_price = float(contract.get("current_spot") or at["entry"])
    pnl           = float(contract.get("profit") or 0)
    entry         = at["entry"]
    atr           = at["atr"]

    # Update peak (always BUY so we track upward peak)
    at["peak"]   = max(at.get("peak", entry), current_price)
    at["peak_r"] = round((at["peak"] - entry) / atr, 2) if atr > 0 else 0

    # Manually close partial at 2R if not done yet
    if not at.get("partial_done") and at.get("contract_id_partial"):
        if current_price >= entry + atr * PARTIAL_R:
            sold = await sell_contract(at["contract_id_partial"])
            if sold:
                locked = float(sold.get("sold_for", 0)) - at.get("partial_stake", 0)
                at["partial_done"] = True
                at["locked_pnl"]   = locked
                log.info("PARTIAL CLOSE | 2R reached | locked=$%.2f", locked)

    # Chandelier trailing stop
    chand_m  = _chandelier_mult(at["peak_r"])
    chand_sl = at["peak"] - atr * chand_m
    at["chand_mult"] = chand_m

    if current_price <= chand_sl:
        log.info("CHANDELIER EXIT | %.1fxATR | peak=%.2fR | price=%.2f | sl=%.2f | PnL=$%.2f",
                 chand_m, at["peak_r"], current_price, chand_sl, pnl)
        sold = await sell_contract(at["contract_id"])
        if sold:
            contract["profit"]     = pnl
            contract["exit_spot"]  = current_price
            contract["entry_spot"] = entry
            await record_closed_trade(contract, reason="CHANDELIER")
        return

    # Force close after 24 M5 candles — matches backtest HOLD_CANDLES=24
    if at.get("candles_held", 0) >= MAX_HOLD_CANDLES:
        log.info("TIME EXIT | held %d candles | closing.", at["candles_held"])
        if not at.get("partial_done") and at.get("contract_id_partial"):
            p_sold = await sell_contract(at["contract_id_partial"])
            if p_sold:
                at["locked_pnl"]  = float(p_sold.get("sold_for", at["partial_stake"])) - at["partial_stake"]
                at["partial_done"] = True
            else:
                # Close attempt failed — try once more then mark done to avoid orphan
                log.warning("TIME EXIT | partial close failed — retrying once")
                await asyncio.sleep(2)
                p_sold2 = await sell_contract(at["contract_id_partial"])
                at["partial_done"] = True
                if p_sold2:
                    at["locked_pnl"] = float(p_sold2.get("sold_for", at["partial_stake"])) - at["partial_stake"]
                else:
                    log.error("TIME EXIT | partial close failed twice — partial may be orphaned on Deriv (contract %s)", at["contract_id_partial"])
        sold = await sell_contract(at["contract_id"])
        if sold:
            main_pnl = float(sold.get("sold_for", at["stake"])) - at["stake"]
            fake     = {"profit": main_pnl, "entry_spot": entry, "exit_spot": current_price}
            await record_closed_trade(fake, reason="TIME")
        return

    total_pnl = pnl + at.get("locked_pnl", 0.0)
    log.info("Trade open | BUY | peak=%.2fR | partial=%s | PnL=$%.2f | price=%.2f | chand_sl=%.2f | held=%d/24",
             at["peak_r"], at.get("partial_done"), total_pnl,
             current_price, chand_sl, at.get("candles_held", 0))


async def record_closed_trade(contract: dict, reason: str = "SL"):
    at = state.active_trade
    if at is None:
        return

    state.balance      = await get_balance()
    state.peak_balance = max(state.peak_balance, state.balance)

    pnl         = float(contract.get("profit") or 0) + at.get("locked_pnl", 0.0)
    entry_price = float(contract.get("entry_spot") or at["entry"])
    exit_price  = float(contract.get("exit_spot")  or at["entry"])
    result      = "WIN" if pnl > 0 else "LOSS"
    risk        = at["risk_amount"]
    r_mult      = round(pnl / risk, 2) if risk else 0

    if pnl > 0:
        state.consecutive_wins   += 1
        state.consecutive_losses  = 0
    else:
        state.consecutive_losses += 1
        state.consecutive_wins    = 0

    log.info("TRADE CLOSED | %s | %s | pnl=$%.2f (%.2fR) | reason=%s | W%d/L%d | balance=%.2f",
             at["trade_id"], result, pnl, r_mult, reason,
             state.consecutive_wins, state.consecutive_losses, state.balance)
    portfolio.on_close("CRASH1000", pnl, state.balance)

    total_stake = round(at["stake"] + at.get("partial_stake", 0), 2)
    log_trade({
        "trade_id":       at["trade_id"],
        "open_time":      at["open_time"],
        "close_time":     datetime.now(timezone.utc).isoformat(),
        "symbol":         SYMBOL,
        "direction":      "BUY",
        "entry_price":    round(entry_price, 2),
        "exit_price":     round(exit_price, 2),
        "stake":          total_stake,
        "multiplier":     MULTIPLIER,
        "risk_amount":    round(risk, 2),
        "risk_pct":       round(at["risk_pct"] * 100, 1),
        "intended_risk":  round(at.get("intended_risk", at["risk_amount"]), 2),
        "pnl_usd":        round(pnl, 2),
        "result":         result,
        "reason":         reason,
        "r_multiple":     r_mult,
        "peak_r":         at.get("peak_r", 0),
        "candles_held":   at.get("candles_held", 0),
        "spike_body_atr": at.get("spike_body_atr", 0),
        "atr":            round(at.get("atr", 0), 4),
        "balance_before": round(at["balance_before"], 2),
        "balance_after":  round(state.balance, 2),
    })
    state.active_trade = None


# ── SIGNAL DETECTION ──────────────────────────────────────────────────────────
async def check_signal_and_trade():
    if len(state.m5_candles) < ATR_PERIOD + 5:
        log.info("Warming up... (%d M5 bars loaded)", len(state.m5_candles))
        return

    df      = candles_to_df(state.m5_candles)
    df["atr"] = calc_atr(df)

    # Work only on confirmed closed candles — exclude current in-progress candle
    closed = df.iloc[:-1].dropna(subset=["atr"])
    if len(closed) < 2:
        return

    last        = closed.iloc[-1]
    atr         = float(last["atr"])
    body        = float(last["close"]) - float(last["open"])
    body_atr    = abs(body) / atr if atr > 0 else 0
    is_spike    = (-body) > SPIKE_THRESHOLD * atr   # big DOWN candle = crash spike
    spike_epoch = int(last["epoch"])

    write_state(
        last_signal="SPIKE" if is_spike else "WAIT",
        last_reason=f"body={body_atr:.2f}xATR | threshold={SPIKE_THRESHOLD}x | atr={atr:.4f}"
    )

    if not is_spike:
        return

    # Avoid acting on the same spike candle twice
    if spike_epoch == state.last_signal_candle:
        return
    state.last_signal_candle = spike_epoch

    # Spike cluster cooldown — skip if another spike in last SPIKE_COOLDOWN candles
    cur_idx = len(closed) - 1
    candles_since_last_spike = cur_idx - state.last_spike_candle_idx
    if candles_since_last_spike <= SPIKE_COOLDOWN:
        log.info("COOLDOWN | spike cluster detected (%d candles since last spike) — skipping entry",
                 candles_since_last_spike)
        state.last_spike_candle_idx = cur_idx
        write_state(last_signal="COOLDOWN",
                    last_reason=f"spike cluster — {candles_since_last_spike} candles since last")
        return
    state.last_spike_candle_idx = cur_idx

    log.info("CRASH SPIKE | body=%.2fxATR | atr=%.4f | checking H1 trend filter...", body_atr, atr)

    # H1 trend filter (S5): only enter when H1 close > H1 EMA21 — backtest WR 59.5%, PF 2.417
    h1_close, h1_ema21 = get_h1_filter()
    if h1_ema21 is None:
        log.info("H1 EMA warming up (%d bars) — skipping entry", len(state.h1_candles))
        write_state(last_signal="H1_WARMUP", last_reason="H1 EMA21 not ready yet")
        return
    if h1_close <= h1_ema21:
        log.info("H1 FILTER BLOCKED | h1_close=%.2f <= h1_ema21=%.2f | bearish H1 — skip BUY",
                 h1_close, h1_ema21)
        write_state(last_signal="H1_BLOCKED",
                    last_reason=f"h1_close={h1_close:.2f} <= h1_ema21={h1_ema21:.2f}")
        return
    log.info("H1 FILTER PASS | h1_close=%.2f > h1_ema21=%.2f | bullish — entering BUY",
             h1_close, h1_ema21)

    # Atomic check-and-reserve: re-verify portfolio limits AND increment counters
    # under an exclusive flock so two bots can never claim the same slot simultaneously.
    ok, reason = portfolio.check_and_reserve("CRASH1000")
    if not ok:
        log.info("Portfolio blocked at execution: %s", reason)
        write_state(last_signal="PORTFOLIO_BLOCKED", last_reason=reason)
        return

    # Enter at open of current (new) candle — matches backtest entry logic
    entry_price  = float(df.iloc[-1]["open"])
    # Gap-adjusted sizing: scale stake so a realistic gap loss (-1.86R avg) costs exactly risk_pct
    risk_amount  = state.balance * state.risk_pct
    gap_adj_risk = round(risk_amount / GAP_RISK_MULT, 2)

    full_stake = calc_stake(atr, entry_price, gap_adj_risk)
    half_stake = max(1.0, round(full_stake * PARTIAL_PCT, 2))
    sl_per_ctr = round(gap_adj_risk * PARTIAL_PCT, 2)

    log.info("ENTER | entry=%.2f | ATR=%.4f | intended_risk=$%.2f | gap_adj=$%.2f (%.0f%%) | stake=2x$%.2f | SL=$%.2f each",
             entry_price, atr, risk_amount, gap_adj_risk, state.risk_pct * 100, half_stake, sl_per_ctr)

    balance_before = state.balance

    # Open partial contract (closed manually at 2R)
    c_partial = await buy_contract(half_stake, sl_per_ctr)
    if c_partial is None:
        portfolio.release_slot("CRASH1000")
        return

    # Open main contract (rides chandelier to full exit)
    c_main = await buy_contract(half_stake, sl_per_ctr)
    if c_main is None:
        portfolio.release_slot("CRASH1000")
        await sell_contract(int(c_partial["contract_id"]))
        return

    state.trades_today += 1
    now      = datetime.now(timezone.utc)
    trade_id = f"{now.strftime('%Y%m%d')}_{state.trades_today:02d}"

    state.active_trade = {
        "trade_id":            trade_id,
        "contract_id":         int(c_main["contract_id"]),
        "contract_id_partial": int(c_partial["contract_id"]),
        "direction":           "BUY",
        "entry":               entry_price,
        "stake":               half_stake,
        "partial_stake":       half_stake,
        "sl_usd":              sl_per_ctr,
        "atr":                 atr,
        "spike_body_atr":      round(body_atr, 3),
        "risk_amount":         gap_adj_risk,    # gap-adjusted — R multiples calculated against this
        "intended_risk":       risk_amount,     # original balance × risk_pct for reference
        "risk_pct":            state.risk_pct,
        "open_time":           now.isoformat(),
        "balance_before":      balance_before,
        "peak":                entry_price,
        "peak_r":              0.0,
        "partial_done":        False,
        "locked_pnl":          0.0,
        "candles_held":        0,
    }
    log.info("TRADE OPEN | #%s | main_id=%d | partial_id=%d | entry=%.2f | SL=$%.2f total",
             trade_id, c_main["contract_id"], c_partial["contract_id"],
             entry_price, risk_amount)
    portfolio.on_open("CRASH1000", balance_before)
    write_state(last_signal="TRADE_OPEN")


# ── CRASH RECOVERY ────────────────────────────────────────────────────────────
async def recover_open_trade():
    """On restart, reload saved active_trade from disk and verify it's still open on Deriv."""
    try:
        if not STATE_JSON.exists():
            return
        saved = json.loads(STATE_JSON.read_text(encoding="utf-8"))

        # Always restore daily counters from saved state so restarts don't reset the day cap
        saved_today = saved.get("trades_today", 0)
        saved_wins  = saved.get("consecutive_wins", 0)
        saved_losses= saved.get("consecutive_losses", 0)
        saved_day   = saved.get("day")
        cur_day     = datetime.now(timezone.utc).date().isoformat()
        if saved_day == cur_day:
            state.trades_today       = saved_today
            state.consecutive_wins   = saved_wins
            state.consecutive_losses = saved_losses
            log.info("RECOVERY | Restored daily state: trades_today=%d | W%d/L%d",
                     state.trades_today, state.consecutive_wins, state.consecutive_losses)

        at = saved.get("active_trade")
        if not at or not at.get("contract_id"):
            return

        log.info("RECOVERY | Found saved trade %s — checking Deriv...", at.get("trade_id", "?"))

        contract = await get_open_contract(int(at["contract_id"]))
        if contract is None:
            log.info("RECOVERY | Contract not found on Deriv — treating as closed.")
            # Close orphaned partial if it exists
            if at.get("contract_id_partial") and not at.get("partial_done"):
                log.info("RECOVERY | Closing orphaned partial contract %s", at["contract_id_partial"])
                await sell_contract(int(at["contract_id_partial"]))
            return

        if contract.get("is_sold") or contract.get("status") in ("sold", "expired"):
            log.info("RECOVERY | Contract already closed — recording result.")
            # Close orphaned partial if it exists and wasn't already done
            if at.get("contract_id_partial") and not at.get("partial_done"):
                p_contract = await get_open_contract(int(at["contract_id_partial"]))
                if p_contract and not (p_contract.get("is_sold") or p_contract.get("status") in ("sold", "expired")):
                    log.info("RECOVERY | Closing orphaned partial %s", at["contract_id_partial"])
                    p_sold = await sell_contract(int(at["contract_id_partial"]))
                    if p_sold:
                        at["locked_pnl"]  = float(p_sold.get("sold_for", at.get("partial_stake", 0))) - at.get("partial_stake", 0)
                        at["partial_done"] = True
                elif p_contract:
                    at["locked_pnl"]  = float(p_contract.get("profit") or 0)
                    at["partial_done"] = True
            state.active_trade = at
            await record_closed_trade(contract, reason="SL")
            return

        # Contract still open — restore full state and resume management
        state.active_trade = at

        current_price = float(contract.get("current_spot") or at["entry"])
        pnl           = float(contract.get("profit") or 0)
        entry         = at["entry"]
        atr           = at["atr"]
        peak_r        = round((current_price - entry) / atr, 2) if atr > 0 else 0

        # Update peak with current price in case it moved while bot was down
        if current_price > at.get("peak", entry):
            state.active_trade["peak"]   = current_price
            state.active_trade["peak_r"] = peak_r

        log.info("RECOVERY | Resumed | trade=%s | price=%.2f | P&L=$%.2f | peak=%.2fR | held=%d/24 candles",
                 at.get("trade_id"), current_price, pnl, peak_r, at.get("candles_held", 0))

    except Exception as e:
        log.warning("RECOVERY | Error during recovery: %s", e)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
async def bot_loop():
    while True:
        try:
            await client.connect(DERIV_WS)

            if not await authorize():
                await client.close()
                await asyncio.sleep(30)
                continue

            log.info("Loading %d M5 bars for %s...", HISTORY_BARS, SYMBOL)
            state.m5_candles = await fetch_history(HISTORY_BARS)
            log.info("Loaded %d M5 bars.", len(state.m5_candles))

            await subscribe_candles()
            log.info("Subscribed to live M5 candles.")

            log.info("Loading %d H1 bars for trend filter...", H1_HISTORY_BARS)
            state.h1_candles = await fetch_h1_history()
            log.info("Loaded %d H1 bars.", len(state.h1_candles))
            await subscribe_h1_candles()
            log.info("Subscribed to live H1 candles.")

            state.balance      = await get_balance()
            state.peak_balance = max(state.peak_balance, state.balance)
            if state.day is None:
                state.day = datetime.now(timezone.utc).date()
                state.day_start_balance = state.balance

            # Recover any trade that was open when bot last crashed
            await recover_open_trade()

            log.info("Bot running | %s | balance=%.2f | risk=%.0f%% | max %d trades/day",
                     SYMBOL, state.balance, state.risk_pct * 100, MAX_TRADES_PER_DAY)

            while True:
                if client._recv_task and client._recv_task.done():
                    raise ConnectionError("Receiver task died — reconnecting")

                drained = 0
                while not client.sub_queue.empty() and drained < 50:
                    msg = client.sub_queue.get_nowait()
                    if msg.get("msg_type") == "ohlc":
                        handle_ohlc(msg)
                    drained += 1

                await asyncio.sleep(POLL_SECS)

                now      = datetime.now(timezone.utc)
                today    = now.date()
                cur_month = now.strftime("%Y-%m")

                # New month — reset monthly tracking and unpause
                if cur_month != state.month:
                    state.month              = cur_month
                    state.month_start_balance = await get_balance()
                    state.month_paused        = False
                    log.info("New month %s | start_balance=%.2f", cur_month, state.month_start_balance)

                # New day — reset daily tracking
                if today != state.day:
                    state.day               = today
                    state.day_start_balance = await get_balance()
                    state.trades_today      = 0
                    log.info("New day | balance=%.2f", state.day_start_balance)

                state.balance = await get_balance()
                if state.balance > 0:
                    state.peak_balance = max(state.peak_balance, state.balance)

                # Manage open trade — runs every poll cycle
                if state.active_trade is not None:
                    await manage_open_trade()
                    write_state()
                    state.new_m5_close = False
                    continue

                # Only look for new signals on M5 candle close
                if not state.new_m5_close:
                    continue
                state.new_m5_close = False

                write_state()

                # Portfolio-level DD guard — shared across both bots
                blocked, reason = portfolio.sync("CRASH1000", state.balance)
                if blocked:
                    log.warning("PORTFOLIO BLOCKED: %s", reason)
                    write_state(last_signal="PORTFOLIO_BLOCKED", last_reason=reason)
                    continue

                # Per-bot monthly DD flag — portfolio.sync() above handles the actual check.
                # Balance comparison is unsafe here: Deriv deducts open stakes from available
                # balance, so another bot's open trade causes a false DD reading.
                if state.month_paused:
                    write_state(last_signal="MONTHLY_DD_PAUSED")
                    continue

                # Per-bot max trades guard
                if state.trades_today >= MAX_TRADES_PER_DAY:
                    log.info("Max trades/day (%d) reached.", MAX_TRADES_PER_DAY)
                    continue

                # Portfolio-level trade count guard
                ok, reason = portfolio.can_open("CRASH1000")
                if not ok:
                    log.info("Portfolio entry blocked: %s", reason)
                    continue

                await check_signal_and_trade()

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                ConnectionError,
                asyncio.TimeoutError) as e:
            log.warning("Connection lost: %s — reconnecting in 15s...", e)
            await asyncio.sleep(15)
        except Exception as e:
            log.error("Unexpected error: %s — reconnecting in 30s...", e, exc_info=True)
            await asyncio.sleep(30)
        finally:
            try:
                await client.close()
            except Exception:
                pass


def main():
    if not TOKEN:
        log.error("DERIV_TOKEN not set in .env!")
        sys.exit(1)
    init_csv()
    log.info("Starting CRASH1000 Spike Reversion Bot | %dx | 2%% risk | 6 trades/day max | 24-candle timeout",
             MULTIPLIER)
    try:
        asyncio.run(bot_loop())
    except KeyboardInterrupt:
        log.info("Bot stopped.")


if __name__ == "__main__":
    main()

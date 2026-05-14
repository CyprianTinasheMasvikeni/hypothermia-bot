"""
xau_bot.py
==========
XAUUSD M5 EMA Pullback -- Deriv WebSocket live trading bot.
Runs on Oracle server (or any Linux/Windows host with Python 3.11+).

Strategy recap:
  Entry  : last closed M5 bar closes within 0.50 ATR of EMA50,
            H1 close > H1 EMA21 => BUY  |  H1 close < H1 EMA21 => SELL
            bullish candle for BUY, bearish candle for SELL
            Session 07:00-20:59 UTC | cooldown 2 bars | max 12 trades/day
  Exit   : Chandelier trailing stop (tiered, x1.6 mult) | 50% partial at 2R | 6-bar time exit

Backtest result (Chandelier x1.6):
  N=1322  WR=53.6%  PF=1.888  OOS=1.930  AvgR=+0.355  120/month
  $50 -> $5,000 in 11 months at 1% compound | $35,707 at 2% compound

Run:
  DERIV_TOKEN=<your_token> python xau_bot.py
  or set DERIV_TOKEN in .env file in the same folder
"""
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

import websockets
import portfolio_risk as portfolio

from xau_config import (
    DERIV_WS, TOKEN,
    SYMBOL, MULTIPLIER,
    M5_GRAN, H1_GRAN, HISTORY_BARS, H1_HISTORY_BARS,
    M5_EMA_PERIOD, H1_EMA_PERIOD, ATR_PERIOD,
    ZONE_ATR_MULT, SESSION_START, SESSION_END, COOLDOWN_BARS,
    CHANDELIER_TIERS, CHAND_MULT, PARTIAL_R, PARTIAL_PCT,
    MAX_HOLD_CANDLES, SL_ATR_MULT,
    RISK_PCT_BASE, MAX_TRADES_PER_DAY,
    DAILY_DD_LIMIT, MONTHLY_DD_LIMIT,
    POLL_SECS, LOG_CSV, STATE_JSON,
)
from xau_strategy import calc_m5, calc_h1, get_signal, chandelier_stop

# ---- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("xau_bot")


# ---- Bot state --------------------------------------------------------------
class BotState:
    def __init__(self):
        self.balance             = 0.0
        self.peak_balance        = 0.0
        self.day                 = None
        self.day_start_balance   = 0.0
        self.trades_today        = 0
        self.active_trade        = None
        self.m5_candles          = []
        self.h1_candles          = []
        self.new_m5_close        = False
        self.last_signal_candle  = None   # epoch of last candle that fired a signal
        self.last_entry_epoch    = None   # epoch of candle that produced last trade entry
        self.month               = None
        self.month_start_balance = 0.0
        self.month_paused        = False

    @property
    def risk_pct(self) -> float:
        return RISK_PCT_BASE


state = BotState()


# ---- Deriv WebSocket client (identical to boom/crash bots) ------------------
class DerivClient:
    def __init__(self):
        self.ws          = None
        self.pending     = {}
        self.req_counter = 0
        self.sub_queue   = asyncio.Queue()
        self._recv_task  = None

    async def connect(self, uri: str):
        self.ws = await websockets.connect(
            uri, ping_interval=30, ping_timeout=10, open_timeout=20
        )
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


# ---- Helper functions -------------------------------------------------------
def candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"].astype(int), unit="s", utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df[["time", "epoch", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def calc_stake(atr: float, entry_price: float, risk_amount: float) -> float:
    """stake = risk_amount / (multiplier * atr / entry_price)"""
    sl_factor = MULTIPLIER * SL_ATR_MULT * atr / entry_price
    if sl_factor <= 0:
        return max(1.0, round(risk_amount, 2))
    return max(1.0, round(risk_amount / sl_factor, 2))


def _chandelier_mult(peak_r: float) -> float:
    """Tiered ATR trail distance (x CHAND_MULT baked in)."""
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            mult = m
    return mult * CHAND_MULT


# ---- CSV trade journal ------------------------------------------------------
CSV_FIELDS = [
    "trade_id", "open_time", "close_time", "symbol", "direction",
    "entry_price", "exit_price", "stake", "multiplier",
    "risk_amount", "risk_pct", "pnl_usd", "result", "reason",
    "r_multiple", "peak_r", "candles_held",
    "atr", "ema50_at_entry", "h1_ema21_at_entry",
    "balance_before", "balance_after",
]


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
    log.info("JOURNAL | %s | pnl=$%.2f | %s | %s | peak=%.2fR",
             record.get("trade_id", ""), float(record.get("pnl_usd", 0)),
             record.get("result", ""), record.get("reason", ""),
             float(record.get("peak_r", 0)))


# ---- State persistence ------------------------------------------------------
def write_state(last_signal: str = "WAIT", last_reason: str = ""):
    try:
        h1_close, h1_ema21 = get_h1_filter()
        STATE_JSON.write_text(json.dumps({
            "balance":             round(state.balance, 2),
            "peak_balance":        round(state.peak_balance, 2),
            "day":                 state.day.isoformat() if state.day else None,
            "trades_today":        state.trades_today,
            "risk_pct":            round(state.risk_pct * 100, 1),
            "last_signal":         last_signal,
            "last_reason":         last_reason,
            "active_trade":        state.active_trade,
            "last_entry_epoch":    state.last_entry_epoch,
            "month":               state.month,
            "month_start_balance": round(state.month_start_balance, 2),
            "month_paused":        state.month_paused,
            "h1_close":            round(h1_close, 5) if h1_close else None,
            "h1_ema21":            round(h1_ema21, 5) if h1_ema21 else None,
            "h1_regime":           ("BULL" if h1_close > h1_ema21 else "BEAR") if (h1_close and h1_ema21) else "WARMUP",
            "updated_at":          datetime.now(timezone.utc).isoformat(),
        }, default=str), encoding="utf-8")
    except Exception:
        pass


# ---- Deriv API calls --------------------------------------------------------
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


async def fetch_m5_history(count: int) -> list:
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   M5_GRAN,
        "count":         count,
        "end":           "latest",
    })
    if "error" in resp:
        log.error("M5 history failed: %s", resp["error"]["message"])
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


async def subscribe_m5():
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   M5_GRAN,
        "count":         1,
        "end":           "latest",
        "subscribe":     1,
    })
    if "error" in resp:
        log.error("M5 subscribe failed: %s", resp["error"]["message"])


async def subscribe_h1():
    await client.send({
        "ticks_history": SYMBOL,
        "style":         "candles",
        "granularity":   H1_GRAN,
        "count":         1,
        "end":           "latest",
        "subscribe":     1,
    })


async def open_contract(direction: str, stake: float, sl_usd: float):
    """Open a MULTUP (BUY) or MULTDOWN (SELL) contract on frxXAUUSD."""
    ctype = "MULTUP" if direction == "BUY" else "MULTDOWN"
    resp  = await client.send({
        "buy": 1,
        "price": stake,
        "parameters": {
            "amount":        stake,
            "basis":         "stake",
            "contract_type": ctype,
            "currency":      "USD",
            "limit_order":   {"stop_loss": round(sl_usd, 2)},
            "multiplier":    MULTIPLIER,
            "symbol":        SYMBOL,
        },
    })
    if "error" in resp:
        log.error("%s open failed: %s", ctype, resp["error"]["message"])
        return None
    contract = resp.get("buy", {})
    log.info("CONTRACT OPEN | %s | id=%s | stake=%.2f | SL=$%.2f",
             ctype, contract.get("contract_id"), stake, sl_usd)
    return contract


async def close_contract(contract_id: int):
    resp = await client.send({"sell": contract_id, "price": 0})
    if "error" in resp:
        log.error("Close failed: %s", resp["error"]["message"])
        return None
    return resp.get("sell", {})


async def get_open_contract(contract_id: int):
    try:
        resp = await client.send({"proposal_open_contract": 1, "contract_id": contract_id})
        if "error" in resp:
            return None
        return resp.get("proposal_open_contract")
    except Exception:
        return None


# ---- Candle stream handling -------------------------------------------------
def handle_ohlc(msg: dict):
    ohlc = msg.get("ohlc", {})
    gran = int(ohlc.get("granularity", 0))

    c = {
        "epoch": int(ohlc["open_time"]),
        "open":  float(ohlc["open"]),
        "high":  float(ohlc["high"]),
        "low":   float(ohlc["low"]),
        "close": float(ohlc["close"]),
    }

    if gran == H1_GRAN:
        lst = state.h1_candles
        if lst and lst[-1]["epoch"] == c["epoch"]:
            lst[-1] = c
        else:
            lst.append(c)
            if len(lst) > H1_HISTORY_BARS + 5:
                del lst[:len(lst) - (H1_HISTORY_BARS + 5)]
        return

    if gran != M5_GRAN:
        return

    lst = state.m5_candles
    if lst and lst[-1]["epoch"] == c["epoch"]:
        lst[-1] = c
    else:
        lst.append(c)
        state.new_m5_close = True
        if len(lst) > HISTORY_BARS + 20:
            del lst[:len(lst) - (HISTORY_BARS + 20)]


def get_h1_filter():
    """Return (h1_close, h1_ema21) of last COMPLETED H1 bar. (None, None) while warming up."""
    if len(state.h1_candles) < H1_EMA_PERIOD + 1:
        return None, None
    closes = pd.Series([c["close"] for c in state.h1_candles])
    ema21  = closes.ewm(span=H1_EMA_PERIOD, adjust=False).mean()
    return float(state.h1_candles[-2]["close"]), float(ema21.iloc[-2])


# ---- Trade management -------------------------------------------------------
async def manage_open_trade():
    at = state.active_trade
    if at is None:
        return

    if state.new_m5_close:
        at["candles_held"] = at.get("candles_held", 0) + 1

    direction = at["direction"]
    entry     = float(at["entry"])
    atr       = float(at["atr"])

    # Check if partial SL was hit by Deriv while bot was polling
    if not at.get("partial_done") and at.get("contract_id_partial"):
        pc = await get_open_contract(at["contract_id_partial"])
        if pc and (pc.get("is_sold") or pc.get("status") in ("sold", "expired")):
            at["partial_done"] = True
            at["locked_pnl"]   = float(pc.get("profit") or 0)
            log.info("PARTIAL SL HIT | locked_pnl=$%.2f", at["locked_pnl"])

    # Poll main contract
    contract = await get_open_contract(at["contract_id"])
    if contract is None:
        return

    # Deriv closed main via SL
    if contract.get("is_sold") or contract.get("status") in ("sold", "expired"):
        await record_closed_trade(contract, reason="SL")
        return

    current  = float(contract.get("current_spot") or entry)
    pnl      = float(contract.get("profit") or 0)

    # Update peak (for BUY: track max, for SELL: track min)
    if direction == "BUY":
        at["peak"]   = max(at.get("peak", entry), current)
        at["peak_r"] = round((at["peak"] - entry) / atr, 2) if atr > 0 else 0.0
    else:
        at["peak"]   = min(at.get("peak", entry), current)
        at["peak_r"] = round((entry - at["peak"]) / atr, 2) if atr > 0 else 0.0

    # Partial close at 2R
    if not at.get("partial_done") and at.get("contract_id_partial"):
        hit_partial = (
            (direction == "BUY"  and current >= entry + atr * PARTIAL_R) or
            (direction == "SELL" and current <= entry - atr * PARTIAL_R)
        )
        if hit_partial:
            sold = await close_contract(at["contract_id_partial"])
            if sold:
                locked = float(sold.get("sold_for", 0)) - at.get("partial_stake", 0)
                at["partial_done"] = True
                at["locked_pnl"]   = locked
                log.info("PARTIAL CLOSE | %s +%.1fR | locked=$%.2f", direction, PARTIAL_R, locked)

    # Chandelier trailing stop
    chand_m  = _chandelier_mult(at["peak_r"])
    extreme  = at["peak"]  # max high for BUY, min low for SELL
    if direction == "BUY":
        chand_sl  = extreme - atr * chand_m
        hit_chand = current <= chand_sl
    else:
        chand_sl  = extreme + atr * chand_m
        hit_chand = current >= chand_sl

    if hit_chand:
        log.info("CHANDELIER EXIT | %s | %.2fxATR | peak=%.2fR | price=%.5f | sl=%.5f | pnl=$%.2f",
                 direction, chand_m, at["peak_r"], current, chand_sl, pnl)
        sold = await close_contract(at["contract_id"])
        if sold:
            contract["profit"]     = pnl
            contract["exit_spot"]  = current
            contract["entry_spot"] = entry
            await record_closed_trade(contract, reason="CHANDELIER")
        return

    # Time exit after MAX_HOLD_CANDLES M5 bars (30 min)
    if at.get("candles_held", 0) >= MAX_HOLD_CANDLES:
        log.info("TIME EXIT | held %d candles | closing %s.", at["candles_held"], direction)
        if not at.get("partial_done") and at.get("contract_id_partial"):
            p_sold = await close_contract(at["contract_id_partial"])
            if p_sold:
                at["locked_pnl"]   = float(p_sold.get("sold_for", at["partial_stake"])) - at["partial_stake"]
                at["partial_done"] = True
            else:
                log.warning("TIME EXIT | partial close failed -- retrying once")
                await asyncio.sleep(2)
                p_sold2 = await close_contract(at["contract_id_partial"])
                at["partial_done"] = True
                if p_sold2:
                    at["locked_pnl"] = float(p_sold2.get("sold_for", at["partial_stake"])) - at["partial_stake"]
                else:
                    log.error("TIME EXIT | partial close failed twice -- may be orphaned (id=%s)", at["contract_id_partial"])
        sold = await close_contract(at["contract_id"])
        if sold:
            main_pnl = float(sold.get("sold_for", at["stake"])) - at["stake"]
            fake = {"profit": main_pnl, "entry_spot": entry, "exit_spot": current}
            await record_closed_trade(fake, reason="TIME")
        return

    total_pnl = pnl + at.get("locked_pnl", 0.0)
    log.info("Trade open | %s | peak=%.2fR | partial=%s | pnl=$%.2f | price=%.5f | chand_sl=%.5f | held=%d/%d",
             direction, at["peak_r"], at.get("partial_done"), total_pnl,
             current, chand_sl, at.get("candles_held", 0), MAX_HOLD_CANDLES)


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
    r_mult      = round(pnl / risk, 2) if risk else 0.0

    log.info("TRADE CLOSED | %s | %s | pnl=$%.2f (%.2fR) | reason=%s | balance=%.2f",
             at["trade_id"], result, pnl, r_mult, reason, state.balance)
    portfolio.on_close(SYMBOL, pnl, state.balance)

    total_stake = round(at["stake"] + at.get("partial_stake", 0), 2)
    log_trade({
        "trade_id":          at["trade_id"],
        "open_time":         at["open_time"],
        "close_time":        datetime.now(timezone.utc).isoformat(),
        "symbol":            SYMBOL,
        "direction":         at["direction"],
        "entry_price":       round(entry_price, 5),
        "exit_price":        round(exit_price, 5),
        "stake":             total_stake,
        "multiplier":        MULTIPLIER,
        "risk_amount":       round(risk, 2),
        "risk_pct":          round(at["risk_pct"] * 100, 1),
        "pnl_usd":           round(pnl, 2),
        "result":            result,
        "reason":            reason,
        "r_multiple":        r_mult,
        "peak_r":            at.get("peak_r", 0),
        "candles_held":      at.get("candles_held", 0),
        "atr":               round(at.get("atr", 0), 5),
        "ema50_at_entry":    round(at.get("ema50_at_entry", 0), 5),
        "h1_ema21_at_entry": round(at.get("h1_ema21_at_entry", 0), 5),
        "balance_before":    round(at["balance_before"], 2),
        "balance_after":     round(state.balance, 2),
    })
    state.active_trade = None


# ---- Signal detection & trade entry ----------------------------------------
async def check_signal_and_trade():
    min_bars = M5_EMA_PERIOD + ATR_PERIOD + 5
    if len(state.m5_candles) < min_bars:
        log.info("Warming up... (%d/%d M5 bars)", len(state.m5_candles), min_bars)
        return

    m5_df = candles_to_df(state.m5_candles)
    m5_df = calc_m5(m5_df)

    # H1 warmup check
    if len(state.h1_candles) < H1_EMA_PERIOD + 1:
        log.info("H1 EMA warming up (%d bars) -- skipping entry", len(state.h1_candles))
        write_state(last_signal="H1_WARMUP")
        return

    h1_df = candles_to_df(state.h1_candles)
    h1_df = calc_h1(h1_df)

    direction, atr = get_signal(m5_df, h1_df)

    # Derive signal epoch from last closed bar for dedup/cooldown
    closed_bars = m5_df.iloc[:-1].dropna(subset=["atr", "ema50"])
    if len(closed_bars) < 2:
        return
    last_closed     = closed_bars.iloc[-1]
    signal_epoch    = int(last_closed["epoch"])
    ema50_val       = float(last_closed["ema50"])
    h1_close, h1_e21 = get_h1_filter()

    # Log every closed bar scan for transparency
    bar_hour = last_closed["time"].hour
    write_state(
        last_signal=direction,
        last_reason=(
            f"close={float(last_closed['close']):.5f} ema50={ema50_val:.5f} "
            f"atr={atr:.5f} "
            f"h1={f'{h1_close:.5f}' if h1_close is not None else 'N/A'} "
            f"h1ema21={f'{h1_e21:.5f}' if h1_e21 is not None else 'N/A'} "
            f"hour={bar_hour}"
        )
    )

    if direction == "WAIT":
        return

    # Same-candle dedup
    if signal_epoch == state.last_signal_candle:
        return
    state.last_signal_candle = signal_epoch

    # Cooldown: need > COOLDOWN_BARS since last entry
    if state.last_entry_epoch is not None:
        bars_since = (signal_epoch - state.last_entry_epoch) / M5_GRAN
        if bars_since <= COOLDOWN_BARS:
            log.info("COOLDOWN: only %.1f bars since last entry (need >%d)", bars_since, COOLDOWN_BARS)
            return

    log.info("SIGNAL %s | close=%.5f | ema50=%.5f | atr=%.5f | H1close=%.5f | H1ema21=%.5f",
             direction, float(last_closed["close"]), ema50_val, atr,
             h1_close or 0, h1_e21 or 0)

    # Enter at open of current (forming) candle -- matches backtest
    entry_price  = float(m5_df.iloc[-1]["open"])
    risk_amount  = state.balance * state.risk_pct
    full_stake   = calc_stake(atr, entry_price, risk_amount)
    half_stake   = max(1.0, round(full_stake * PARTIAL_PCT, 2))
    sl_per_ctr   = max(0.01, round(risk_amount * PARTIAL_PCT, 2))

    log.info("ENTER %s | entry=%.5f | ATR=%.5f | risk=$%.2f (%.0f%%) | stake=2x$%.2f | SL=$%.2f each",
             direction, entry_price, atr, risk_amount, state.risk_pct * 100, half_stake, sl_per_ctr)

    # Atomic check-and-reserve: re-verify portfolio limits AND increment counters
    # under an exclusive flock so two bots can never claim the same slot simultaneously.
    ok, reason = portfolio.check_and_reserve(SYMBOL)
    if not ok:
        log.info("Portfolio blocked at execution: %s", reason)
        write_state(last_signal="PORTFOLIO_BLOCKED", last_reason=reason)
        return

    balance_before = state.balance

    # Open partial contract first (closed manually at 2R or time exit)
    c_partial = await open_contract(direction, half_stake, sl_per_ctr)
    if c_partial is None:
        portfolio.release_slot(SYMBOL)
        return

    # Open main contract (rides chandelier to full exit)
    c_main = await open_contract(direction, half_stake, sl_per_ctr)
    if c_main is None:
        portfolio.release_slot(SYMBOL)
        await close_contract(int(c_partial["contract_id"]))
        return

    state.trades_today   += 1
    state.last_entry_epoch = signal_epoch
    now      = datetime.now(timezone.utc)
    trade_id = f"{now.strftime('%Y%m%d')}_{state.trades_today:02d}"

    state.active_trade = {
        "trade_id":            trade_id,
        "contract_id":         int(c_main["contract_id"]),
        "contract_id_partial": int(c_partial["contract_id"]),
        "direction":           direction,
        "entry":               entry_price,
        "stake":               half_stake,
        "partial_stake":       half_stake,
        "sl_usd":              sl_per_ctr,
        "atr":                 atr,
        "ema50_at_entry":      ema50_val,
        "h1_ema21_at_entry":   h1_e21 or 0.0,
        "risk_amount":         risk_amount,
        "risk_pct":            state.risk_pct,
        "open_time":           now.isoformat(),
        "balance_before":      balance_before,
        "peak":                entry_price,   # BUY: max high | SELL: min low
        "peak_r":              0.0,
        "partial_done":        False,
        "locked_pnl":          0.0,
        "candles_held":        0,
    }
    log.info("TRADE OPEN | #%s | main=%d | partial=%d | entry=%.5f | direction=%s",
             trade_id, c_main["contract_id"], c_partial["contract_id"],
             entry_price, direction)
    portfolio.on_open(SYMBOL, balance_before)
    write_state(last_signal="TRADE_OPEN")


# ---- Crash recovery ---------------------------------------------------------
async def recover_open_trade():
    """On restart, reload saved state from disk and verify trade still open on Deriv."""
    try:
        if not STATE_JSON.exists():
            return
        saved = json.loads(STATE_JSON.read_text(encoding="utf-8"))

        # Restore daily counters if same day
        saved_day = saved.get("day")
        cur_day   = datetime.now(timezone.utc).date().isoformat()
        if saved_day == cur_day:
            state.trades_today       = saved.get("trades_today", 0)
            state.last_entry_epoch   = saved.get("last_entry_epoch")
            log.info("RECOVERY | Restored daily state: trades_today=%d", state.trades_today)

        # Restore month pause flag
        state.month_paused        = saved.get("month_paused", False)
        state.month_start_balance = saved.get("month_start_balance", 0.0)

        at = saved.get("active_trade")
        if not at or not at.get("contract_id"):
            return

        log.info("RECOVERY | Found saved trade %s -- verifying on Deriv...", at.get("trade_id", "?"))

        contract = await get_open_contract(int(at["contract_id"]))
        if contract is None:
            log.info("RECOVERY | Contract not found -- treating as closed.")
            if at.get("contract_id_partial") and not at.get("partial_done"):
                log.info("RECOVERY | Closing orphaned partial %s", at["contract_id_partial"])
                await close_contract(int(at["contract_id_partial"]))
            return

        if contract.get("is_sold") or contract.get("status") in ("sold", "expired"):
            log.info("RECOVERY | Contract already closed -- recording.")
            if at.get("contract_id_partial") and not at.get("partial_done"):
                pc = await get_open_contract(int(at["contract_id_partial"]))
                if pc and not (pc.get("is_sold") or pc.get("status") in ("sold", "expired")):
                    p_sold = await close_contract(int(at["contract_id_partial"]))
                    if p_sold:
                        at["locked_pnl"]  = float(p_sold.get("sold_for", at.get("partial_stake", 0))) - at.get("partial_stake", 0)
                        at["partial_done"] = True
                elif pc:
                    at["locked_pnl"]  = float(pc.get("profit") or 0)
                    at["partial_done"] = True
            state.active_trade = at
            await record_closed_trade(contract, reason="SL")
            return

        # Still open -- resume management
        state.active_trade = at
        current = float(contract.get("current_spot") or at["entry"])
        pnl     = float(contract.get("profit") or 0)
        entry   = float(at["entry"])
        atr     = float(at["atr"])
        direction = at["direction"]

        # Update peak in case price moved while bot was down
        if direction == "BUY":
            if current > at.get("peak", entry):
                at["peak"]   = current
                at["peak_r"] = round((current - entry) / atr, 2)
        else:
            if current < at.get("peak", entry):
                at["peak"]   = current
                at["peak_r"] = round((entry - current) / atr, 2)

        log.info("RECOVERY | Resumed | trade=%s | %s | price=%.5f | pnl=$%.2f | peak=%.2fR | held=%d/%d",
                 at.get("trade_id"), direction, current, pnl,
                 at.get("peak_r", 0), at.get("candles_held", 0), MAX_HOLD_CANDLES)

    except Exception as e:
        log.warning("RECOVERY | Error: %s", e)


# ---- Main async loop --------------------------------------------------------
async def bot_loop():
    while True:
        try:
            await client.connect(DERIV_WS)

            if not await authorize():
                await client.close()
                await asyncio.sleep(30)
                continue

            log.info("Loading %d M5 bars for %s...", HISTORY_BARS, SYMBOL)
            state.m5_candles = await fetch_m5_history(HISTORY_BARS)
            log.info("Loaded %d M5 bars.", len(state.m5_candles))
            await subscribe_m5()
            log.info("Subscribed to live M5 candles.")

            log.info("Loading %d H1 bars for trend filter...", H1_HISTORY_BARS)
            state.h1_candles = await fetch_h1_history()
            log.info("Loaded %d H1 bars.", len(state.h1_candles))
            await subscribe_h1()
            log.info("Subscribed to live H1 candles.")

            state.balance      = await get_balance()
            state.peak_balance = max(state.peak_balance, state.balance)
            if state.day is None:
                state.day               = datetime.now(timezone.utc).date()
                state.day_start_balance = state.balance

            await recover_open_trade()

            log.info("Bot running | %s | balance=%.2f | risk=%.0f%% | max %d/day",
                     SYMBOL, state.balance, state.risk_pct * 100, MAX_TRADES_PER_DAY)

            while True:
                if client._recv_task and client._recv_task.done():
                    raise ConnectionError("Receiver task died -- reconnecting")

                # Drain queued subscription messages (candle updates)
                drained = 0
                while not client.sub_queue.empty() and drained < 50:
                    msg = client.sub_queue.get_nowait()
                    if msg.get("msg_type") == "ohlc":
                        handle_ohlc(msg)
                    drained += 1

                await asyncio.sleep(POLL_SECS)

                now       = datetime.now(timezone.utc)
                today     = now.date()
                cur_month = now.strftime("%Y-%m")

                # New month
                if cur_month != state.month:
                    state.month               = cur_month
                    state.month_start_balance = await get_balance()
                    state.month_paused        = False
                    log.info("New month %s | start_balance=%.2f", cur_month, state.month_start_balance)

                # New day
                if today != state.day:
                    state.day               = today
                    state.day_start_balance = await get_balance()
                    state.trades_today      = 0
                    state.last_entry_epoch  = None
                    log.info("New day | balance=%.2f", state.day_start_balance)

                state.balance = await get_balance()
                if state.balance > 0:
                    state.peak_balance = max(state.peak_balance, state.balance)

                # Manage open trade every poll
                if state.active_trade is not None:
                    await manage_open_trade()
                    write_state()
                    state.new_m5_close = False
                    continue

                # Only scan for new signals on M5 bar close
                if not state.new_m5_close:
                    continue
                state.new_m5_close = False

                write_state()

                # Portfolio-level DD + concurrent-slot gate (mirrors crash/boom pattern).
                # portfolio.sync() uses realized P&L from the shared state file, so it is
                # immune to the available_balance distortion caused by open stakes.
                blocked, reason = portfolio.sync("XAUUSD", state.balance)
                if blocked:
                    log.warning("PORTFOLIO BLOCKED: %s", reason)
                    write_state(last_signal="PORTFOLIO_BLOCKED", last_reason=reason)
                    continue

                # Per-session filter: only trade 07:00-20:59 UTC
                if not (SESSION_START <= now.hour <= SESSION_END):
                    continue

                # Per-bot max trades guard
                if state.trades_today >= MAX_TRADES_PER_DAY:
                    log.info("Max trades/day (%d) reached.", MAX_TRADES_PER_DAY)
                    continue

                # Portfolio pre-gate: re-checks DD flags + daily trade count + concurrent slots
                ok, reason = portfolio.can_open("XAUUSD")
                if not ok:
                    log.info("Portfolio gate: %s", reason)
                    continue

                await check_signal_and_trade()

        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.WebSocketException,
            ConnectionError,
            asyncio.TimeoutError,
        ) as e:
            log.warning("Connection lost: %s -- reconnecting in 15s...", e)
            await asyncio.sleep(15)
        except Exception as e:
            log.error("Unexpected error: %s -- reconnecting in 30s...", e, exc_info=True)
            await asyncio.sleep(30)
        finally:
            try:
                await client.close()
            except Exception:
                pass


# ---- Entry point ------------------------------------------------------------
def main():
    if not TOKEN:
        log.error("DERIV_TOKEN not set. Add it to .env file or set env var.")
        sys.exit(1)
    init_csv()
    log.info(
        "Starting XAUUSD M5 EMA Pullback Bot | %dx | %.0f%% risk | %d/day max | "
        "%d-bar timeout | Chandelier x%.1f | session %02d:00-%02d:59 UTC",
        MULTIPLIER, RISK_PCT_BASE * 100, MAX_TRADES_PER_DAY,
        MAX_HOLD_CANDLES, CHAND_MULT, SESSION_START, SESSION_END
    )
    try:
        asyncio.run(bot_loop())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")


if __name__ == "__main__":
    main()

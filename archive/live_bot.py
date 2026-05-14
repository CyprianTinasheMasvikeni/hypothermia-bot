"""
Live Trading Bot — Step Index
Strategy : step_trend  (M15 trend bias  +  M5 entry confirmation)
Session  : 09:00 – 19:00 GMT
Exit     : Partial close 50% at 2R, then Chandelier 3×ATR on remainder
           Locks in profit on winners that could reverse, lets giants keep running
Risk     : 5 % of balance per trade  |  max 6 trades/day
Kill sw. : 3 % daily DD  |  15 % account DD from peak
"""

import sys
import csv
import time
import os
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load .env if present (used on Linux server)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

import MetaTrader5 as mt5
import pandas as pd

import strategy_step_trend as strategy

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SYMBOL        = "Step Index"
TREND_TF      = mt5.TIMEFRAME_M15
ENTRY_TF      = mt5.TIMEFRAME_M5
TREND_BARS    = 500
ENTRY_BARS    = 300
POLL_SECS     = 15
SESSION_START = 9           # GMT hour (inclusive)
SESSION_END   = 19          # GMT hour (exclusive)
MAGIC         = 20250420

RISK_PCT_BASE      = 0.05   # default risk
RISK_PCT_HOT       = 0.08   # after 2 consecutive wins — trend confirmed
RISK_PCT_COLD      = 0.03   # after 2 consecutive losses — defensive mode
STREAK_THRESHOLD   = 2      # how many in a row to change mode
SL_ATR_MULT        = 1.0    # initial SL = 1 × ATR from entry
# Progressive Chandelier — loosens early, tightens as trade runs into profit
# 0-2R: 3.0xATR  |  2-4R: 2.5xATR  |  4R+: 2.0xATR
CHANDELIER_TIERS   = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R          = 2.0    # close 25% of position when price reaches +2R
PARTIAL_PCT        = 0.25   # fraction of position closed at partial level
MAX_HOLD_CANDLES   = 96     # 96 × M5 = 8 hours — lets giants run all session
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT     = 0.03
ACCOUNT_DD_LIMIT   = 0.15

LOG_CSV = BASE_DIR / "live_trades.csv"
LOG_TXT = BASE_DIR / "bot.log"

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("live_bot")


# ─────────────────────────────────────────────────────────────────────────────
#  MT5 HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def connect() -> bool:
    login    = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server   = os.environ.get("MT5_SERVER")
    path     = os.environ.get("MT5_PATH") or None

    kwargs = {}
    if login and password and server:
        kwargs = dict(login=int(login), password=password, server=server)
        if path:
            kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        log.error("MT5 init failed: %s", mt5.last_error())
        return False
    info = mt5.account_info()
    if info:
        log.info("Connected | account=%s | server=%s | balance=%.2f",
                 info.login, info.server, info.balance)
    return True


def account_balance() -> float:
    info = mt5.account_info()
    return info.balance if info else 0.0


def fetch(symbol: str, tf, bars: int) -> pd.DataFrame | None:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) < 50:
        log.warning("No data for %s", symbol)
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def open_position_by_magic(symbol: str, magic: int):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None
    for p in positions:
        if p.magic == magic:
            return p
    return None


def calc_lot(symbol: str, sl_distance: float, risk_amount: float) -> float:
    info = mt5.symbol_info(symbol)
    if info is None or info.trade_tick_size == 0:
        return info.volume_min if info else 0.01
    unit_value = info.trade_tick_value / info.trade_tick_size
    if unit_value <= 0:
        return info.volume_min
    lot = risk_amount / (sl_distance * unit_value)
    lot = max(info.volume_min,
              min(info.volume_max,
                  round(lot / info.volume_step) * info.volume_step))
    return round(lot, 2)


def place_order(symbol: str, direction: str, sl: float, tp: float, lot: float):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("Cannot get tick for %s", symbol)
        return None
    price      = tick.ask if direction == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    digits     = mt5.symbol_info(symbol).digits

    if not mt5.symbol_info(symbol).visible:
        mt5.symbol_select(symbol, True)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           round(sl, digits),
        "tp":           round(tp, digits),
        "deviation":    30,
        "magic":        MAGIC,
        "comment":      "step_trend",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("Order failed [%s] retcode=%s", direction,
                  result.retcode if result else "None")
        return None
    log.info("ORDER PLACED | %s | lot=%.2f | price=%.5f | SL=%.5f | TP=%.5f",
             direction, lot, price, sl, tp)
    return result


def modify_sltp(ticket: int, symbol: str, new_sl: float, new_tp: float) -> bool:
    """Move SL and TP on an open position (step-up TP update)."""
    digits = mt5.symbol_info(symbol).digits
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "position": ticket,
        "sl":       round(new_sl, digits),
        "tp":       round(new_tp, digits),
    }
    result = mt5.order_send(request)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        log.info("STEP-UP | ticket=%s | new SL=%.5f | new TP=%.5f", ticket, new_sl, new_tp)
    else:
        log.error("MODIFY failed | ticket=%s | retcode=%s", ticket,
                  result.retcode if result else "None")
    return ok


def get_closing_deal(ticket: int, open_time_utc: datetime):
    deals = mt5.history_deals_get(
        open_time_utc - timedelta(seconds=10),
        datetime.now(timezone.utc) + timedelta(seconds=10),
    )
    if not deals:
        return None
    for d in deals:
        if d.position_id == ticket and d.entry == mt5.DEAL_ENTRY_OUT:
            return d
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  PARTIAL CLOSE
# ─────────────────────────────────────────────────────────────────────────────

def partial_close_position(ticket: int, symbol: str, direction: str, lot_to_close: float) -> bool:
    """Close `lot_to_close` of an open position to lock in partial profit."""
    tick       = mt5.symbol_info_tick(symbol)
    info       = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return False
    price      = tick.bid if direction == "BUY" else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if direction == "BUY" else mt5.ORDER_TYPE_BUY
    lot        = max(info.volume_min,
                     round(lot_to_close / info.volume_step) * info.volume_step)
    lot        = round(lot, 2)
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "position":     ticket,
        "price":        price,
        "deviation":    30,
        "magic":        MAGIC,
        "comment":      "partial_close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        log.info("PARTIAL CLOSE | ticket=%s | lot=%.2f | price=%.5f", ticket, lot, price)
    else:
        log.error("PARTIAL CLOSE FAILED | ticket=%s | retcode=%s", ticket,
                  result.retcode if result else "None")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  CHANDELIER EXIT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _chandelier_mult(peak_r: float) -> float:
    """Pick the tightest applicable Chandelier tier based on R reached."""
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            mult = m
    return mult


def check_chandelier(open_pos, active_trade: dict) -> dict:
    """
    Called every poll cycle while a position is open.
    1. At +2R: closes 25% of the position to lock in profit (one-time).
    2. Trails SL with a progressive Chandelier — loosens early, tightens
       as the trade runs deeper into profit (3x -> 2.5x -> 2.0x ATR).
    SL only ever moves in the favourable direction.
    """
    direction = active_trade["direction"]
    ticket    = active_trade["ticket"]
    symbol    = active_trade["symbol"]
    atr       = active_trade["atr"]
    d         = 1 if direction == "BUY" else -1

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return active_trade

    current_price = tick.bid if direction == "BUY" else tick.ask

    # ── Partial close at 2R (one-time, 25%) ─────────────────────────────────
    if not active_trade.get("partial_done"):
        partial_target = active_trade["entry"] + d * atr * PARTIAL_R
        hit_partial = (direction == "BUY" and current_price >= partial_target) or \
                      (direction == "SELL" and current_price <= partial_target)
        if hit_partial:
            lot_to_close = round(active_trade["lot"] * PARTIAL_PCT, 2)
            ok = partial_close_position(ticket, symbol, direction, lot_to_close)
            if ok:
                locked = (partial_target - active_trade["entry"]) * d * \
                         (active_trade["lot"] * PARTIAL_PCT)
                active_trade = {**active_trade,
                                "partial_done": True,
                                "lot": round(active_trade["lot"] * (1 - PARTIAL_PCT), 2),
                                "locked_pnl": round(locked, 4)}
                log.info("PARTIAL LOCKED | +%.1fR on %.0f%% | locked=$%.2f | remaining lot=%.2f",
                         PARTIAL_R, PARTIAL_PCT * 100, locked, active_trade["lot"])

    # ── Update peak ───────────────────────────────────────────────────────────
    old_peak = active_trade.get("peak", active_trade["entry"])
    new_peak = max(old_peak, current_price) if direction == "BUY" else min(old_peak, current_price)

    # ── Progressive Chandelier SL ─────────────────────────────────────────────
    peak_r       = abs(new_peak - active_trade["entry"]) / atr
    chand_mult   = _chandelier_mult(peak_r)
    new_chand_sl = new_peak - d * atr * chand_mult
    old_sl       = active_trade["sl"]
    if direction == "BUY":
        updated_sl = max(old_sl, new_chand_sl)
    else:
        updated_sl = min(old_sl, new_chand_sl)

    if abs(updated_sl - old_sl) > mt5.symbol_info(symbol).point * 2:
        modify_sltp(ticket, symbol, updated_sl, 0)
        active_trade = {**active_trade, "sl": updated_sl, "peak": new_peak,
                        "peak_r": round(peak_r, 2),
                        "chand_mult": chand_mult}
        log.info("CHANDELIER | %.1fxATR tier | SL=%.5f | peak=%.2fR",
                 chand_mult, updated_sl, peak_r)
    elif new_peak != old_peak:
        active_trade = {**active_trade, "peak": new_peak,
                        "peak_r": round(peak_r, 2),
                        "chand_mult": chand_mult}

    return active_trade


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "trade_id", "open_time", "close_time", "symbol", "direction",
    "entry_price", "initial_sl", "lot", "risk_amount", "risk_pct",
    "exit_price", "pnl_usd", "result", "reason",
    "r_multiple", "peak_r", "partial_done", "locked_pnl",
    "atr", "chandelier_mult", "partial_r",
    "balance_before", "balance_after", "account",
]

def init_csv():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        log.info("Journal created: %s", LOG_CSV)

def log_trade(record: dict):
    with open(LOG_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
            {k: record.get(k, "") for k in CSV_FIELDS}
        )
    log.info("JOURNAL | %s | %s | entry=%.5f exit=%.5f | pnl=$%.2f | %s | peak=%.2fR",
             record.get("trade_id", ""), record.get("direction", ""),
             float(record.get("entry_price", 0)), float(record.get("exit_price", 0)),
             float(record.get("pnl_usd", 0)), record.get("result", ""),
             float(record.get("peak_r", 0)))


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_signal(risk_pct: float = RISK_PCT_BASE) -> dict:
    trend_raw = fetch(SYMBOL, TREND_TF, TREND_BARS)
    entry_raw = fetch(SYMBOL, ENTRY_TF, ENTRY_BARS)
    if trend_raw is None or entry_raw is None:
        return {"signal": "WAIT", "reason": "No data"}

    trend_df = strategy.calculate_indicators(trend_raw.iloc[:-1].copy())
    entry_df = strategy.calculate_indicators(entry_raw.iloc[:-1].copy())

    trend_res = strategy.analyze_setup(trend_df)
    entry_res = strategy.analyze_setup(entry_df)

    trend_bias = trend_res.get("checks", {}).get("trend_bias", "NEUTRAL")
    entry_sig  = entry_res.get("signal", "WAIT")

    if trend_bias not in {"STRONG_BUY", "STRONG_SELL"}:
        return {"signal": "WAIT", "reason": f"Trend bias={trend_bias}"}
    if entry_sig == "WAIT":
        return {"signal": "WAIT", "reason": entry_res.get("reason", "No entry")}

    required = "BUY" if trend_bias == "STRONG_BUY" else "SELL"
    if entry_sig != required:
        return {"signal": "WAIT", "reason": f"Entry={entry_sig} vs trend={trend_bias}"}

    atr = float(entry_df.iloc[-1]["atr"])
    if pd.isna(atr) or atr <= 0:
        return {"signal": "WAIT", "reason": "ATR invalid"}

    balance     = account_balance()
    risk_amt    = balance * risk_pct
    sl_dist     = atr * SL_ATR_MULT
    tick        = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return {"signal": "WAIT", "reason": "Tick unavailable"}

    entry_price = tick.ask if entry_sig == "BUY" else tick.bid
    d           = 1 if entry_sig == "BUY" else -1
    sl          = entry_price - d * sl_dist
    # No fixed TP — chandelier will manage the exit
    # Set initial TP very far away so MT5 doesn't interfere
    tp          = entry_price + d * sl_dist * 100
    lot         = calc_lot(SYMBOL, sl_dist, risk_amt)

    return {
        "signal":      entry_sig,
        "reason":      entry_res.get("reason", ""),
        "entry":       entry_price,
        "sl":          sl,
        "tp":          tp,
        "lot":         lot,
        "atr":         atr,
        "risk_amount": risk_amt,
        "balance":     balance,
        "entry_candle_time": str(entry_df.iloc[-1]["time"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION GUARD
# ─────────────────────────────────────────────────────────────────────────────

SKIP_HOURS = {11, 14}   # net-negative hours in backtests, skip entries

def in_session() -> bool:
    now_hour = datetime.now(timezone.utc).hour
    return SESSION_START <= now_hour < SESSION_END and now_hour not in SKIP_HOURS


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

class BotState:
    def __init__(self, balance: float):
        self.peak_balance       = balance
        self.day                = datetime.now(timezone.utc).date()
        self.day_start_balance  = balance
        self.trades_today       = 0
        self.last_signal_candle = None
        self.active_trade       = None
        self.consecutive_wins   = 0
        self.consecutive_losses = 0

    @property
    def risk_pct(self) -> float:
        if self.consecutive_wins >= STREAK_THRESHOLD:
            return RISK_PCT_HOT
        if self.consecutive_losses >= STREAK_THRESHOLD:
            return RISK_PCT_COLD
        return RISK_PCT_BASE


def main():
    if not connect():
        return

    init_csv()
    balance = account_balance()
    state   = BotState(balance)

    tiers_str = "/".join(f"{m}x@{int(r)}R" for r, m in CHANDELIER_TIERS)
    log.info(
        "Bot started | %s | risk=%d/%d/%d%% (cold/base/hot) | partial%.0f%%@%.1fR | Chandelier %s | session=%02d:00-%02d:00 GMT (skip 11,14)",
        SYMBOL, RISK_PCT_COLD*100, RISK_PCT_BASE*100, RISK_PCT_HOT*100,
        PARTIAL_PCT*100, PARTIAL_R, tiers_str, SESSION_START, SESSION_END,
    )

    try:
        while True:
            now   = datetime.now(timezone.utc)
            today = now.date()

            # ── MT5 reconnect guard (handles network drops on server) ────────
            if mt5.account_info() is None:
                log.warning("MT5 connection lost — reconnecting...")
                mt5.shutdown()
                time.sleep(5)
                if not connect():
                    log.error("Reconnect failed. Retrying in 60s...")
                    time.sleep(60)
                    continue
                log.info("MT5 reconnected.")

            # ── Daily reset ──────────────────────────────────────────────────
            if today != state.day:
                state.day              = today
                state.day_start_balance = account_balance()
                state.trades_today     = 0
                log.info("New day | balance=%.2f", state.day_start_balance)

            balance = account_balance()
            if balance > 0:
                state.peak_balance = max(state.peak_balance, balance)

            # ── Account kill switch ──────────────────────────────────────────
            if balance > 0 and balance < state.peak_balance * (1 - ACCOUNT_DD_LIMIT):
                log.error("ACCOUNT DD LIMIT (%.0f%%) HIT. Shutting down.", ACCOUNT_DD_LIMIT * 100)
                break

            # ── Check open position ──────────────────────────────────────────
            open_pos = open_position_by_magic(SYMBOL, MAGIC)

            if open_pos is not None:
                if state.active_trade is not None:
                    state.active_trade = check_chandelier(open_pos, state.active_trade)
                peak_r = state.active_trade.get("peak_r", 0) if state.active_trade else 0
                log.info("Position open | ticket=%s | peak=%.2fR | current_profit=%.2f",
                         open_pos.ticket, peak_r, open_pos.profit)
                time.sleep(POLL_SECS)
                continue

            # ── Position just closed — record to journal ─────────────────────
            if state.active_trade is not None:
                closed_balance = account_balance()
                pnl    = round(closed_balance - state.active_trade["balance_before"], 2)
                risk   = state.active_trade["risk_amount"]
                r_mult = round(pnl / risk, 2) if risk else 0

                exit_price = state.active_trade["entry"]
                reason     = "UNKNOWN"
                open_dt    = datetime.fromisoformat(state.active_trade["open_time"])
                if not open_dt.tzinfo:
                    open_dt = open_dt.replace(tzinfo=timezone.utc)
                deal = get_closing_deal(state.active_trade["ticket"], open_dt)
                if deal:
                    exit_price = deal.price
                    reason = ("TP"    if deal.reason == mt5.DEAL_REASON_TP  else
                              "SL"    if deal.reason == mt5.DEAL_REASON_SL  else
                              "OTHER")

                result = "WIN" if pnl > 0 else "LOSS"

                # Update streak for dynamic risk
                if pnl > 0:
                    state.consecutive_wins   += 1
                    state.consecutive_losses  = 0
                else:
                    state.consecutive_losses += 1
                    state.consecutive_wins    = 0
                next_risk = state.risk_pct
                log.info("TRADE CLOSED | #%s | %s | pnl=$%.2f (%.2fR) | reason=%s | partial=%s | streak=W%d/L%d | next_risk=%.0f%% | balance=%.2f",
                         state.active_trade["trade_id"], result, pnl, r_mult,
                         reason, state.active_trade.get("partial_done", False),
                         state.consecutive_wins, state.consecutive_losses,
                         next_risk * 100, closed_balance)

                log_trade({
                    "trade_id":        state.active_trade["trade_id"],
                    "open_time":       state.active_trade["open_time"],
                    "close_time":      now.isoformat(),
                    "symbol":          SYMBOL,
                    "direction":       state.active_trade["direction"],
                    "entry_price":     state.active_trade["entry"],
                    "initial_sl":      state.active_trade["initial_sl"],
                    "lot":             state.active_trade["lot"],
                    "risk_amount":     round(risk, 2),
                    "risk_pct":        round(state.risk_pct * 100, 1),
                    "exit_price":      round(exit_price, 5),
                    "pnl_usd":         pnl,
                    "result":          result,
                    "reason":          reason,
                    "r_multiple":      r_mult,
                    "peak_r":          state.active_trade.get("peak_r", 0),
                    "partial_done":    state.active_trade.get("partial_done", False),
                    "locked_pnl":      round(state.active_trade.get("locked_pnl", 0.0), 4),
                    "atr":             state.active_trade.get("atr", ""),
                    "chandelier_mult": state.active_trade.get("chand_mult", CHANDELIER_TIERS[0][1]),
                    "partial_r":       PARTIAL_R,
                    "balance_before":  round(state.active_trade["balance_before"], 2),
                    "balance_after":   round(closed_balance, 2),
                    "account":         mt5.account_info().login,
                })
                state.active_trade = None

            # ── Session / risk guards ────────────────────────────────────────
            if not in_session():
                log.info("Outside session (%02d:00-%02d:00 GMT). Waiting...",
                         SESSION_START, SESSION_END)
                time.sleep(60)
                continue

            if balance < state.day_start_balance * (1 - DAILY_DD_LIMIT):
                log.warning("Daily DD limit hit. No more trades today.")
                time.sleep(60)
                continue

            if state.trades_today >= MAX_TRADES_PER_DAY:
                log.info("Max trades/day (%d) reached.", MAX_TRADES_PER_DAY)
                time.sleep(60)
                continue

            # ── Signal check ─────────────────────────────────────────────────
            sig = get_signal(risk_pct=state.risk_pct)
            log.info("Signal: %s | risk=%.0f%% | %s",
                     sig["signal"], state.risk_pct * 100, sig.get("reason", ""))

            if sig["signal"] not in {"BUY", "SELL"}:
                time.sleep(POLL_SECS)
                continue

            candle_ts = sig.get("entry_candle_time")
            if candle_ts == state.last_signal_candle:
                time.sleep(POLL_SECS)
                continue
            state.last_signal_candle = candle_ts

            # ── Place trade ──────────────────────────────────────────────────
            balance_before = account_balance()
            order_result = place_order(
                symbol    = SYMBOL,
                direction = sig["signal"],
                sl        = sig["sl"],
                tp        = sig["tp"],
                lot       = sig["lot"],
            )

            if order_result is not None:
                state.trades_today += 1
                trade_id = f"{now.strftime('%Y%m%d')}_{state.trades_today:02d}"
                time.sleep(0.5)
                new_pos = open_position_by_magic(SYMBOL, MAGIC)
                ticket  = new_pos.ticket if new_pos else 0

                state.active_trade = {
                    "trade_id":     trade_id,
                    "ticket":       ticket,
                    "symbol":       SYMBOL,
                    "open_time":    now.isoformat(),
                    "direction":    sig["signal"],
                    "entry":        sig["entry"],
                    "initial_sl":   sig["sl"],
                    "sl":           sig["sl"],
                    "lot":          sig["lot"],
                    "risk_amount":  sig["risk_amount"],
                    "atr":          sig.get("atr", ""),
                    "balance_before": balance_before,
                    "peak":         sig["entry"],
                    "peak_r":       0.0,
                    "partial_done": False,
                    "locked_pnl":   0.0,
                }
                log.info(
                    "TRADE OPEN | #%s | ticket=%s | %s | lot=%.2f | entry=%.5f | "
                    "SL=%.5f | Chandelier progressive 3/2.5/2xATR | no fixed TP",
                    trade_id, ticket, sig["signal"], sig["lot"],
                    sig["entry"], sig["sl"],
                )

            time.sleep(POLL_SECS)

    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
    finally:
        mt5.shutdown()
        log.info("MT5 connection closed.")


if __name__ == "__main__":
    main()

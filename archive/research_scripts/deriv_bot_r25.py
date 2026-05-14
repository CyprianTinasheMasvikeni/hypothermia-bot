"""
Deriv API Live Trading Bot — Volatility 25 Index (R_25)
Strategy : step_trend  (M15 trend bias + M5 entry confirmation)
Session  : 24hr — no session filter (synthetic index, trends around the clock)
Exit     : Chandelier 3×ATR (progressive) via price monitoring
Risk     : 5% of balance per trade | max 6 trades/day
Kill sw. : 3% daily DD | 15% account DD from peak
"""
import asyncio
import json
import logging
import os
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path

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
import strategy_step_trend as strategy

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SYMBOL            = "R_25"
MULTIPLIER        = 100
M15_GRAN          = 900
M5_GRAN           = 300
TREND_BARS        = 500
ENTRY_BARS        = 300

RISK_PCT_BASE     = 0.05
RISK_PCT_HOT      = 0.08
RISK_PCT_COLD     = 0.03
STREAK_THRESHOLD  = 2
SL_ATR_MULT       = 1.0
CHANDELIER_TIERS  = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R         = 2.0
PARTIAL_PCT       = 0.50
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT    = 0.03
ACCOUNT_DD_LIMIT  = 0.15
POLL_SECS         = 5

DERIV_WS = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
TOKEN    = os.environ.get("DERIV_TOKEN", "")

LOG_CSV    = BASE_DIR / "data" / "live_trades_r25.csv"
LOG_TXT    = BASE_DIR / "bot_r25.log"
STATE_JSON = BASE_DIR / "state_r25.json"

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("deriv_bot_r25")


# ── BOT STATE ─────────────────────────────────────────────────────────────────
class BotState:
    def __init__(self):
        self.balance           = 0.0
        self.peak_balance      = 0.0
        self.day               = None
        self.day_start_balance = 0.0
        self.trades_today      = 0
        self.last_signal_candle = None
        self.active_trade      = None
        self.consecutive_wins  = 0
        self.consecutive_losses = 0
        self.m15_candles       = []
        self.m5_candles        = []
        self.new_m5_close      = False

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
        self.ws            = None
        self.pending       = {}
        self.req_counter   = 0
        self.sub_queue     = asyncio.Queue()
        self._recv_task    = None

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
    return df[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)


def calc_stake(atr: float, entry_price: float, risk_amount: float) -> float:
    sl_factor = MULTIPLIER * SL_ATR_MULT * atr / entry_price
    if sl_factor <= 0:
        return max(1.0, round(risk_amount, 2))
    stake = risk_amount / sl_factor
    return max(1.0, round(stake, 2))


def _chandelier_mult(peak_r: float) -> float:
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            mult = m
    return mult


# ── CSV JOURNAL ───────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "trade_id", "open_time", "close_time", "symbol", "direction",
    "entry_price", "exit_price", "stake", "multiplier",
    "risk_amount", "risk_pct", "pnl_usd", "result", "reason",
    "r_multiple", "peak_r", "atr", "balance_before", "balance_after",
]

def write_state(trend_bias: str = "NEUTRAL", last_signal: str = "WAIT", last_reason: str = ""):
    try:
        at = state.active_trade
        STATE_JSON.write_text(json.dumps({
            "balance":            round(state.balance, 2),
            "peak_balance":       round(state.peak_balance, 2),
            "trades_today":       state.trades_today,
            "consecutive_wins":   state.consecutive_wins,
            "consecutive_losses": state.consecutive_losses,
            "risk_pct":           round(state.risk_pct * 100, 1),
            "trend_bias":         trend_bias,
            "last_signal":        last_signal,
            "last_reason":        last_reason,
            "active_trade":       at,
            "updated_at":         datetime.now(timezone.utc).isoformat(),
        }, default=str), encoding="utf-8")
    except Exception:
        pass


def init_csv():
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

def log_trade(record: dict):
    with open(LOG_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(
            {k: record.get(k, "") for k in CSV_FIELDS}
        )
    log.info("JOURNAL | %s | %s | pnl=$%.2f | %s | peak=%.2fR",
             record.get("trade_id", ""), record.get("direction", ""),
             float(record.get("pnl_usd", 0)), record.get("result", ""),
             float(record.get("peak_r", 0)))


# ── DERIV API CALLS ───────────────────────────────────────────────────────────
async def authorize() -> bool:
    resp = await client.send({"authorize": TOKEN})
    if "error" in resp:
        log.error("Auth failed: %s", resp["error"]["message"])
        return False
    state.balance = float(resp["authorize"]["balance"])
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


async def fetch_history(granularity: int, count: int) -> list:
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style": "candles",
        "granularity": granularity,
        "count": count,
        "end": "latest",
    })
    if "error" in resp:
        log.error("History failed: %s", resp["error"]["message"])
        return []
    return resp.get("candles", [])


async def subscribe_candles(granularity: int):
    resp = await client.send({
        "ticks_history": SYMBOL,
        "style": "candles",
        "granularity": granularity,
        "count": 1,
        "end": "latest",
        "subscribe": 1,
    })
    if "error" in resp:
        log.error("Subscribe failed (gran=%d): %s", granularity, resp["error"]["message"])


async def buy_contract(direction: str, stake: float, sl_usd: float) -> dict | None:
    ctype = "MULTUP" if direction == "BUY" else "MULTDOWN"
    resp = await client.send({
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
        log.error("Buy failed: %s", resp["error"]["message"])
        return None
    contract = resp.get("buy", {})
    log.info("CONTRACT OPEN | %s | id=%s | stake=%.2f | SL=$%.2f",
             direction, contract.get("contract_id"), stake, sl_usd)
    return contract


async def sell_contract(contract_id: int) -> dict | None:
    resp = await client.send({"sell": contract_id, "price": 0})
    if "error" in resp:
        log.error("Sell failed: %s", resp["error"]["message"])
        return None
    return resp.get("sell", {})


async def get_open_contract(contract_id: int) -> dict | None:
    try:
        resp = await client.send({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
        })
        if "error" in resp:
            return None
        return resp.get("proposal_open_contract")
    except Exception:
        return None


# ── CANDLE UPDATE ─────────────────────────────────────────────────────────────
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
    if gran == M15_GRAN:
        lst = state.m15_candles
    elif gran == M5_GRAN:
        lst = state.m5_candles
    else:
        return

    if lst and lst[-1]["epoch"] == c["epoch"]:
        lst[-1] = c
    else:
        lst.append(c)
        if gran == M5_GRAN:
            state.new_m5_close = True
        cap = TREND_BARS + 20 if gran == M15_GRAN else ENTRY_BARS + 20
        if len(lst) > cap:
            del lst[:len(lst) - cap]


# ── TRADE MANAGEMENT ──────────────────────────────────────────────────────────
async def manage_open_trade():
    at = state.active_trade
    if at is None:
        return

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

    if contract.get("is_sold") or contract.get("status") in ("sold", "expired"):
        await record_closed_trade(contract, reason="SL")
        return

    current_price = float(contract.get("current_spot") or at["entry"])
    pnl           = float(contract.get("profit") or 0)
    direction     = at["direction"]
    entry         = at["entry"]
    atr           = at["atr"]
    d             = 1 if direction == "BUY" else -1

    old_peak = at.get("peak", entry)
    new_peak = max(old_peak, current_price) if direction == "BUY" \
               else min(old_peak, current_price)
    at["peak"]   = new_peak
    at["peak_r"] = round(abs(new_peak - entry) / atr, 2) if atr > 0 else 0

    if not at.get("partial_done") and at.get("contract_id_partial"):
        partial_price = entry + d * atr * PARTIAL_R
        hit_partial   = (direction == "BUY"  and current_price >= partial_price) or \
                        (direction == "SELL" and current_price <= partial_price)
        if hit_partial:
            sold = await sell_contract(at["contract_id_partial"])
            if sold:
                locked = float(sold.get("sold_for", 0)) - at.get("partial_stake", 0)
                at["partial_done"] = True
                at["locked_pnl"]   = locked
                log.info("PARTIAL CLOSE | %.1fR reached | locked=$%.2f", PARTIAL_R, locked)

    chand_m  = _chandelier_mult(at["peak_r"])
    chand_sl = new_peak - d * atr * chand_m
    at["chand_mult"] = chand_m

    hit_chandelier = (direction == "BUY"  and current_price <= chand_sl) or \
                     (direction == "SELL" and current_price >= chand_sl)

    if hit_chandelier:
        log.info("CHANDELIER EXIT | %.1fxATR | peak=%.2fR | price=%.2f sl=%.2f | PnL=$%.2f",
                 chand_m, at["peak_r"], current_price, chand_sl, pnl)
        sold = await sell_contract(at["contract_id"])
        if sold:
            contract["profit"]     = pnl
            contract["exit_spot"]  = current_price
            contract["entry_spot"] = entry
            await record_closed_trade(contract, reason="CHANDELIER")
        return

    total_pnl = pnl + at.get("locked_pnl", 0.0)
    log.info("Trade open | %s | peak=%.2fR | partial=%s | PnL=$%.2f | price=%.2f | chand_sl=%.2f",
             direction, at["peak_r"], at.get("partial_done"), total_pnl, current_price, chand_sl)


async def record_closed_trade(contract: dict, reason: str = "SL"):
    at = state.active_trade
    if at is None:
        return

    state.balance = await get_balance()

    pnl         = float(contract.get("profit") or 0) + at.get("locked_pnl", 0.0)
    entry_price = float(contract.get("entry_spot") or at["entry"])
    exit_price  = float(contract.get("exit_spot") or at["entry"])
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

    total_stake = round(at["stake"] + at.get("partial_stake", 0), 2)
    log_trade({
        "trade_id":       at["trade_id"],
        "open_time":      at["open_time"],
        "close_time":     datetime.now(timezone.utc).isoformat(),
        "symbol":         SYMBOL,
        "direction":      at["direction"],
        "entry_price":    round(entry_price, 5),
        "exit_price":     round(exit_price, 5),
        "stake":          total_stake,
        "multiplier":     MULTIPLIER,
        "risk_amount":    round(risk, 2),
        "risk_pct":       round(at["risk_pct"] * 100, 1),
        "pnl_usd":        round(pnl, 2),
        "result":         result,
        "reason":         reason,
        "r_multiple":     r_mult,
        "peak_r":         at.get("peak_r", 0),
        "atr":            round(at.get("atr", 0), 4),
        "balance_before": round(at["balance_before"], 2),
        "balance_after":  round(state.balance, 2),
    })
    state.active_trade = None


# ── SIGNAL + ENTRY ────────────────────────────────────────────────────────────
async def check_signal_and_trade():
    if len(state.m15_candles) < 220 or len(state.m5_candles) < 50:
        log.info("Waiting for candle history (%d M15, %d M5)...",
                 len(state.m15_candles), len(state.m5_candles))
        return

    m15_df = candles_to_df(state.m15_candles)
    m5_df  = candles_to_df(state.m5_candles)

    trend_df = strategy.calculate_indicators(m15_df.iloc[:-1].copy())
    entry_df = strategy.calculate_indicators(m5_df.iloc[:-1].copy())

    trend_res  = strategy.analyze_setup(trend_df)
    entry_res  = strategy.analyze_setup(entry_df)
    trend_bias = trend_res.get("checks", {}).get("trend_bias", "NEUTRAL")
    entry_sig  = entry_res.get("signal", "WAIT")

    log.info("Signal check | trend=%s | entry=%s | %s",
             trend_bias, entry_sig, entry_res.get("reason", ""))
    write_state(trend_bias, entry_sig, entry_res.get("reason", ""))

    if trend_bias not in {"STRONG_BUY", "STRONG_SELL"}:
        return
    required = "BUY" if trend_bias == "STRONG_BUY" else "SELL"
    if entry_sig != required:
        return

    atr = float(entry_df.iloc[-1]["atr"])
    if pd.isna(atr) or atr <= 0:
        return

    candle_ts = str(entry_df.iloc[-1]["time"])
    if candle_ts == state.last_signal_candle:
        return
    state.last_signal_candle = candle_ts

    entry_price     = float(m5_df.iloc[-1]["close"])
    risk_amount     = state.balance * state.risk_pct
    full_stake      = calc_stake(atr, entry_price, risk_amount)
    half_stake      = max(1.0, round(full_stake * PARTIAL_PCT, 2))
    sl_per_contract = round(risk_amount * PARTIAL_PCT, 2)

    log.info("SIGNAL %s | entry=%.2f | ATR=%.4f | stake=%.2f (2x%.2f) | SL=$%.2f | reason=%s",
             entry_sig, entry_price, atr, full_stake, half_stake, risk_amount, entry_res.get("reason", ""))

    balance_before = state.balance

    contract_partial = await buy_contract(entry_sig, half_stake, sl_per_contract)
    if contract_partial is None:
        return

    contract_main = await buy_contract(entry_sig, half_stake, sl_per_contract)
    if contract_main is None:
        await sell_contract(int(contract_partial["contract_id"]))
        return

    state.trades_today += 1
    now      = datetime.now(timezone.utc)
    trade_id = f"{now.strftime('%Y%m%d')}_{state.trades_today:02d}"

    state.active_trade = {
        "trade_id":            trade_id,
        "contract_id":         int(contract_main["contract_id"]),
        "contract_id_partial": int(contract_partial["contract_id"]),
        "direction":           entry_sig,
        "entry":               entry_price,
        "stake":               half_stake,
        "partial_stake":       half_stake,
        "sl_usd":              sl_per_contract,
        "atr":                 atr,
        "risk_amount":         risk_amount,
        "risk_pct":            state.risk_pct,
        "open_time":           now.isoformat(),
        "balance_before":      balance_before,
        "peak":                entry_price,
        "peak_r":              0.0,
        "partial_done":        False,
        "locked_pnl":          0.0,
    }
    log.info("TRADE OPEN | #%s | %s | main_id=%d | partial_id=%d | entry=%.2f | SL=$%.2f",
             trade_id, entry_sig, contract_main["contract_id"],
             contract_partial["contract_id"], entry_price, risk_amount)


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
async def bot_loop():
    while True:
        try:
            await client.connect(DERIV_WS)

            if not await authorize():
                await client.close()
                await asyncio.sleep(30)
                continue

            log.info("Loading candle history...")
            state.m15_candles = await fetch_history(M15_GRAN, TREND_BARS)
            state.m5_candles  = await fetch_history(M5_GRAN,  ENTRY_BARS)
            log.info("Loaded %d M15 + %d M5 candles", len(state.m15_candles), len(state.m5_candles))

            await subscribe_candles(M15_GRAN)
            await subscribe_candles(M5_GRAN)
            log.info("Subscribed to live candles.")

            state.balance = await get_balance()
            state.peak_balance = max(state.peak_balance, state.balance)
            if state.day is None:
                state.day = datetime.now(timezone.utc).date()
                state.day_start_balance = state.balance

            log.info("Bot running | balance=%.2f | risk=%.0f%%",
                     state.balance, state.risk_pct * 100)

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

                now   = datetime.now(timezone.utc)
                today = now.date()

                if today != state.day:
                    state.day = today
                    state.day_start_balance = await get_balance()
                    state.trades_today = 0
                    log.info("New day | balance=%.2f", state.day_start_balance)

                state.balance = await get_balance()
                if state.balance > 0:
                    state.peak_balance = max(state.peak_balance, state.balance)

                if state.active_trade is not None:
                    await manage_open_trade()
                    state.new_m5_close = False
                    continue

                if not state.new_m5_close:
                    continue
                state.new_m5_close = False

                write_state()

                # Account DD — requires 3 consecutive candle closes below threshold
                # (prevents false triggers from crash1000 stake deductions on shared account)
                if state.balance > 0 and state.balance < state.peak_balance * (1 - ACCOUNT_DD_LIMIT):
                    state.dd_strike = getattr(state, "dd_strike", 0) + 1
                    log.warning("ACCOUNT DD strike %d/3 | balance=%.2f peak=%.2f",
                                state.dd_strike, state.balance, state.peak_balance)
                    if state.dd_strike >= 3:
                        log.error("ACCOUNT DD LIMIT (%.0f%%) HIT. Shutting down.", ACCOUNT_DD_LIMIT * 100)
                        return
                else:
                    state.dd_strike = 0

                # No session filter — R_25 trades 24hrs

                if state.balance < state.day_start_balance * (1 - DAILY_DD_LIMIT):
                    log.warning("Daily DD limit hit. No more trades today.")
                    continue

                if state.trades_today >= MAX_TRADES_PER_DAY:
                    log.info("Max trades/day (%d) reached.", MAX_TRADES_PER_DAY)
                    continue

                await check_signal_and_trade()

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                ConnectionError,
                asyncio.TimeoutError) as e:
            log.warning("Connection lost: %s. Reconnecting in 15s...", e)
            await asyncio.sleep(15)
        except Exception as e:
            log.error("Unexpected error: %s. Reconnecting in 30s...", e, exc_info=True)
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
    log.info("Starting Deriv bot | symbol=%s | multiplier=%dx | 24hr no session filter", SYMBOL, MULTIPLIER)
    try:
        asyncio.run(bot_loop())
    except KeyboardInterrupt:
        log.info("Bot stopped by user.")


if __name__ == "__main__":
    main()

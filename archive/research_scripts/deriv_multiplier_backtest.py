"""
Deriv Multiplier Backtester — Step Index (stpRNG) x100
Mirrors deriv_bot.py exactly:
  - Multiplier stake math
  - Progressive chandelier tiers [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
  - Session 09:00-19:00 GMT, skip hours {11, 14}
  - 5% risk/trade (3% cold streak, 8% hot streak)
  - 3% daily DD + 15% account DD kill switches
  - Max 6 trades/day

Usage:
    python deriv_multiplier_backtest.py                        # stpRNG cached data
    python deriv_multiplier_backtest.py --refresh              # re-fetch from Deriv API
    python deriv_multiplier_backtest.py --balance 100          # start with $100
    python deriv_multiplier_backtest.py --compare              # compare all Volatility indices
    python deriv_multiplier_backtest.py --symbol R_75          # single symbol
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import strategy_step_trend as strategy

# ── SETTINGS (must match deriv_bot.py exactly) ────────────────────────────────
MULTIPLIER         = 100
M15_GRAN           = 900
M5_GRAN            = 300
SESSION_START      = 9
SESSION_END        = 19
SKIP_HOURS         = {11, 14}

RISK_PCT_BASE      = 0.05
RISK_PCT_HOT       = 0.08
RISK_PCT_COLD      = 0.03
STREAK_THRESHOLD   = 2
SL_ATR_MULT        = 1.0
CHANDELIER_TIERS   = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R          = 2.0
PARTIAL_PCT        = 0.50
MAX_HOLD_CANDLES   = 96
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT     = 0.03
ACCOUNT_DD_LIMIT   = 0.15

DERIV_WS = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

COMPARE_SYMBOLS = ["stpRNG", "R_25", "R_10", "BOOM1000", "CRASH500", "CRASH900", "RB200", "stpRNG5", "JD50", "1HZ90V"]

TRADE_FIELDS = [
    "trade_id", "open_time", "close_time", "direction",
    "entry_price", "exit_price", "stake", "sl_price", "atr",
    "risk_amount", "risk_pct", "pnl_usd", "result", "reason",
    "r_multiple", "peak_r", "balance_before", "balance_after",
]


# ── DATA FETCHING ─────────────────────────────────────────────────────────────
async def _fetch(symbol: str, gran: int, target: int) -> list:
    all_candles = []
    end = "latest"
    async with websockets.connect(DERIV_WS, ping_interval=20) as ws:
        while len(all_candles) < target:
            await ws.send(json.dumps({
                "ticks_history": symbol,
                "style":         "candles",
                "granularity":   gran,
                "count":         5000,
                "end":           end,
            }))
            raw  = await asyncio.wait_for(ws.recv(), timeout=60)
            resp = json.loads(raw)
            if "error" in resp:
                print(f"  API error: {resp['error']['message']}")
                break
            batch = resp.get("candles", [])
            if not batch:
                break
            all_candles = batch + all_candles
            print(f"    {len(all_candles):,} candles fetched (gran={gran}s)...")
            if len(batch) < 5000:
                break
            end = batch[0]["epoch"] - 1
            await asyncio.sleep(0.5)
    seen = {c["epoch"]: c for c in all_candles}
    return sorted(seen.values(), key=lambda x: x["epoch"])


def load_candles(symbol: str, gran: int, target: int, refresh: bool) -> pd.DataFrame:
    gran_label = "M15" if gran == M15_GRAN else "M5"
    cache = BASE_DIR / "data" / f"cache_{symbol}_{gran_label}.csv"
    if not refresh and cache.exists():
        print(f"  Loading {cache.name}...")
        df = pd.read_csv(cache, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        print(f"    {len(df):,} rows.")
        return df
    print(f"  Fetching {target:,} bars (gran={gran}s) from Deriv for {symbol}...")
    raw = asyncio.run(_fetch(symbol, gran, target))
    df  = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["epoch"].astype(int), unit="s", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df[["time", "open", "high", "low", "close"]].sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"    Saved {len(df):,} rows to {cache.name}")
    return df


# ── HELPERS ───────────────────────────────────────────────────────────────────
def calc_stake(atr: float, entry: float, risk: float) -> float:
    sf = MULTIPLIER * SL_ATR_MULT * atr / entry
    return max(1.0, round(risk / sf, 2)) if sf > 0 else max(1.0, round(risk, 2))


def chandelier_mult(peak_r: float) -> float:
    m = CHANDELIER_TIERS[0][1]
    for min_r, tier_m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            m = tier_m
    return m


NO_SESSION_FILTER = False  # overridden by --no-session flag
RELAXED_MODE      = False  # overridden by --relaxed flag
NO_KILL_SWITCH    = False  # overridden by --no-kill flag (analysis only)
# account DD overrideable via --account-dd

def in_session(hour: int) -> bool:
    if NO_SESSION_FILTER:
        return True
    return SESSION_START <= hour < SESSION_END and hour not in SKIP_HOURS


def apply_relaxed_strategy():
    """Loosen confirmation, pullback, and trend-slope thresholds."""
    strategy.MIN_CONFIRMATION_BODY_RATIO   = 0.35   # was 0.50
    strategy.MIN_CONTINUATION_CLOSE_RATIO  = 0.55   # was 0.70
    strategy.MAX_PULLBACK_DISTANCE_ATR     = 2.0    # was 1.20
    strategy.MIN_EMA_SLOPE                 = 0.3    # was 0.60
    strategy.WEAK_EMA_SLOPE                = 0.1    # was 0.20


def get_risk_pct(wins: int, losses: int) -> float:
    if wins >= STREAK_THRESHOLD:
        return RISK_PCT_HOT
    if losses >= STREAK_THRESHOLD:
        return RISK_PCT_COLD
    return RISK_PCT_BASE


def calc_pnl(direction: str, stake: float, entry: float, exit_p: float) -> float:
    d = 1 if direction == "BUY" else -1
    return stake * MULTIPLIER * d * (exit_p - entry) / entry


def make_record(tid, active, exit_p, close_time, pnl, bal_after, reason):
    result = "WIN" if pnl > 0 else "LOSS"
    if reason == "FORCED_EXIT":
        result = "FORCED_EXIT"
    return {
        "trade_id":       tid,
        "open_time":      active["open_time"],
        "close_time":     str(close_time)[:19],
        "direction":      active["direction"],
        "entry_price":    round(active["entry"], 4),
        "exit_price":     round(float(exit_p), 4),
        "stake":          active["stake"],
        "sl_price":       round(active["sl_price"], 4),
        "atr":            round(active["atr"], 4),
        "risk_amount":    round(active["risk_amount"], 2),
        "risk_pct":       round(active["risk_pct"] * 100, 1),
        "pnl_usd":        round(pnl, 2),
        "result":         result,
        "reason":         reason,
        "r_multiple":     round(pnl / active["risk_amount"], 2) if active["risk_amount"] else 0,
        "peak_r":         round(active.get("peak_r", 0), 2),
        "balance_before": round(active["balance_before"], 2),
        "balance_after":  round(bal_after, 2),
    }


# ── SIMULATION ────────────────────────────────────────────────────────────────
def simulate(m15_df: pd.DataFrame, m5_df: pd.DataFrame, start_balance: float):
    print("Pre-computing indicators...")
    m15_ind = strategy.calculate_indicators(m15_df.copy())
    m5_ind  = strategy.calculate_indicators(m5_df.copy())

    # Int64 timestamps for fast O(log n) M15 lookup
    m15_ns = m15_ind["time"].astype("int64").values
    m5_ns  = m5_ind["time"].astype("int64").values

    trades        = []
    balance       = start_balance
    peak_balance  = start_balance
    active        = None
    wins = losses = trade_count = trades_today = 0
    day           = None
    day_start_bal = start_balance
    last_sig_ts   = None
    WARMUP        = 250

    total_iter = len(m5_ind) - WARMUP - 1
    print(f"Simulating {total_iter:,} M5 candles...")

    for i in range(WARMUP, len(m5_ind) - 1):
        if (i - WARMUP) % 2000 == 0:
            pct = (i - WARMUP) / total_iter * 100
            print(f"  {pct:.0f}%  balance=${balance:,.2f}  trades={trade_count}")

        row   = m5_ind.iloc[i]
        hour  = row["time"].hour
        today = row["time"].date()

        if today != day:
            day           = today
            day_start_bal = balance
            trades_today  = 0

        if not NO_KILL_SWITCH and balance < peak_balance * (1 - ACCOUNT_DD_LIMIT):
            print(f"  ACCOUNT DD LIMIT hit on {today}. Stopping.")
            break

        # ── Manage open trade ──────────────────────────────────────────────
        if active is not None:
            hi  = float(row["high"])
            lo  = float(row["low"])
            d   = 1 if active["direction"] == "BUY" else -1
            sl  = active["sl_price"]
            atr = active["atr"]
            active["age"] = active.get("age", 0) + 1

            # Time exit — force close after MAX_HOLD_CANDLES
            if active["age"] >= MAX_HOLD_CANDLES:
                exit_p = float(row["close"])
                pnl = calc_pnl(active["direction"], active["stake"], active["entry"], exit_p)
                pnl += active.get("locked_pnl", 0.0)
                balance += pnl
                peak_balance = max(peak_balance, balance)
                if pnl > 0: wins += 1; losses = 0
                else:       losses += 1; wins = 0
                trade_count += 1
                trades.append(make_record(trade_count, active, exit_p, row["time"], pnl, balance, "TIME_EXIT"))
                active = None
                continue

            sl_hit = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            if sl_hit:
                pnl = calc_pnl(active["direction"], active["stake"], active["entry"], sl)
                pnl += active.get("locked_pnl", 0.0)
                balance += pnl
                peak_balance = max(peak_balance, balance)
                if pnl > 0: wins += 1; losses = 0
                else:       losses += 1; wins = 0
                trade_count += 1
                trades.append(make_record(trade_count, active, sl, row["time"], pnl, balance, "SL"))
                active = None
                continue

            # Partial close at 2R — lock in 50% of position
            if not active.get("partial_done"):
                partial_price = active["entry"] + d * atr * PARTIAL_R
                hit_partial = (d == 1 and hi >= partial_price) or (d == -1 and lo <= partial_price)
                if hit_partial:
                    locked = calc_pnl(active["direction"], active["stake"] * PARTIAL_PCT, active["entry"], partial_price)
                    active["locked_pnl"]   = active.get("locked_pnl", 0.0) + locked
                    active["stake"]       *= (1.0 - PARTIAL_PCT)
                    active["partial_done"] = True

            # Update peak
            if d == 1:
                active["peak"] = max(active["peak"], hi)
            else:
                active["peak"] = min(active["peak"], lo)
            active["peak_r"] = abs(active["peak"] - active["entry"]) / atr if atr > 0 else 0

            cm       = chandelier_mult(active["peak_r"])
            chand_sl = active["peak"] - d * atr * cm

            chand_hit = (d == 1 and lo <= chand_sl) or (d == -1 and hi >= chand_sl)
            if chand_hit:
                pnl = calc_pnl(active["direction"], active["stake"], active["entry"], chand_sl)
                pnl += active.get("locked_pnl", 0.0)
                balance += pnl
                peak_balance = max(peak_balance, balance)
                if pnl > 0: wins += 1; losses = 0
                else:       losses += 1; wins = 0
                trade_count += 1
                trades.append(make_record(trade_count, active, chand_sl, row["time"], pnl, balance, "CHANDELIER"))
                active = None
            continue

        # ── Signal check ───────────────────────────────────────────────────
        if not in_session(hour):
            continue
        if trades_today >= MAX_TRADES_PER_DAY:
            continue
        if not NO_KILL_SWITCH and balance < day_start_bal * (1 - DAILY_DD_LIMIT):
            continue

        # M15 slice up to (but not including) the current open M15 candle
        m15_idx = int(np.searchsorted(m15_ns, m5_ns[i], side="right"))
        if m15_idx < 222:
            continue

        # analyze_setup reads only iloc[-1] and iloc[-2] — slicing is O(1)
        trend_res = strategy.analyze_setup(m15_ind.iloc[:m15_idx - 1])
        entry_res = strategy.analyze_setup(m5_ind.iloc[:i + 1])  # candle i just closed

        trend_bias = trend_res.get("checks", {}).get("trend_bias", "NEUTRAL")
        entry_sig  = entry_res.get("signal", "WAIT")

        if RELAXED_MODE:
            valid_biases = {"STRONG_BUY", "STRONG_SELL", "WEAK_BUY", "WEAK_SELL"}
        else:
            valid_biases = {"STRONG_BUY", "STRONG_SELL"}
        if trend_bias not in valid_biases:
            continue
        required = "BUY" if "BUY" in trend_bias else "SELL"
        if entry_sig != required:
            continue

        atr = float(m5_ind.iloc[i]["atr"])
        if pd.isna(atr) or atr <= 0:
            continue

        sig_ts = str(m5_ind.iloc[i]["time"])
        if sig_ts == last_sig_ts:
            continue
        last_sig_ts = sig_ts

        entry_price = float(m5_ind.iloc[i + 1]["open"])
        r_pct       = get_risk_pct(wins, losses)
        risk_amt    = balance * r_pct
        stake       = calc_stake(atr, entry_price, risk_amt)
        d           = 1 if required == "BUY" else -1
        sl_price    = entry_price - d * atr * SL_ATR_MULT

        trades_today += 1
        active = {
            "open_time":      str(row["time"])[:19],
            "direction":      required,
            "entry":          entry_price,
            "stake":          stake,
            "sl_price":       sl_price,
            "atr":            atr,
            "risk_amount":    risk_amt,
            "risk_pct":       r_pct,
            "balance_before": balance,
            "peak":           entry_price,
            "peak_r":         0.0,
            "locked_pnl":     0.0,
            "partial_done":   False,
            "age":            0,
        }

    # Force-close any trade still open at end of data
    if active is not None:
        last = m5_ind.iloc[-1]
        ep   = float(last["close"])
        pnl  = calc_pnl(active["direction"], active["stake"], active["entry"], ep)
        pnl += active.get("locked_pnl", 0.0)
        balance += pnl
        trade_count += 1
        trades.append(make_record(trade_count, active, ep, last["time"], pnl, balance, "FORCED_EXIT"))

    return trades, balance


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def calc_summary(trades, start_balance, final_balance) -> dict:
    total  = len(trades)
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    net    = sum(t["pnl_usd"] for t in trades)
    wr     = len(wins) / total * 100 if total else 0
    avg_w  = sum(t["pnl_usd"] for t in wins)  / len(wins)   if wins   else 0
    avg_l  = sum(t["pnl_usd"] for t in losses) / len(losses) if losses else 0
    avg_r  = sum(t["r_multiple"] for t in trades) / total if total else 0
    peak = start_balance
    max_dd = 0.0
    for t in trades:
        b = t["balance_after"]
        peak = max(peak, b)
        max_dd = max(max_dd, (peak - b) / peak)
    ret = (final_balance / start_balance - 1) * 100
    return {
        "total": total, "wins": len(wins), "losses": len(losses),
        "win_rate": wr, "net": net, "ret": ret, "max_dd": max_dd * 100,
        "avg_win": avg_w, "avg_loss": avg_l, "avg_r": avg_r,
        "sl_ex": sum(1 for t in trades if t["reason"] == "SL"),
        "ch_ex": sum(1 for t in trades if t["reason"] == "CHANDELIER"),
        "period_start": trades[0]["open_time"][:10] if trades else "",
        "period_end":   trades[-1]["close_time"][:10] if trades else "",
    }


def print_summary(symbol, trades, start_balance, final_balance):
    if not trades:
        print("\nNo trades generated — try --refresh to fetch fresh data.")
        return
    s = calc_summary(trades, start_balance, final_balance)
    print("\n" + "=" * 57)
    print(f"  DERIV MULTIPLIER BACKTEST — {symbol} x{MULTIPLIER}")
    print("=" * 57)
    print(f"  Start balance : ${start_balance:>10,.2f}")
    print(f"  Final balance : ${final_balance:>10,.2f}")
    print(f"  Net P&L       : ${s['net']:>+10,.2f}")
    print(f"  Return        : {s['ret']:>+9.1f}%")
    print(f"  Max drawdown  : {s['max_dd']:>9.1f}%")
    print("-" * 57)
    print(f"  Total trades  : {s['total']}")
    print(f"  Wins          : {s['wins']}  ({s['win_rate']:.1f}%)")
    print(f"  Losses        : {s['losses']}")
    print(f"  SL exits      : {s['sl_ex']}")
    print(f"  Chandelier    : {s['ch_ex']}")
    print(f"  Avg win       : ${s['avg_win']:>+8.2f}")
    print(f"  Avg loss      : ${s['avg_loss']:>+8.2f}")
    print(f"  Avg R/trade   : {s['avg_r']:>+8.2f}R")
    if trades:
        print(f"  Period        : {s['period_start']} -> {s['period_end']}")
    print("=" * 57)


def run_symbol(symbol: str, args) -> dict | None:
    print(f"\n{'=' * 57}")
    print(f"  {symbol}")
    print(f"{'=' * 57}")
    print("Loading data:")
    m15_df = load_candles(symbol, M15_GRAN, args.m15_bars, args.refresh)
    m5_df  = load_candles(symbol, M5_GRAN,  args.m5_bars,  args.refresh)
    if m15_df.empty or m5_df.empty:
        print(f"  No data for {symbol}, skipping.")
        return None
    start = max(m15_df["time"].min(), m5_df["time"].min())
    end   = min(m15_df["time"].max(), m5_df["time"].max())
    m15_df = m15_df[(m15_df["time"] >= start) & (m15_df["time"] <= end)].reset_index(drop=True)
    m5_df  = m5_df[(m5_df["time"] >= start) & (m5_df["time"] <= end)].reset_index(drop=True)
    print(f"\nAligned: {start.date()} -> {end.date()}\n")
    trades, final = simulate(m15_df, m5_df, args.balance)
    print_summary(symbol, trades, args.balance, final)
    if trades:
        out = BASE_DIR / f"backtest_{symbol}_trades.csv"
        pd.DataFrame(trades)[TRADE_FIELDS].to_csv(out, index=False)
        print(f"Trade journal -> {out.name}")
        s = calc_summary(trades, args.balance, final)
        s["symbol"] = symbol
        s["final_balance"] = round(final, 2)
        return s
    return None


def print_comparison_table(results: list):
    if not results:
        return
    print("\n\n" + "=" * 80)
    print("  COMPARISON TABLE — step_trend strategy | All Volatility Indices")
    print("=" * 80)
    print(f"  {'Symbol':<10} {'Trades':>6} {'WinRate':>8} {'Return':>8} {'MaxDD':>7} {'AvgR':>6} {'FinalBal':>12}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["ret"], reverse=True):
        print(f"  {r['symbol']:<10} {r['total']:>6} {r['win_rate']:>7.1f}% {r['ret']:>+7.1f}% "
              f"{r['max_dd']:>6.1f}% {r['avg_r']:>+5.2f}R  ${r['final_balance']:>10,.2f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Deriv Multiplier Backtest")
    parser.add_argument("--balance",  type=float, default=10000.0,  help="Starting balance")
    parser.add_argument("--refresh",  action="store_true",          help="Re-fetch data from Deriv")
    parser.add_argument("--m5-bars",  type=int,   default=55000,    help="M5 bars to fetch")
    parser.add_argument("--m15-bars", type=int,   default=20000,    help="M15 bars to fetch")
    parser.add_argument("--symbol",   type=str,   default="stpRNG", help="Single symbol to backtest")
    parser.add_argument("--compare",    action="store_true", help="Run all volatility indices")
    parser.add_argument("--no-session", action="store_true", help="Trade 24hrs, no session filter")
    parser.add_argument("--relaxed",    action="store_true", help="Loosen confirmation, pullback, and trend thresholds")
    parser.add_argument("--no-kill",    action="store_true", help="Disable DD kill switches (analysis only)")
    parser.add_argument("--account-dd", type=float, default=None,
                        help="Override account DD kill switch (e.g. 0.20 = 20%%)")
    args = parser.parse_args()

    global NO_SESSION_FILTER, RELAXED_MODE, NO_KILL_SWITCH, ACCOUNT_DD_LIMIT
    if args.no_session:
        NO_SESSION_FILTER = True
    if args.relaxed:
        RELAXED_MODE = True
        apply_relaxed_strategy()
    if args.no_kill:
        NO_KILL_SWITCH = True
    if args.account_dd is not None:
        ACCOUNT_DD_LIMIT = args.account_dd

    print(f"\nDeriv Multiplier Backtest | x{MULTIPLIER} | ${args.balance:,.0f} start")
    mode_tag = "[RELAXED]" if RELAXED_MODE else "[STRICT]"
    kill_tag = "NO KILL SWITCH" if NO_KILL_SWITCH else f"AccountDD={ACCOUNT_DD_LIMIT:.0%} DailyDD={DAILY_DD_LIMIT:.0%}"
    if NO_SESSION_FILTER:
        print(f"Session: 24hr (no filter)  {mode_tag}  [{kill_tag}]")
    else:
        print(f"Session: {SESSION_START}:00-{SESSION_END}:00 GMT | Skip hours: {SKIP_HOURS}  {mode_tag}  [{kill_tag}]")

    if args.compare:
        results = []
        for sym in COMPARE_SYMBOLS:
            r = run_symbol(sym, args)
            if r:
                results.append(r)
        print_comparison_table(results)
    else:
        r = run_symbol(args.symbol, args)
        if not r:
            print("No trades generated.")


if __name__ == "__main__":
    main()

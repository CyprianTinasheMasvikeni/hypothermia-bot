y"""
Gold (XAUUSD) Backtester
Strategy: breakout during London/NY overlap session (13:00-17:00 UTC)
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from execution_rules import MAX_HOLD_CANDLES, MAX_TRADES_PER_DAY, START_BALANCE, build_trade_plan
import strategy_gold as strategy

SYMBOL        = "XAUUSD"
TIMEFRAME     = "M15"
BARS          = 15000
WINDOWS       = 5
TP_R          = 2.0     # Gold runs far — use 2R
SESSION_START = 13      # London/NY overlap UTC
SESSION_END   = 17


TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
}


def connect():
    if not mt5.initialize():
        print("MT5 connection failed.")
        return False
    info = mt5.account_info()
    if info:
        print(f"Connected | account={info.login} | server={info.server} | balance=${info.balance:.2f}")
    return True


def fetch(symbol, tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], start_pos, bars)
    if rates is None or len(rates) < 100:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def exit_trade(trade, candle, balance, trades, max_hold):
    trade["age"] += 1
    d = trade["type"]
    price = candle["close"]
    hit_tp    = (d == "BUY" and price >= trade["tp"]) or (d == "SELL" and price <= trade["tp"])
    hit_sl    = (d == "BUY" and price <= trade["sl"]) or (d == "SELL" and price >= trade["sl"])
    timed_out = trade["age"] >= max_hold
    if not (hit_tp or hit_sl or timed_out):
        return None
    reason = "TP" if hit_tp else "SL" if hit_sl else "TIME"
    result = "WIN" if hit_tp else "LOSS"
    pnl = trade["risk_amount"] * trade["tp_r"] if result == "WIN" else -trade["risk_amount"]
    rec = {
        **trade,
        "exit": price, "exit_time": candle["time"],
        "result": result, "reason": reason,
        "pnl": round(pnl, 2), "final_balance": round(balance + pnl, 2),
    }
    trades.append(rec)
    return rec


def simulate(df):
    df = strategy.calculate_indicators(df)
    trades = []
    balance = START_BALANCE
    open_trade = None
    trades_by_day = {}

    for i in range(50, len(df) - 1):
        candle      = df.iloc[i]
        next_candle = df.iloc[i + 1]
        day = candle["time"].date()
        trades_by_day.setdefault(day, 0)

        if open_trade is not None:
            closed = exit_trade(open_trade, candle, balance, trades, MAX_HOLD_CANDLES)
            if closed is not None:
                balance = closed["final_balance"]
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue
        if not (SESSION_START <= candle["time"].hour < SESSION_END):
            continue
        if trades_by_day[day] >= MAX_TRADES_PER_DAY:
            continue

        result = strategy.analyze_setup(df.iloc[: i + 1])
        sig = result.get("signal", "WAIT")
        if sig == "WAIT":
            continue

        atr = candle["atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = next_candle["open"]
        plan = build_trade_plan(sig, entry_price, atr, balance, tp_r=TP_R)
        open_trade = {
            **plan,
            "entry_time":     next_candle["time"],
            "signal_time":    candle["time"],
            "entry_index":    i + 1,
            "balance_before": balance,
            "age":    0,
            "closed": False,
        }

    if open_trade is not None:
        f = df.iloc[-1]
        pnl = -open_trade["risk_amount"]
        trades.append({
            **open_trade,
            "exit": f["close"], "exit_time": f["time"],
            "result": "FORCED", "reason": "FORCED_EXIT",
            "pnl": round(pnl, 2), "final_balance": round(balance + pnl, 2),
        })
    return trades


def monthly_breakdown(all_trades):
    if not all_trades:
        return
    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    df["win"]   = df["result"] == "WIN"
    bm = df.groupby("month").agg(
        trades=("result", "count"),
        wins=("win", "sum"),
        net=("pnl", "sum"),
    ).reset_index()
    bm["wr"] = (bm["wins"] / bm["trades"] * 100).round(1)
    print()
    print("Monthly breakdown:")
    print(f"{'Month':<12} | Trades | WR%   | Net")
    print("-" * 42)
    for _, r in bm.iterrows():
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"{str(r['month']):<12} |  {int(r['trades']):4d}  | {r['wr']:5.1f}% | ${r['net']:+7.2f} [{tag}]")
    print(f"\nOverall: {len(df)} trades | {int(df['win'].sum())} wins | {df['win'].mean()*100:.1f}% WR | net ${df['pnl'].sum():.2f}")


def main():
    if not connect():
        return

    print()
    print(f"Symbol: {SYMBOL} | TF: {TIMEFRAME} | Session: {SESSION_START:02d}:00-{SESSION_END:02d}:00 UTC")
    print(f"TP: {TP_R}R | Windows: {WINDOWS} x {BARS} bars")
    print()

    all_trades = []
    window_results = []

    for w in range(WINDOWS):
        start_pos = w * BARS
        raw = fetch(SYMBOL, TIMEFRAME, BARS, start_pos)
        if raw is None:
            print(f"Window {w+1}: no data")
            continue
        trades = simulate(raw)
        wins = sum(1 for t in trades if t["result"] == "WIN")
        net  = sum(t["pnl"] for t in trades)
        wr   = wins / len(trades) * 100 if trades else 0
        tag  = "PROFIT" if net > 0 else "LOSS" if trades else "NO TRADES"
        print(f"Window {w+1} | trades={len(trades):3d} | WR={wr:5.1f}% | net=${net:+7.2f} [{tag}]")
        all_trades.extend(trades)
        window_results.append({"net": net, "trades": len(trades)})

    mt5.shutdown()

    profitable = sum(1 for r in window_results if r["net"] > 0)
    print()
    print(f"Profitable windows: {profitable}/{len(window_results)}")
    monthly_breakdown(all_trades)


if __name__ == "__main__":
    main()

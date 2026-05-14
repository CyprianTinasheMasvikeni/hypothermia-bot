import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from execution_rules import MAX_HOLD_CANDLES, MAX_TRADES_PER_DAY, START_BALANCE, TP_R, build_trade_plan
import strategy_real_markets as strategy

PAIRS         = ["XAUUSD", "EURUSD", "GBPUSD", "Wall Street 30", "US Tech 100", "US SP 500"]
TREND_TF      = "M15"
ENTRY_TF      = "M5"
BARS          = 15000
WINDOWS       = 5
SESSION_START = 9
SESSION_END   = 19

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
}


def connect():
    if not mt5.initialize():
        print("MT5 connection failed. Open MT5 and log in first.")
        return False
    info = mt5.account_info()
    if info:
        print(f"Connected | account={info.login} | server={info.server} | balance=${info.balance:.2f}")
    return True


def fetch(symbol, tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], start_pos, bars)
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def exit_trade(trade, candle, balance, trades, max_hold):
    trade["age"] += 1
    d = trade["type"]
    price = candle["close"]
    hit_tp = (d == "BUY" and price >= trade["tp"]) or (d == "SELL" and price <= trade["tp"])
    hit_sl = (d == "BUY" and price <= trade["sl"]) or (d == "SELL" and price >= trade["sl"])
    timed_out = trade["age"] >= max_hold
    if not (hit_tp or hit_sl or timed_out):
        return None
    reason = "TP" if hit_tp else "SL" if hit_sl else "TIME"
    result = "WIN" if hit_tp else "LOSS"
    pnl = trade["risk_amount"] * trade["tp_r"] if result == "WIN" else -trade["risk_amount"]
    rec = {
        **trade,
        "exit": price,
        "exit_time": candle["time"],
        "result": result,
        "reason": reason,
        "pnl": round(pnl, 2),
        "final_balance": round(balance + pnl, 2),
    }
    trades.append(rec)
    return rec


def simulate(symbol, trend_df, entry_df):
    trades = []
    balance = START_BALANCE
    open_trade = None
    trades_by_day = {}

    for i in range(220, len(entry_df) - 1):
        candle = entry_df.iloc[i]
        next_candle = entry_df.iloc[i + 1]
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

        trend_slice = trend_df[trend_df["time"] <= candle["time"]].copy()
        if len(trend_slice) < 220:
            continue
        entry_slice = entry_df.iloc[: i + 1].copy()

        t_res = strategy.analyze_setup(trend_slice)
        e_res = strategy.analyze_setup(entry_slice)
        bias  = t_res.get("checks", {}).get("trend_bias", "NEUTRAL")
        sig   = e_res.get("signal", "WAIT")

        if sig == "WAIT" or bias not in {"STRONG_BUY", "STRONG_SELL"}:
            continue
        if sig == "BUY"  and bias != "STRONG_BUY":
            continue
        if sig == "SELL" and bias != "STRONG_SELL":
            continue

        atr = entry_slice["atr"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        plan = build_trade_plan(sig, next_candle["open"], atr, balance, tp_r=TP_R)
        open_trade = {
            **plan,
            "symbol":         symbol,
            "entry_time":     next_candle["time"],
            "signal_time":    candle["time"],
            "entry_index":    i + 1,
            "balance_before": balance,
            "age":    0,
            "closed": False,
        }

    if open_trade is not None:
        f = entry_df.iloc[-1]
        pnl = -open_trade["risk_amount"]
        trades.append({
            **open_trade,
            "exit": f["close"], "exit_time": f["time"],
            "result": "FORCED", "reason": "FORCED_EXIT",
            "pnl": round(pnl, 2), "final_balance": round(balance + pnl, 2),
        })
    return trades


def print_pair(symbol, trades):
    if not trades:
        print(f"{symbol:<10} | trades=  0 | WR=  0.0% | net=     $0.00 [NO TRADES]")
        return {"symbol": symbol, "trades": 0, "wins": 0, "wr": 0.0, "net": 0.0}
    wins = sum(1 for t in trades if t["result"] == "WIN")
    net  = sum(t["pnl"] for t in trades)
    wr   = wins / len(trades) * 100
    tag  = "PROFIT" if net > 0 else "LOSS"
    print(f"{symbol:<10} | trades={len(trades):3d} | WR={wr:5.1f}% | net={net:+9.2f} [{tag}]")
    return {"symbol": symbol, "trades": len(trades), "wins": wins, "wr": wr, "net": net}


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
    print("Monthly breakdown (all pairs combined):")
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
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"TF: {TREND_TF} trend + {ENTRY_TF} entry | {WINDOWS} windows x {BARS} bars | Session {SESSION_START:02d}:00-{SESSION_END:02d}:00")
    print()

    all_results = []
    all_trades  = []

    for symbol in PAIRS:
        pair_trades = []
        for w in range(WINDOWS):
            t_raw = fetch(symbol, TREND_TF, BARS, w * BARS)
            e_raw = fetch(symbol, ENTRY_TF,  BARS, w * BARS)
            if t_raw is None or e_raw is None:
                continue
            pair_trades.extend(simulate(
                symbol,
                strategy.calculate_indicators(t_raw),
                strategy.calculate_indicators(e_raw),
            ))
        r = print_pair(symbol, pair_trades)
        all_results.append(r)
        all_trades.extend(pair_trades)

    mt5.shutdown()

    print()
    print("=" * 50)
    profitable = sum(1 for r in all_results if r["net"] > 0)
    print(f"Profitable pairs: {profitable}/{len(all_results)}")
    print()
    print("Ranking:")
    for i, r in enumerate(sorted(all_results, key=lambda x: x["net"], reverse=True), 1):
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"  {i}. {r['symbol']:<10} | {r['trades']:3d} trades | {r['wr']:.1f}% WR | ${r['net']:+.2f} [{tag}]")

    monthly_breakdown(all_trades)


if __name__ == "__main__":
    main()

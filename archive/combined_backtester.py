"""
Combined backtester: step_trend on M30 as market regime filter,
scalper on M5 for entries. Only trades when step_trend says
STRONG_BUY or STRONG_SELL — scalper sits out in choppy markets.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import MetaTrader5 as mt5
import pandas as pd

from bot import load_strategy
from execution_rules import (
    MAX_HOLD_CANDLES, MAX_TRADES_PER_DAY, START_BALANCE, TP_R,
    build_trade_plan, close_trade as finalize_trade,
)
from step_trend_backtester import fetch_data, prepare_analysis, get_latest_trend_slice, exit_trade_if_needed

SYMBOL       = "Step Index"
REGIME_TF    = "M30"   # step_trend reads this for market regime
ENTRY_TF     = "M5"    # scalper reads this for entries
BARS         = 15000
SESSION_START = 9
SESSION_END   = 19
WINDOWS      = [0, 15000, 30000, 45000, 60000]


def get_regime(step_trend_module, regime_slice):
    """Return STRONG_BUY, STRONG_SELL, or NEUTRAL from step_trend analysis."""
    if regime_slice is None or len(regime_slice) < 220:
        return "NEUTRAL"
    analysis = step_trend_module.analyze_setup(regime_slice)
    bias = analysis.get("checks", {}).get("trend_bias", "NEUTRAL")
    choppy = analysis.get("checks", {}).get("market_choppy", True)
    if choppy or bias == "NEUTRAL":
        return "NEUTRAL"
    if bias in {"STRONG_BUY", "WEAK_BUY"}:
        return "BUY"
    if bias in {"STRONG_SELL", "WEAK_SELL"}:
        return "SELL"
    return "NEUTRAL"


def get_scalper_signal(scalper_module, entry_slice):
    """Return BUY, SELL, or WAIT from scalper analysis."""
    if len(entry_slice) < 60:
        return "WAIT"
    analysis = scalper_module.analyze_setup(entry_slice)
    return analysis.get("signal", "WAIT")


def simulate(step_trend_mod, scalper_mod, regime_df, entry_df):
    trades = []
    balance = START_BALANCE
    open_trade = None
    trades_by_day = {}

    for i in range(220, len(entry_df) - 1):
        candle = entry_df.iloc[i]
        next_candle = entry_df.iloc[i + 1]
        trade_day = candle["time"].date()
        trades_by_day.setdefault(trade_day, 0)

        if open_trade is not None:
            closed = exit_trade_if_needed(open_trade, candle, balance, trades, MAX_HOLD_CANDLES)
            if closed is not None:
                balance = closed["final_balance"]
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue
        if not (SESSION_START <= candle["time"].hour < SESSION_END):
            continue
        if trades_by_day[trade_day] >= MAX_TRADES_PER_DAY:
            continue

        # Step 1: check market regime using step_trend on M30
        regime_slice = get_latest_trend_slice(regime_df, candle["time"])
        regime = get_regime(step_trend_mod, regime_slice)
        if regime == "NEUTRAL":
            continue

        # Step 2: check scalper entry signal on M5
        entry_slice = entry_df.iloc[: i + 1].copy()
        scalper_signal = get_scalper_signal(scalper_mod, entry_slice)
        if scalper_signal == "WAIT":
            continue

        # Step 3: both must agree on direction
        if scalper_signal != regime:
            continue

        atr = entry_slice["atr"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = next_candle["open"]
        plan = build_trade_plan(scalper_signal, entry_price, atr, balance, tp_r=TP_R)
        open_trade = {
            **plan,
            "entry_time": next_candle["time"],
            "signal_time": candle["time"],
            "regime": regime,
            "entry_index": i + 1,
            "balance_before": balance,
            "age": 0,
            "closed": False,
        }

    if open_trade is not None:
        final = entry_df.iloc[-1]
        trades.append(finalize_trade(open_trade, final["close"], final["time"], "FORCED_EXIT", balance))

    return trades


def summarize(trades, label):
    if not trades:
        print(f"{label} | trades=0 | no activity")
        return {"net": 0, "trades": 0, "wins": 0, "wr": 0}
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    net = sum(t["pnl"] for t in trades)
    wr = wins / len(trades) * 100
    print(f"{label} | trades={len(trades)} | wins={wins} | losses={losses} | WR={wr:.1f}% | net=${net:.2f}")
    return {"net": net, "trades": len(trades), "wins": wins, "wr": wr}


def monthly_breakdown(all_trades):
    if not all_trades:
        return
    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    df["win"] = df["result"] == "WIN"
    by_month = df.groupby("month").agg(
        trades=("result", "count"),
        wins=("win", "sum"),
        net=("pnl", "sum"),
    ).reset_index()
    by_month["wr"] = (by_month["wins"] / by_month["trades"] * 100).round(1)
    print()
    print("Monthly breakdown (combined across all windows):")
    print(f"{'Month':<12} | Trades | WR%   | Net $100 | Net $1000")
    print("-" * 55)
    for _, r in by_month.iterrows():
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"{str(r['month']):<12} |  {int(r['trades']):4d}  | {r['wr']:5.1f}% | ${r['net']:+7.2f}  | ${r['net']*10:+8.2f}  [{tag}]")
    print()
    print(f"Overall: {len(df)} trades | {df['win'].sum()} wins | {df['win'].mean()*100:.1f}% WR | net ${df['pnl'].sum():.2f} on $100")


def main():
    if not mt5.initialize():
        print("MT5 failed"); return

    print("Connected to MT5")
    print(f"Regime filter: step_trend on {REGIME_TF} | Entry: scalper on {ENTRY_TF}")
    print(f"Session: {SESSION_START:02d}:00 – {SESSION_END:02d}:00 | Symbol: {SYMBOL}")
    print()

    step_trend_mod = load_strategy("step_trend")
    scalper_mod = load_strategy("scalper")

    all_trades = []
    window_results = []

    for start_pos in WINDOWS:
        regime_raw = fetch_data(SYMBOL, REGIME_TF, BARS, start_pos=start_pos)
        entry_raw = fetch_data(SYMBOL, ENTRY_TF, BARS, start_pos=start_pos)
        if regime_raw is None or entry_raw is None:
            print(f"window_start={start_pos} | no data"); continue

        regime_df = prepare_analysis(regime_raw, step_trend_mod)
        entry_df = prepare_analysis(entry_raw, scalper_mod)

        trades = simulate(step_trend_mod, scalper_mod, regime_df, entry_df)
        all_trades.extend(trades)

        label = f"window_start={start_pos}"
        r = summarize(trades, label)
        window_results.append(r)

    mt5.shutdown()

    profitable = sum(1 for r in window_results if r["net"] > 0)
    total = len(window_results)
    print()
    print(f"Profitable windows: {profitable}/{total}")
    monthly_breakdown(all_trades)


if __name__ == "__main__":
    main()

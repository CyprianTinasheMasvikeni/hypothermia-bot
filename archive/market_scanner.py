"""
Multi-market scanner — tests the step_trend strategy across all major
Deriv synthetic indices and ranks them by profitability.
"""
import sys
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from bot import combine_analyses, load_strategy
from execution_rules import (
    MAX_HOLD_CANDLES,
    MAX_TRADES_PER_DAY,
    START_BALANCE,
    TP_R,
    build_trade_plan,
    close_trade as finalize_trade,
)
from step_trend_backtester import (
    fetch_data,
    prepare_analysis,
    get_latest_trend_slice,
    build_signal,
    exit_trade_if_needed,
    calculate_summary,
)

TREND_TIMEFRAME = "M15"
ENTRY_TIMEFRAME = "M5"
BARS = 75000
STRATEGY = "step_trend"
SESSION_START = 9
SESSION_END = 19

DERIV_MARKETS = [
    "Step Index",
    "Volatility 10 Index",
    "Volatility 25 Index",
    "Volatility 50 Index",
    "Volatility 75 Index",
    "Volatility 100 Index",
    "Crash 300 Index",
    "Crash 500 Index",
    "Crash 1000 Index",
    "Boom 300 Index",
    "Boom 500 Index",
    "Boom 1000 Index",
    "Jump 10 Index",
    "Jump 25 Index",
    "Jump 50 Index",
    "Jump 75 Index",
    "Jump 100 Index",
]

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}


def simulate(strategy_module, trend_df, entry_df, symbol):
    trades = []
    balance = START_BALANCE
    open_trade = None
    trades_by_day = {}
    min_bars = 220

    for i in range(min_bars, len(entry_df) - 1):
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

        entry_slice = entry_df.iloc[: i + 1].copy()
        trend_slice = get_latest_trend_slice(trend_df, candle["time"])
        if trend_slice is None or len(trend_slice) < 220:
            continue

        signal_pack = build_signal(strategy_module, trend_slice, entry_slice)
        signal = signal_pack["signal"]
        if signal not in {"BUY", "SELL"}:
            continue

        atr = entry_slice["atr"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = next_candle["open"]
        plan = build_trade_plan(signal, entry_price, atr, balance, tp_r=TP_R)

        open_trade = {
            **plan,
            "entry_time": next_candle["time"],
            "signal_time": candle["time"],
            "entry_index": i + 1,
            "balance_before": balance,
            "age": 0,
            "higher_bias": signal_pack["higher_bias"],
            "bias_strength": signal_pack["bias_strength"],
            "reason": signal_pack["reason"],
            "closed": False,
        }

    if open_trade is not None:
        final = entry_df.iloc[-1]
        trades.append(finalize_trade(open_trade, final["close"], final["time"], "FORCED_EXIT", balance))

    return trades


def scan_market(strategy_module, symbol):
    trend_raw = fetch_data(symbol, TREND_TIMEFRAME, BARS)
    entry_raw = fetch_data(symbol, ENTRY_TIMEFRAME, BARS)

    if trend_raw is None or entry_raw is None:
        return None

    trend_df = prepare_analysis(trend_raw, strategy_module)
    entry_df = prepare_analysis(entry_raw, strategy_module)
    trades = simulate(strategy_module, trend_df, entry_df, symbol)

    if not trades:
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "final_balance": START_BALANCE,
            "profitable_months": 0,
            "total_months": 0,
        }

    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    monthly = df.groupby("month")["pnl"].sum()
    profitable_months = (monthly > 0).sum()
    total_months = len(monthly)

    wins = sum(1 for t in trades if t["result"] == "WIN")
    losses = sum(1 for t in trades if t["result"] == "LOSS")
    net = sum(t["pnl"] for t in trades)
    win_rate = wins / len(trades) * 100 if trades else 0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "net_profit": round(net, 2),
        "final_balance": round(START_BALANCE + net, 2),
        "profitable_months": int(profitable_months),
        "total_months": int(total_months),
    }


def main():
    if not mt5.initialize():
        print(f"MT5 failed: {mt5.last_error()}")
        return

    print(f"Connected to MT5")
    print(f"Strategy: {STRATEGY} | Timeframes: {TREND_TIMEFRAME}+{ENTRY_TIMEFRAME} | Session: {SESSION_START:02d}:00-{SESSION_END:02d}:00 | Bars: {BARS}")
    print(f"Scanning {len(DERIV_MARKETS)} markets...\n")

    strategy_module = load_strategy(STRATEGY)
    results = []

    for symbol in DERIV_MARKETS:
        print(f"  Testing {symbol}...", end=" ", flush=True)
        result = scan_market(strategy_module, symbol)
        if result is None:
            print("no data")
            continue
        results.append(result)
        print(f"done — {result['trades']} trades | {result['win_rate']}% WR | net ${result['net_profit']:.2f}")

    mt5.shutdown()

    if not results:
        print("No results.")
        return

    results.sort(key=lambda x: x["net_profit"], reverse=True)

    print("\n" + "=" * 80)
    print("MARKET RANKING — Best to Worst")
    print("=" * 80)
    print(f"{'Rank':<5} {'Symbol':<28} {'Trades':>6} {'WR%':>6} {'Net $100':>10} {'Net $1000':>10} {'Profit Months':>14}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        month_str = f"{r['profitable_months']}/{r['total_months']}"
        tag = " <<< BEST" if rank == 1 else ""
        print(
            f"{rank:<5} {r['symbol']:<28} {r['trades']:>6} {r['win_rate']:>5.1f}% "
            f"${r['net_profit']:>8.2f} ${r['net_profit']*10:>8.2f}  {month_str:>10}{tag}"
        )

    print("\nTop 3 markets to trade:")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['symbol']} — {r['win_rate']}% win rate | ${r['net_profit']:.2f} net on $100 | {r['profitable_months']}/{r['total_months']} months profitable")


if __name__ == "__main__":
    main()

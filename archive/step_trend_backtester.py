import argparse
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from bot import combine_analyses, load_strategy
from config import ENTRY_TIMEFRAME, PAIR, STRATEGY, TREND_TIMEFRAME
from execution_rules import (
    MAX_TRADES_PER_DAY,
    START_BALANCE,
    TP_R,
    build_trade_plan,
    close_trade as finalize_trade,
)
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
}

DEFAULT_TP_R = TP_R
DEFAULT_MAX_HOLD_CANDLES = 96   # 96 x M5 = 8 hours, matches live bot chandelier
DEFAULT_MAX_TRADES_PER_DAY = MAX_TRADES_PER_DAY
OUTPUT_CSV = "strategy_step_trend_backtest.csv"
DEFAULT_TIMEFRAME_PAIRS = [
    (TREND_TIMEFRAME, ENTRY_TIMEFRAME),
    ("M15", "M5"),
    ("M5", "M1"),
]


def connect_mt5():
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    print("Connected to MT5")
    return True


def fetch_data(symbol, timeframe, bars, start_pos=0):
    mt5_timeframe = TIMEFRAME_MAP.get(timeframe)
    if mt5_timeframe is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, start_pos, bars)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df.columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def prepare_analysis(df, strategy_module):
    closed_df = df.copy()
    return strategy_module.calculate_indicators(closed_df)


def get_latest_trend_slice(trend_df, entry_time):
    eligible = trend_df[trend_df["time"] <= entry_time]
    if eligible.empty:
        return None
    return eligible


def build_signal(strategy_module, trend_slice, entry_slice):
    if hasattr(strategy_module, "analyze_setup"):
        trend_analysis = strategy_module.analyze_setup(trend_slice)
        entry_analysis = strategy_module.analyze_setup(entry_slice)
        trend_analysis["latest_closed_candle"] = trend_slice.iloc[-1][["time", "open", "high", "low", "close"]]
        entry_analysis["latest_closed_candle"] = entry_slice.iloc[-1][["time", "open", "high", "low", "close"]]
        combined = combine_analyses(trend_analysis, entry_analysis)
        combined["trend_analysis"] = trend_analysis
        combined["entry_analysis"] = entry_analysis
        return combined

    # Fallback for strategies that only have generate_signal
    trend_signal = strategy_module.generate_signal(trend_slice)
    entry_signal = strategy_module.generate_signal(entry_slice)

    if trend_signal == entry_signal and trend_signal in {"BUY", "SELL"}:
        final_signal = trend_signal
        reason = f"Both timeframes agree: {final_signal}"
        bias = trend_signal
        strength = "strong"
    else:
        final_signal = "WAIT"
        reason = f"No agreement: trend={trend_signal}, entry={entry_signal}"
        bias = "NEUTRAL"
        strength = "neutral"

    dummy_candle = entry_slice.iloc[-1][["time", "open", "high", "low", "close"]]
    trend_analysis = {"signal": trend_signal, "reason": reason, "checks": {}, "latest_closed_candle": trend_slice.iloc[-1][["time", "open", "high", "low", "close"]], "timeframe": "trend"}
    entry_analysis = {"signal": entry_signal, "reason": reason, "checks": {}, "latest_closed_candle": dummy_candle, "next_open_candle": dummy_candle, "timeframe": "entry"}
    return {
        "signal": final_signal,
        "reason": reason,
        "higher_bias": bias,
        "bias_strength": strength,
        "bias_reason": reason,
        "trend_analysis": trend_analysis,
        "entry_analysis": entry_analysis,
    }


def simulate_strategy(
    strategy_module,
    trend_df,
    entry_df,
    tp_r=DEFAULT_TP_R,
    max_hold_candles=DEFAULT_MAX_HOLD_CANDLES,
    max_trades_per_day=DEFAULT_MAX_TRADES_PER_DAY,
    strong_bias_only=False,
):
    trades = []
    balance = START_BALANCE
    open_trade = None
    trades_by_day = {}

    min_entry_bars = 220
    for i in range(min_entry_bars, len(entry_df) - 1):
        candle = entry_df.iloc[i]
        next_candle = entry_df.iloc[i + 1]
        trade_day = candle["time"].date()
        trades_by_day.setdefault(trade_day, 0)

        if open_trade is not None:
            closed_trade = exit_trade_if_needed(open_trade, candle, balance, trades, max_hold_candles)
            if closed_trade is not None:
                balance = closed_trade["final_balance"]
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue

        # Golden session: 09:00 – 19:00 only
        if not (9 <= candle["time"].hour < 19):
            continue

        if trades_by_day[trade_day] >= max_trades_per_day:
            continue

        entry_slice = entry_df.iloc[: i + 1].copy()
        trend_slice = get_latest_trend_slice(trend_df, candle["time"])
        if trend_slice is None or len(trend_slice) < 220:
            continue

        signal_pack = build_signal(strategy_module, trend_slice, entry_slice)
        signal = signal_pack["signal"]
        if signal not in {"BUY", "SELL"}:
            continue
        if strong_bias_only and signal_pack.get("bias_strength") != "strong":
            continue

        atr = entry_slice["atr"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            continue

        entry_price = next_candle["open"]
        plan = build_trade_plan(signal, entry_price, atr, balance, tp_r=tp_r)

        open_trade = {
            **plan,
            "entry_time": next_candle["time"],
            "signal_time": candle["time"],
            "trend_time": trend_slice.iloc[-1]["time"],
            "entry_index": i + 1,
            "balance_before": balance,
            "age": 0,
            "atr": atr,
            "peak": entry_price,
            "partial_done": False,
            "locked_pnl": 0.0,
            "reason": signal_pack["reason"],
            "higher_bias": signal_pack["higher_bias"],
            "bias_strength": signal_pack["bias_strength"],
            "entry_confirmation": signal_pack["entry_analysis"]["checks"].get("confirmation_type", "none"),
            "closed": False,
        }

    if open_trade is not None:
        final_candle = entry_df.iloc[-1]
        trades.append(finalize_trade(open_trade, final_candle["close"], final_candle["time"], "FORCED_EXIT", balance))

    return trades


CHANDELIER_MULT = 3.0   # SL trails 3xATR behind the highest peak reached
PARTIAL_R       = 2.0   # Close 50% of position at this R multiple
PARTIAL_PCT     = 0.5   # Fraction of position to close at partial target


def exit_trade_if_needed(trade, candle, balance, trades, max_hold_candles):
    trade["age"] += 1
    t   = trade["type"]
    hi  = candle["high"]
    lo  = candle["low"]
    atr = trade["atr"]
    d   = 1 if t == "BUY" else -1

    # Partial close: lock in PARTIAL_PCT of position at PARTIAL_R
    if not trade.get("partial_done"):
        partial_target = trade["entry"] + d * atr * PARTIAL_R
        hit_partial = (t == "BUY" and hi >= partial_target) or \
                      (t == "SELL" and lo <= partial_target)
        if hit_partial:
            locked = (partial_target - trade["entry"]) * d * (trade["size"] * PARTIAL_PCT)
            trade["locked_pnl"]   = locked
            trade["size"]        *= (1.0 - PARTIAL_PCT)   # remaining position
            trade["partial_done"] = True

    # Update peak for Chandelier (uses original entry; SL NOT moved to breakeven)
    if t == "BUY":
        trade["peak"] = max(trade.get("peak", trade["entry"]), hi)
    else:
        trade["peak"] = min(trade.get("peak", trade["entry"]), lo)

    # Chandelier SL: trails 3xATR behind the peak, only moves in our favour
    chand_sl = trade["peak"] - d * atr * CHANDELIER_MULT
    if t == "BUY":
        trade["sl"] = max(trade["sl"], chand_sl)
    else:
        trade["sl"] = min(trade["sl"], chand_sl)

    sl_hit = (t == "BUY" and lo <= trade["sl"]) or (t == "SELL" and hi >= trade["sl"])
    timed  = trade["age"] >= max_hold_candles

    def _close(exit_price, exit_time, result):
        closed = finalize_trade(trade, exit_price, exit_time, result, balance)
        locked = trade.get("locked_pnl", 0.0)
        closed["pnl"]           += locked
        closed["final_balance"] += locked
        # Reclassify result based on total pnl so stats are accurate
        if closed["pnl"] > 0 and closed["result"] == "LOSS":
            closed["result"] = "WIN"
        elif closed["pnl"] <= 0 and closed["result"] == "WIN":
            closed["result"] = "LOSS"
        closed["r_multiple"] = closed["pnl"] / trade.get("risk_amount", 1)
        closed["partial_done"] = trade.get("partial_done", False)
        return closed

    if sl_hit:
        result = "WIN" if (trade["sl"] > trade["entry"] and t == "BUY") or \
                          (trade["sl"] < trade["entry"] and t == "SELL") else "LOSS"
        closed_trade = _close(trade["sl"], candle["time"], result)
        trades.append(closed_trade)
        return closed_trade
    if timed:
        closed_trade = _close(candle["close"], candle["time"], "TIME_EXIT")
        trades.append(closed_trade)
        return closed_trade
    return None


def summarize_trades(trades, strategy_name, trend_timeframe, entry_timeframe, strong_bias_only=False, pair=None):
    total = len(trades)
    wins = sum(1 for trade in trades if trade["result"] == "WIN")
    losses = sum(1 for trade in trades if trade["result"] == "LOSS")
    timed = sum(1 for trade in trades if trade["result"] == "TIME_EXIT")
    forced = sum(1 for trade in trades if trade["result"] == "FORCED_EXIT")
    net_profit = sum(trade["pnl"] for trade in trades)
    win_rate = (wins / total * 100) if total else 0
    final_balance = trades[-1]["final_balance"] if trades else START_BALANCE

    print("")
    print(f"Strategy: {strategy_name}")
    print(f"Pair: {pair if pair else PAIR}")
    print(f"Trend timeframe: {trend_timeframe}")
    print(f"Entry timeframe: {entry_timeframe}")
    print(f"Strong bias only: {strong_bias_only}")
    print(f"Total trades: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Time exits: {timed}")
    print(f"Forced exits: {forced}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Net profit: {net_profit:.2f}")
    print(f"Final balance: {final_balance:.2f}")


def calculate_summary(trades, trend_timeframe, entry_timeframe):
    total = len(trades)
    wins = sum(1 for trade in trades if trade["result"] == "WIN")
    losses = sum(1 for trade in trades if trade["result"] == "LOSS")
    timed = sum(1 for trade in trades if trade["result"] == "TIME_EXIT")
    forced = sum(1 for trade in trades if trade["result"] == "FORCED_EXIT")
    net_profit = sum(trade["pnl"] for trade in trades)
    win_rate = (wins / total * 100) if total else 0
    final_balance = trades[-1]["final_balance"] if trades else START_BALANCE

    return {
        "trend_timeframe": trend_timeframe,
        "entry_timeframe": entry_timeframe,
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "time_exits": timed,
        "forced_exits": forced,
        "win_rate": round(win_rate, 2),
        "net_profit": round(net_profit, 2),
        "final_balance": round(final_balance, 2),
    }


def run_backtest_for_pair(
    strategy_name,
    trend_timeframe,
    entry_timeframe,
    bars,
    tp_r,
    max_hold_candles,
    max_trades_per_day,
    strong_bias_only=False,
    start_pos=0,
    pair=None,
):
    symbol = pair if pair else PAIR
    strategy_module = load_strategy(strategy_name)
    trend_raw = fetch_data(symbol, trend_timeframe, bars=bars, start_pos=start_pos)
    entry_raw = fetch_data(symbol, entry_timeframe, bars=bars, start_pos=start_pos)
    if trend_raw is None or entry_raw is None:
        return None, None

    trend_df = prepare_analysis(trend_raw, strategy_module)
    entry_df = prepare_analysis(entry_raw, strategy_module)
    trades = simulate_strategy(
        strategy_module,
        trend_df,
        entry_df,
        tp_r=tp_r,
        max_hold_candles=max_hold_candles,
        max_trades_per_day=max_trades_per_day,
        strong_bias_only=strong_bias_only,
    )
    summary = calculate_summary(trades, trend_timeframe, entry_timeframe)
    summary["strategy"] = strategy_name
    summary["strong_bias_only"] = strong_bias_only
    summary["start_pos"] = start_pos
    if trades:
        summary["first_trade_time"] = str(trades[0]["entry_time"])
        summary["last_trade_time"] = str(trades[-1]["exit_time"])
    else:
        summary["first_trade_time"] = ""
        summary["last_trade_time"] = ""
    return trades, summary


def print_timeframe_comparison(results):
    if not results:
        print("No timeframe comparison results available.")
        return

    print("")
    print("Timeframe comparison:")
    for result in sorted(results, key=lambda item: item["net_profit"], reverse=True):
        print(
            f"{result['trend_timeframe']} + {result['entry_timeframe']} | "
            f"strategy={result.get('strategy', STRATEGY)} | "
            f"strong_only={result.get('strong_bias_only', False)} | "
            f"start_pos={result.get('start_pos', 0)} | "
            f"trades={result['total_trades']} | wins={result['wins']} | "
            f"losses={result['losses']} | win_rate={result['win_rate']:.2f}% | "
            f"net={result['net_profit']:.2f} | balance={result['final_balance']:.2f}"
        )


def print_window_comparison(results):
    if not results:
        print("No window comparison results available.")
        return

    print("")
    print("Window comparison:")
    for result in sorted(results, key=lambda item: item["start_pos"]):
        print(
            f"window_start={result['start_pos']} | "
            f"strategy={result.get('strategy', STRATEGY)} | "
            f"trades={result['total_trades']} | wins={result['wins']} | "
            f"losses={result['losses']} | win_rate={result['win_rate']:.2f}% | "
            f"net={result['net_profit']:.2f} | balance={result['final_balance']:.2f} | "
            f"first={result['first_trade_time']} | last={result['last_trade_time']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest the Step Trend multi-timeframe strategy.")
    parser.add_argument("--bars", type=int, default=15000, help="Bars to fetch for each timeframe.")
    parser.add_argument(
        "--strategy",
        default=STRATEGY,
        help=f"Strategy module to backtest. Default: {STRATEGY}",
    )
    parser.add_argument(
        "--pair",
        default=PAIR,
        help=f"Market symbol to backtest. Default: {PAIR}",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=0,
        help="Starting MT5 position offset for fetching historical bars.",
    )
    parser.add_argument(
        "--trend-timeframe",
        default=TREND_TIMEFRAME,
        help=f"Higher timeframe for market bias. Default: {TREND_TIMEFRAME}",
    )
    parser.add_argument(
        "--entry-timeframe",
        default=ENTRY_TIMEFRAME,
        help=f"Lower timeframe for entries. Default: {ENTRY_TIMEFRAME}",
    )
    parser.add_argument("--tp-r", type=float, default=DEFAULT_TP_R, help="Take-profit multiple of risk.")
    parser.add_argument(
        "--max-hold-candles",
        type=int,
        default=DEFAULT_MAX_HOLD_CANDLES,
        help="Maximum number of entry candles to hold a trade.",
    )
    parser.add_argument(
        "--max-trades-per-day",
        type=int,
        default=DEFAULT_MAX_TRADES_PER_DAY,
        help="Daily trade cap for the backtest.",
    )
    parser.add_argument(
        "--compare-timeframes",
        action="store_true",
        help="Compare the default timeframe pairs side by side.",
    )
    parser.add_argument(
        "--strong-bias-only",
        action="store_true",
        help="Only take trades when the higher timeframe bias is strong.",
    )
    parser.add_argument(
        "--compare-windows",
        action="store_true",
        help="Run the current timeframe pair across multiple historical windows.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=3,
        help="Number of historical windows to compare when using --compare-windows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not connect_mt5():
        return

    try:
        if args.compare_timeframes:
            comparison_results = []
            for trend_timeframe, entry_timeframe in DEFAULT_TIMEFRAME_PAIRS:
                trades, summary = run_backtest_for_pair(
                    args.strategy,
                    trend_timeframe,
                    entry_timeframe,
                    args.bars,
                    args.tp_r,
                    args.max_hold_candles,
                    args.max_trades_per_day,
                    args.strong_bias_only,
                    args.start_pos,
                    pair=args.pair,
                )
                if summary is not None:
                    comparison_results.append(summary)
            print_timeframe_comparison(comparison_results)
        elif args.compare_windows:
            window_results = []
            for window_index in range(args.window_count):
                start_pos = args.start_pos + (window_index * args.bars)
                trades, summary = run_backtest_for_pair(
                    args.strategy,
                    args.trend_timeframe,
                    args.entry_timeframe,
                    args.bars,
                    args.tp_r,
                    args.max_hold_candles,
                    args.max_trades_per_day,
                    args.strong_bias_only,
                    start_pos,
                    pair=args.pair,
                )
                if summary is not None:
                    window_results.append(summary)
            print_window_comparison(window_results)
        else:
            trades, _ = run_backtest_for_pair(
                    args.strategy,
                    args.trend_timeframe,
                    args.entry_timeframe,
                    args.bars,
                args.tp_r,
                args.max_hold_candles,
                args.max_trades_per_day,
                args.strong_bias_only,
                args.start_pos,
                pair=args.pair,
            )
            if trades is None:
                print("Unable to fetch enough MT5 data for the backtest.")
                return

            summarize_trades(
                trades,
                args.strategy,
                args.trend_timeframe,
                args.entry_timeframe,
                args.strong_bias_only,
                pair=args.pair,
            )
            output_path = Path(__file__).resolve().parent / OUTPUT_CSV
            pd.DataFrame(trades).to_csv(output_path, index=False)
            print(f"Trades exported to {output_path.name}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

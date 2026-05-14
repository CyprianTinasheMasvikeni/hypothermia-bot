"""
EURUSD Backtester
Strategy: regime-adaptive (trend mode + range mode)
H4 for regime + bias, M15 for entry
Session: 08:00-17:00 GMT
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import strategy_eurusd as strategy
from execution_rules import START_BALANCE

SYMBOL        = "EURUSD"   # overridden by multi-pair scan in main()
M15_TF        = "M15"
H4_TF         = "H4"
BARS          = 30000
H4_BARS       = 4000
WINDOWS       = 8
SESSION_START = 14  # NY session open
SESSION_END   = 20

# Risk settings
RISK_PCT      = 0.01    # 1% per trade
SL_ATR_MULT   = 1.2     # SL = 1.2x ATR
TP_R          = 2.0     # minimum 2:1 RR
PARTIAL_R     = 1.5     # partial close at 1.5R
MAX_HOLD      = 20      # max candles before time exit
MAX_PER_DAY   = 3

# Kill switches
DAILY_DD_LIMIT   = 0.03  # 3% daily drawdown = stop for the day
ACCOUNT_DD_LIMIT = 0.15  # 15% total drawdown = full shutdown

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
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
    if rates is None or len(rates) < 50:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    cols = ["time", "open", "high", "low", "close", "tick_volume"]
    return df[[c for c in cols if c in df.columns]]


def get_h4_bias_at(h4_df, candle_time):
    """Get H4 bias using only bars up to candle_time."""
    if h4_df is None:
        return "NEUTRAL"
    h4_slice = h4_df[h4_df["time"] <= candle_time]
    return strategy.get_h4_bias(h4_slice)


def simulate(m15_df, h4_df):
    m15_df = strategy.calculate_indicators(m15_df)
    trades      = []
    balance     = START_BALANCE
    peak        = START_BALANCE
    open_trade  = None
    trades_by_day = {}
    daily_start_balance = {}
    shutdown    = False

    for i in range(250, len(m15_df) - 1):
        candle      = m15_df.iloc[i]
        next_candle = m15_df.iloc[i + 1]
        day = candle["time"].date()
        trades_by_day.setdefault(day, 0)
        daily_start_balance.setdefault(day, balance)

        if shutdown:
            break

        # Account-level kill switch
        if balance < peak * (1 - ACCOUNT_DD_LIMIT):
            print(f"SHUTDOWN: account drawdown hit {ACCOUNT_DD_LIMIT*100:.0f}% — EA stopped.")
            shutdown = True
            break

        # Manage open trade
        if open_trade is not None:
            open_trade["age"] += 1
            typ    = open_trade["type"]
            price  = candle["close"]
            high   = candle["high"]
            low    = candle["low"]
            risk   = open_trade["risk_amount"]
            entry  = open_trade["entry"]
            sl_dist = open_trade["sl_dist"]

            closed   = False
            result   = None
            pnl      = 0.0
            reason   = ""

            # Phase 1: waiting for partial close at 1.5R
            if not open_trade["partial_done"]:
                partial_price = (entry + sl_dist * PARTIAL_R) if typ == "BUY" else (entry - sl_dist * PARTIAL_R)
                hit_partial = (typ == "BUY" and high >= partial_price) or (typ == "SELL" and low <= partial_price)
                hit_sl      = (typ == "BUY" and low  <= open_trade["sl"]) or (typ == "SELL" and high >= open_trade["sl"])
                timed       = open_trade["age"] >= MAX_HOLD

                if hit_partial and not hit_sl:
                    # Close 50% at 1.5R, move SL to breakeven, start trailing
                    open_trade["partial_done"] = True
                    open_trade["partial_pnl"]  = risk * 0.5 * PARTIAL_R
                    open_trade["sl"]           = entry          # breakeven
                    open_trade["trail_sl"]     = (partial_price - sl_dist * 0.8) if typ == "BUY" else (partial_price + sl_dist * 0.8)
                elif hit_sl or timed:
                    result = "LOSS"
                    pnl    = -risk
                    reason = "SL" if hit_sl else "TIME"
                    closed = True

            # Phase 2: partial done — trailing remaining 50%
            else:
                # Update trail — ratchet in direction of trade
                if typ == "BUY":
                    new_trail = high - sl_dist * 0.8
                    if new_trail > open_trade["trail_sl"]:
                        open_trade["trail_sl"] = new_trail
                    hit_trail = low  <= open_trade["trail_sl"]
                    hit_tp    = high >= open_trade["tp"]
                else:
                    new_trail = low + sl_dist * 0.8
                    if new_trail < open_trade["trail_sl"]:
                        open_trade["trail_sl"] = new_trail
                    hit_trail = high >= open_trade["trail_sl"]
                    hit_tp    = low  <= open_trade["tp"]

                timed = open_trade["age"] >= MAX_HOLD

                if hit_tp:
                    remainder_pnl = risk * 0.5 * TP_R
                    pnl    = open_trade["partial_pnl"] + remainder_pnl
                    result = "WIN"
                    reason = "TP"
                    closed = True
                elif hit_trail or timed:
                    # Remainder closes at trail or time — partial profit locked
                    pnl    = open_trade["partial_pnl"]   # remainder ~0 (near BE or small gain)
                    result = "PARTIAL"
                    reason = "TRAIL" if hit_trail else "TIME"
                    closed = True

            if closed:
                rec = {
                    **open_trade,
                    "exit": price, "exit_time": candle["time"],
                    "result": result, "reason": reason,
                    "pnl": round(pnl, 4),
                    "final_balance": round(balance + pnl, 4),
                }
                trades.append(rec)
                balance = rec["final_balance"]
                peak    = max(peak, balance)
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue

        # Session filter
        if not (SESSION_START <= candle["time"].hour < SESSION_END):
            continue

        # Daily kill switch
        if balance < daily_start_balance[day] * (1 - DAILY_DD_LIMIT):
            continue

        if trades_by_day[day] >= MAX_PER_DAY:
            continue

        # Get H4 bias
        h4_bias = get_h4_bias_at(h4_df, candle["time"])

        # Get signal
        entry_slice = m15_df.iloc[: i + 1]
        result = strategy.analyze_setup(entry_slice, h4_bias=h4_bias)
        sig = result.get("signal", "WAIT")
        if sig == "WAIT":
            continue

        atr = candle["atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        risk_amount = balance * RISK_PCT
        sl_dist     = atr * SL_ATR_MULT
        tp_dist     = sl_dist * TP_R
        entry_price = next_candle["open"]

        if sig == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        open_trade = {
            "type":         sig,
            "entry":        entry_price,
            "sl":           sl,
            "tp":           tp,
            "sl_dist":      sl_dist,
            "risk_amount":  risk_amount,
            "tp_r":         TP_R,
            "entry_time":   next_candle["time"],
            "signal_time":  candle["time"],
            "regime":       result["checks"].get("regime", "?"),
            "conf_type":    result["checks"].get("conf_type", "?"),
            "h4_bias":      h4_bias,
            "balance_before": balance,
            "age":          0,
            "closed":       False,
            "partial_done": False,
            "partial_pnl":  0.0,
            "trail_sl":     None,
        }

    if open_trade is not None:
        f   = m15_df.iloc[-1]
        pnl = -open_trade["risk_amount"]
        trades.append({
            **open_trade,
            "exit": f["close"], "exit_time": f["time"],
            "result": "FORCED", "reason": "FORCED_EXIT",
            "pnl": round(pnl, 4), "final_balance": round(balance + pnl, 4),
        })

    return trades


def monthly_breakdown(all_trades):
    if not all_trades:
        return
    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    df["win"]   = df["result"].isin(["WIN", "PARTIAL"])
    bm = df.groupby("month").agg(
        trades=("result", "count"),
        wins=("win", "sum"),
        net=("pnl", "sum"),
    ).reset_index()
    bm["wr"] = (bm["wins"] / bm["trades"] * 100).round(1)
    print()
    print("Monthly breakdown:")
    print(f"{'Month':<12} | Trades | WR%   | Net")
    print("-" * 45)
    for _, r in bm.iterrows():
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"{str(r['month']):<12} |  {int(r['trades']):3d}   | {r['wr']:5.1f}% | ${r['net']:+7.4f} [{tag}]")

    total_net = df["pnl"].sum()
    total_wr  = df["win"].mean() * 100
    print(f"\nOverall: {len(df)} trades | {int(df['win'].sum())} wins | {total_wr:.1f}% WR | net ${total_net:.4f}")
    print(f"Starting balance ${START_BALANCE:.2f} | Final balance ${START_BALANCE + total_net:.4f}")

    # Breakdown by regime
    if "regime" in df.columns:
        print()
        print("By regime:")
        by_r = df.groupby("regime").agg(
            trades=("result", "count"),
            wins=("win", "sum"),
            net=("pnl", "sum"),
        )
        by_r["wr"] = (by_r["wins"] / by_r["trades"] * 100).round(1)
        print(by_r.to_string())


SCAN_PAIRS = ["US SP 500", "US Tech 100"]


def run_pair(symbol):
    all_trades = []
    for w in range(WINDOWS):
        start_pos = w * BARS
        m15_raw = fetch(symbol, M15_TF, BARS, start_pos)
        h4_raw  = fetch(symbol, H4_TF,  H4_BARS, start_pos // 3)
        if m15_raw is None:
            continue
        trades = simulate(m15_raw, h4_raw)
        all_trades.extend(trades)
    return all_trades


def main():
    if not connect():
        return

    print()
    print(f"Scanning {len(SCAN_PAIRS)} pairs | Session: {SESSION_START:02d}:00-{SESSION_END:02d}:00 GMT | Risk: {RISK_PCT*100:.1f}% | TP: {TP_R}R")
    print(f"Windows: {WINDOWS} x {BARS} bars | Range-only mode")
    print()
    print(f"{'Pair':<12} | Trades | WR%   | Net      | Verdict")
    print("-" * 55)

    results = []
    all_pair_trades = {}
    for symbol in SCAN_PAIRS:
        trades = run_pair(symbol)
        if not trades:
            print(f"{symbol:<16} | {'0':>6} | {'0.0%':>5} | ${'0.00':>7} | NO TRADES")
            results.append({"symbol": symbol, "trades": 0, "wr": 0, "net": 0})
            continue
        wins = sum(1 for t in trades if t["result"] in ["WIN", "PARTIAL"])
        net  = sum(t["pnl"] for t in trades)
        wr   = wins / len(trades) * 100
        tag  = "PROFIT" if net > 0 else "LOSS"
        print(f"{symbol:<16} | {len(trades):>6} | {wr:>5.1f}% | ${net:>+7.2f} | [{tag}]")
        results.append({"symbol": symbol, "trades": len(trades), "wr": wr, "net": net})
        all_pair_trades[symbol] = trades

    mt5.shutdown()

    print()
    print("RANKING (by net profit):")
    results.sort(key=lambda x: x["net"], reverse=True)
    for i, r in enumerate(results, 1):
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"  {i}. {r['symbol']:<16} | {r['trades']:>3} trades | {r['wr']:.1f}% WR | ${r['net']:+.2f} [{tag}]")

    # Monthly breakdown per pair
    for symbol, trades in all_pair_trades.items():
        if not trades:
            continue
        print()
        print(f"--- {symbol} Monthly Breakdown ---")
        df = pd.DataFrame(trades)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["month"] = df["entry_time"].dt.to_period("M")
        df["win"]   = df["result"].isin(["WIN", "PARTIAL"])
        bm = df.groupby("month").agg(trades=("result","count"), wins=("win","sum"), net=("pnl","sum")).reset_index()
        bm["wr"] = (bm["wins"] / bm["trades"] * 100).round(1)
        print(f"{'Month':<12} | Trades | WR%   | Net")
        print("-" * 42)
        for _, r in bm.iterrows():
            tag = "PROFIT" if r["net"] > 0 else "LOSS"
            print(f"{str(r['month']):<12} |  {int(r['trades']):3d}   | {r['wr']:5.1f}% | ${r['net']:+7.4f} [{tag}]")
        total_net = df["pnl"].sum()
        print(f"Overall: {len(df)} trades | {df['win'].mean()*100:.1f}% WR | net ${total_net:.4f}")


def _old_main():
    if not connect():
        return

    print()
    print(f"Symbol: {SYMBOL} | Regime: H4 | Entry: {M15_TF}")
    print(f"Session: {SESSION_START:02d}:00-{SESSION_END:02d}:00 GMT | Risk: {RISK_PCT*100:.1f}% | TP: {TP_R}R")
    print(f"Windows: {WINDOWS} x {BARS} bars")
    print()

    all_trades = []
    window_results = []

    for w in range(WINDOWS):
        start_pos = w * BARS
        m15_raw = fetch(SYMBOL, M15_TF, BARS, start_pos)
        h4_raw  = fetch(SYMBOL, H4_TF,  H4_BARS, start_pos // 3)
        if m15_raw is None:
            print(f"Window {w+1}: no M15 data"); continue

        trades = simulate(m15_raw, h4_raw)
        wins   = sum(1 for t in trades if t["result"] == "WIN")
        net    = sum(t["pnl"] for t in trades)
        wr     = wins / len(trades) * 100 if trades else 0
        tag    = "PROFIT" if net > 0 else "LOSS" if trades else "NO TRADES"
        print(f"Window {w+1} | trades={len(trades):3d} | WR={wr:5.1f}% | net=${net:+7.4f} [{tag}]")
        all_trades.extend(trades)
        window_results.append({"net": net, "trades": len(trades)})

    mt5.shutdown()

    profitable = sum(1 for r in window_results if r["net"] > 0)
    print(f"\nProfitable windows: {profitable}/{len(window_results)}")
    monthly_breakdown(all_trades)


if __name__ == "__main__":
    main()

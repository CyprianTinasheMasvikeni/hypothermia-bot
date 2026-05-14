"""
US Indices Backtester — US SP 500 + US Tech 100
Strategy: regime-adaptive range reversion (strategy_eurusd.py)
Session: 14:00-20:00 GMT (NY session)
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

PAIRS         = ["US SP 500", "US Tech 100"]
M15_TF        = "M15"
H4_TF         = "H4"
BARS          = 15000
H4_BARS       = 2000
WINDOWS       = 5
SESSION_START = 14
SESSION_END   = 20

RISK_PCT      = 0.01
SL_ATR_MULT   = 1.2
TP_R          = 2.0
PARTIAL_R     = 1.5
MAX_HOLD      = 20
MAX_PER_DAY   = 3
DAILY_DD_LIMIT   = 0.03
ACCOUNT_DD_LIMIT = 0.15

TF_MAP = {
    "M15": mt5.TIMEFRAME_M15,
    "H4":  mt5.TIMEFRAME_H4,
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
    if h4_df is None:
        return "NEUTRAL"
    return strategy.get_h4_bias(h4_df[h4_df["time"] <= candle_time])


def simulate(m15_df, h4_df):
    m15_df = strategy.calculate_indicators(m15_df)
    trades, balance, peak = [], START_BALANCE, START_BALANCE
    open_trade, trades_by_day = None, {}
    daily_start_balance, shutdown = {}, False

    for i in range(250, len(m15_df) - 1):
        candle      = m15_df.iloc[i]
        next_candle = m15_df.iloc[i + 1]
        day = candle["time"].date()
        trades_by_day.setdefault(day, 0)
        daily_start_balance.setdefault(day, balance)

        if shutdown:
            break
        if balance < peak * (1 - ACCOUNT_DD_LIMIT):
            shutdown = True
            break

        if open_trade is not None:
            open_trade["age"] += 1
            typ, risk = open_trade["type"], open_trade["risk_amount"]
            entry, sl_dist = open_trade["entry"], open_trade["sl_dist"]
            price, high, low = candle["close"], candle["high"], candle["low"]
            closed, pnl, result, reason = False, 0.0, None, ""

            if not open_trade["partial_done"]:
                partial_price = (entry + sl_dist * PARTIAL_R) if typ == "BUY" else (entry - sl_dist * PARTIAL_R)
                hit_partial = (typ == "BUY" and high >= partial_price) or (typ == "SELL" and low <= partial_price)
                hit_sl      = (typ == "BUY" and low <= open_trade["sl"]) or (typ == "SELL" and high >= open_trade["sl"])
                if hit_partial and not hit_sl:
                    open_trade["partial_done"] = True
                    open_trade["partial_pnl"]  = risk * 0.5 * PARTIAL_R
                    open_trade["sl"]           = entry
                    open_trade["trail_sl"]     = (partial_price - sl_dist * 0.8) if typ == "BUY" else (partial_price + sl_dist * 0.8)
                elif hit_sl or open_trade["age"] >= MAX_HOLD:
                    result, pnl, reason = "LOSS", -risk, "SL" if hit_sl else "TIME"
                    closed = True
            else:
                if typ == "BUY":
                    new_trail = high - sl_dist * 0.8
                    if new_trail > open_trade["trail_sl"]:
                        open_trade["trail_sl"] = new_trail
                    hit_trail = low <= open_trade["trail_sl"]
                    hit_tp    = high >= open_trade["tp"]
                else:
                    new_trail = low + sl_dist * 0.8
                    if new_trail < open_trade["trail_sl"]:
                        open_trade["trail_sl"] = new_trail
                    hit_trail = high >= open_trade["trail_sl"]
                    hit_tp    = low <= open_trade["tp"]

                if hit_tp:
                    pnl, result, reason = open_trade["partial_pnl"] + risk * 0.5 * TP_R, "WIN", "TP"
                    closed = True
                elif hit_trail or open_trade["age"] >= MAX_HOLD:
                    pnl, result, reason = open_trade["partial_pnl"], "PARTIAL", "TRAIL" if hit_trail else "TIME"
                    closed = True

            if closed:
                rec = {**open_trade, "exit": price, "exit_time": candle["time"],
                       "result": result, "reason": reason,
                       "pnl": round(pnl, 4), "final_balance": round(balance + pnl, 4)}
                trades.append(rec)
                balance = rec["final_balance"]
                peak = max(peak, balance)
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue
        if not (SESSION_START <= candle["time"].hour < SESSION_END):
            continue
        if balance < daily_start_balance[day] * (1 - DAILY_DD_LIMIT):
            continue
        if trades_by_day[day] >= MAX_PER_DAY:
            continue

        h4_bias = get_h4_bias_at(h4_df, candle["time"])
        sig = strategy.analyze_setup(m15_df.iloc[:i+1], h4_bias=h4_bias).get("signal", "WAIT")
        if sig == "WAIT":
            continue

        atr = candle["atr"]
        if pd.isna(atr) or atr <= 0:
            continue

        risk_amount = balance * RISK_PCT
        sl_dist     = atr * SL_ATR_MULT
        entry_price = next_candle["open"]
        sl = entry_price - sl_dist if sig == "BUY" else entry_price + sl_dist
        tp = entry_price + sl_dist * TP_R if sig == "BUY" else entry_price - sl_dist * TP_R

        open_trade = {
            "type": sig, "entry": entry_price, "sl": sl, "tp": tp,
            "sl_dist": sl_dist, "risk_amount": risk_amount, "tp_r": TP_R,
            "entry_time": next_candle["time"], "signal_time": candle["time"],
            "balance_before": balance, "age": 0, "closed": False,
            "partial_done": False, "partial_pnl": 0.0, "trail_sl": None,
        }

    if open_trade:
        f = m15_df.iloc[-1]
        pnl = -open_trade["risk_amount"]
        trades.append({**open_trade, "exit": f["close"], "exit_time": f["time"],
                       "result": "FORCED", "reason": "FORCED_EXIT",
                       "pnl": round(pnl, 4), "final_balance": round(balance + pnl, 4)})
    return trades


def run_windows(symbol):
    all_trades = []
    print(f"\n{'='*60}")
    print(f"  {symbol}")
    print(f"{'='*60}")
    print(f"{'Window':<10} | Trades | WR%   | Net      | Result")
    print("-" * 52)
    profitable = 0
    for w in range(WINDOWS):
        start_pos = w * BARS
        m15 = fetch(symbol, M15_TF, BARS, start_pos)
        h4  = fetch(symbol, H4_TF, H4_BARS, start_pos // 3)
        if m15 is None:
            print(f"Window {w+1:<4} | no data")
            continue
        trades = simulate(m15, h4)
        if not trades:
            print(f"Window {w+1:<4} |      0 |   0.0% | $    0.00 | NO TRADES")
            continue
        wins = sum(1 for t in trades if t["result"] in ["WIN", "PARTIAL"])
        net  = sum(t["pnl"] for t in trades)
        wr   = wins / len(trades) * 100
        tag  = "PROFIT" if net > 0 else "LOSS"
        if net > 0:
            profitable += 1
        print(f"Window {w+1:<4} | {len(trades):>6} | {wr:>5.1f}% | ${net:>+7.2f} | [{tag}]")
        all_trades.extend(trades)

    print(f"\nProfitable windows: {profitable}/{WINDOWS}")
    return all_trades


def monthly_breakdown(trades, symbol):
    if not trades:
        print("No trades.")
        return
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    df["win"]   = df["result"].isin(["WIN", "PARTIAL"])
    bm = df.groupby("month").agg(
        trades=("result","count"), wins=("win","sum"), net=("pnl","sum")
    ).reset_index()
    bm["wr"] = (bm["wins"] / bm["trades"] * 100).round(1)

    print(f"\nMonthly breakdown — {symbol}:")
    print(f"{'Month':<12} | Trades | WR%   | Net $100 | Net $1000")
    print("-" * 55)
    for _, r in bm.iterrows():
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"{str(r['month']):<12} |  {int(r['trades']):3d}   | {r['wr']:5.1f}% | ${r['net']:>+7.2f}  | ${r['net']*10:>+8.2f}  [{tag}]")

    total_net = df["pnl"].sum()
    total_wr  = df["win"].mean() * 100
    print(f"\nTotal: {len(df)} trades | {total_wr:.1f}% WR | ${total_net:.2f} on $100 | ${total_net*10:.2f} on $1000")


def combined_summary(all_pair_trades):
    print(f"\n{'='*60}")
    print("  COMBINED (both pairs trading simultaneously)")
    print(f"{'='*60}")
    all_trades = []
    for trades in all_pair_trades.values():
        all_trades.extend(trades)
    if not all_trades:
        print("No trades.")
        return
    df = pd.DataFrame(all_trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["month"] = df["entry_time"].dt.to_period("M")
    df["win"]   = df["result"].isin(["WIN", "PARTIAL"])
    bm = df.groupby("month").agg(
        trades=("result","count"), wins=("win","sum"), net=("pnl","sum")
    ).reset_index()
    bm["wr"] = (bm["wins"] / bm["trades"] * 100).round(1)

    print(f"{'Month':<12} | Trades | WR%   | Net $100 | Net $1000")
    print("-" * 55)
    for _, r in bm.iterrows():
        tag = "PROFIT" if r["net"] > 0 else "LOSS"
        print(f"{str(r['month']):<12} |  {int(r['trades']):3d}   | {r['wr']:5.1f}% | ${r['net']:>+7.2f}  | ${r['net']*10:>+8.2f}  [{tag}]")

    total_net = df["pnl"].sum()
    total_wr  = df["win"].mean() * 100
    profitable_months = (bm["net"] > 0).sum()
    print(f"\nTotal months: {len(bm)} | Profitable: {profitable_months}/{len(bm)}")
    print(f"Total trades: {len(df)} | {total_wr:.1f}% WR")
    print(f"Net on $100:  ${total_net:.2f}")
    print(f"Net on $1000: ${total_net*10:.2f}")
    print(f"Net on $10k:  ${total_net*100:.2f}")


def main():
    if not connect():
        return

    print()
    print(f"Strategy: Range Reversion | Session: {SESSION_START:02d}:00-{SESSION_END:02d}:00 GMT")
    print(f"Risk: {RISK_PCT*100:.1f}% | TP: {TP_R}R | Partial close: {PARTIAL_R}R | Windows: {WINDOWS} x {BARS} bars")

    all_pair_trades = {}
    for symbol in PAIRS:
        trades = run_windows(symbol)
        monthly_breakdown(trades, symbol)
        all_pair_trades[symbol] = trades

    mt5.shutdown()
    combined_summary(all_pair_trades)


if __name__ == "__main__":
    main()

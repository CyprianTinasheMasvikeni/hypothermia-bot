"""
Signal diagnostic — counts how often each condition in analyze_setup()
passes vs fails across the session hours.
Shows exactly what's killing trade frequency.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import strategy_step_trend as strategy

SYMBOL = "Step Index"
BARS   = 15000
SESSION_START = 9; SESSION_END = 19; SKIP_HOURS = {11, 14}

def fetch(tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, start_pos, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time","open","high","low","close","tick_volume"]]

def main():
    mt5.initialize()

    # Use one window
    trend_raw = fetch(mt5.TIMEFRAME_M15, BARS, 0)
    entry_raw  = fetch(mt5.TIMEFRAME_M5,  BARS, 0)
    trend_df   = strategy.calculate_indicators(trend_raw)
    entry_df   = strategy.calculate_indicators(entry_raw)

    def get_bias(t):
        slc = trend_df[trend_df["time"] <= t]
        if len(slc) < 220: return "NEUTRAL"
        return strategy.analyze_setup(slc).get("checks", {}).get("trend_bias", "NEUTRAL")

    counters = {
        "total_session_candles": 0,
        "strong_bias":           0,
        "not_choppy":            0,
        "trend_active":          0,
        "valid_pullback":        0,
        "right_candle_dir":      0,
        "confirmation":          0,
        "momentum":              0,
        "full_signal":           0,
    }

    bias_counts = {}

    for i in range(220, len(entry_df) - 1):
        c    = entry_df.iloc[i]
        hour = c["time"].hour
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS:
            continue

        counters["total_session_candles"] += 1

        bias = get_bias(c["time"])
        bias_counts[bias] = bias_counts.get(bias, 0) + 1

        if bias not in {"STRONG_BUY", "STRONG_SELL"}:
            continue
        counters["strong_bias"] += 1

        es = entry_df.iloc[:i+1].copy()
        chk = strategy.analyze_setup(es).get("checks", {})

        is_buy_bias  = bias == "STRONG_BUY"
        is_sell_bias = bias == "STRONG_SELL"

        not_choppy = not chk.get("market_choppy", True)
        if not_choppy: counters["not_choppy"] += 1

        trend_ok = (is_buy_bias and chk.get("trend_up")) or \
                   (is_sell_bias and chk.get("trend_down"))
        if not_choppy and trend_ok: counters["trend_active"] += 1

        pullback = (is_buy_bias  and chk.get("valid_pullback_buy")) or \
                   (is_sell_bias and chk.get("valid_pullback_sell"))
        if not_choppy and trend_ok and pullback: counters["valid_pullback"] += 1

        right_dir = (is_buy_bias  and chk.get("bullish_candle")) or \
                    (is_sell_bias and chk.get("bearish_candle"))
        if not_choppy and trend_ok and pullback and right_dir:
            counters["right_candle_dir"] += 1

        confirm = (is_buy_bias  and chk.get("bullish_confirmation")) or \
                  (is_sell_bias and chk.get("bearish_confirmation"))
        if not_choppy and trend_ok and pullback and right_dir and confirm:
            counters["confirmation"] += 1

        momentum = (is_buy_bias  and chk.get("momentum_up")) or \
                   (is_sell_bias and chk.get("momentum_down"))
        if not_choppy and trend_ok and pullback and right_dir and confirm and momentum:
            counters["momentum"] += 1
            counters["full_signal"] += 1

    print()
    print("="*60)
    print("  SIGNAL FUNNEL DIAGNOSTIC (1 window, session hours only)")
    print("="*60)
    total = counters["total_session_candles"]
    print(f"\n  Total session candles scanned : {total}")
    print(f"\n  Bias distribution:")
    for b, cnt in sorted(bias_counts.items(), key=lambda x: -x[1]):
        pct = cnt/total*100
        bar = "#" * (cnt * 30 // total)
        print(f"    {b:<15} : {cnt:>5} ({pct:>4.0f}%)  {bar}")

    print(f"\n  Signal funnel (each step requires ALL previous):")
    prev = total
    steps = [
        ("strong_bias",      "STRONG bias on M15"),
        ("not_choppy",       "Market not choppy"),
        ("trend_active",     "M5 trend in same direction"),
        ("valid_pullback",   "Valid pullback to EMA/S&R"),
        ("right_candle_dir", "Candle in right direction"),
        ("confirmation",     "Confirmation pattern (engulf/continuation)"),
        ("momentum",         "RSI momentum confirmed"),
    ]
    for key, label in steps:
        cnt  = counters[key]
        pct  = cnt / total * 100
        drop = (prev - cnt) / prev * 100 if prev > 0 else 0
        flag = " <-- BIGGEST DROP" if drop > 60 else (" <-- big filter" if drop > 35 else "")
        print(f"    {label:<40}: {cnt:>5} ({pct:>4.1f}%)  [-{drop:.0f}%]{flag}")
        prev = cnt

    print(f"\n  Final signals fired : {counters['full_signal']} in this window")
    print(f"  (~{counters['full_signal']/1.72:.1f} per month)")

    print()
    print("="*60)
    print("  WHAT HAPPENS IF WE LOOSEN EACH FILTER?")
    print("="*60)

    # Test: what if we remove momentum requirement?
    no_momentum = counters["confirmation"]
    print(f"\n  Remove RSI momentum check  : {no_momentum} signals  (~{no_momentum/1.72:.1f}/mo)")

    # Test: what if we allow WEAK bias too?
    print(f"  Allow WEAK bias (already tested): same signals -- WEAK never produces entry")

    # Test: what if pullback is dropped?
    no_pullback = counters["right_candle_dir"]
    print(f"  Remove pullback requirement: signals would pass through right_candle + confirm + momentum")

    mt5.shutdown()

if __name__ == "__main__":
    main()

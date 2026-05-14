"""
Strategy Fit Analysis
Answers: WHY do stpRNG and R_25 work while everything else fails?

Measures 5 structural properties that the strategy depends on:
  1. ATR Consistency   -- how predictable is volatility? (low CV = good)
  2. Body Ratio        -- do candles close with conviction? (step index = ~100%)
  3. EMA Stack Clarity -- how often does EMA21>EMA50>EMA200 form and HOLD?
  4. Spike Frequency   -- crash events that break the signal logic
  5. Signal Rate       -- how often all 7 conditions align (too high = low quality)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import strategy_step_trend as strategy

SYMBOLS = [
    "stpRNG", "R_25",          # confirmed winners
    "stpRNG2", "stpRNG3", "stpRNG4", "stpRNG5",  # step siblings
    "R_10", "R_50", "R_75", "R_100",              # volatility siblings
    "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",     # 1Hz siblings
    "RB100", "RB200",                              # range indices
    "JD25", "JD50", "JD75", "JD100",              # jump indices
    "CRASH500", "CRASH900", "CRASH1000",           # crash indices
    "BOOM500", "BOOM1000",                         # boom indices
]


def load_m15(symbol):
    cache = BASE_DIR / "data" / f"cache_{symbol}_M15.csv"
    if not cache.exists():
        return None
    df = pd.read_csv(cache, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def analyze(symbol, df):
    ind = strategy.calculate_indicators(df.copy())
    ind = ind.dropna().reset_index(drop=True)

    if len(ind) < 300:
        return None

    # 1. ATR Consistency: coefficient of variation (lower = more predictable)
    atr_cv = ind["atr"].std() / ind["atr"].mean()

    # 2. Body Ratio: mean body ratio and % of candles with body >= 50%
    body = ind.apply(lambda r: abs(r["close"] - r["open"]) / (r["high"] - r["low"])
                     if r["high"] > r["low"] else 0, axis=1)
    body_mean = body.mean()
    body_pct_50 = (body >= 0.50).mean()

    # 3. EMA Stack: % of candles where EMA21>EMA50>EMA200 OR EMA21<EMA50<EMA200
    bull_stack = (ind["ema_fast"] > ind["ema"]) & (ind["ema"] > ind["ema_long"])
    bear_stack = (ind["ema_fast"] < ind["ema"]) & (ind["ema"] < ind["ema_long"])
    stack_pct = (bull_stack | bear_stack).mean()

    # How stable is the stack: avg consecutive candles in a stack run
    in_stack = (bull_stack | bear_stack).astype(int)
    run_lengths = []
    run = 0
    for v in in_stack:
        if v == 1:
            run += 1
        else:
            if run > 0:
                run_lengths.append(run)
            run = 0
    if run > 0:
        run_lengths.append(run)
    stack_avg_run = np.mean(run_lengths) if run_lengths else 0

    # 4. Spike Frequency: candles where range > 5x ATR (crash/boom events)
    candle_range = ind["high"] - ind["low"]
    spike_pct = (candle_range > ind["atr"] * 5).mean()

    # 5. Signal Rate on M15: how often does STRONG_BUY or STRONG_SELL appear
    # (used as the trend filter in the backtest)
    signal_count = 0
    check_every = 5  # check every 5th candle to speed up
    checked = 0
    for i in range(222, len(ind), check_every):
        bias = strategy.classify_trend_bias(ind.iloc[i])
        if bias in {"STRONG_BUY", "STRONG_SELL"}:
            signal_count += 1
        checked += 1
    signal_rate = signal_count / checked if checked > 0 else 0

    # 6. When STRONG trend fires, does price follow? (directional accuracy)
    # Check if the next 4 candles (1 hour) move in the signalled direction
    correct = 0
    tested = 0
    for i in range(222, len(ind) - 4, check_every * 2):
        bias = strategy.classify_trend_bias(ind.iloc[i])
        if bias == "STRONG_BUY":
            # price should go up in next 4 candles
            future_return = ind.iloc[i+4]["close"] - ind.iloc[i]["close"]
            correct += 1 if future_return > 0 else 0
            tested += 1
        elif bias == "STRONG_SELL":
            future_return = ind.iloc[i]["close"] - ind.iloc[i+4]["close"]
            correct += 1 if future_return > 0 else 0
            tested += 1
    directional_acc = correct / tested if tested > 0 else 0

    return {
        "symbol": symbol,
        "candles": len(ind),
        "atr_cv": atr_cv,
        "body_mean": body_mean,
        "body_pct_50": body_pct_50,
        "stack_pct": stack_pct,
        "stack_avg_run": stack_avg_run,
        "spike_pct": spike_pct,
        "signal_rate": signal_rate,
        "directional_acc": directional_acc,
    }


def main():
    print("=" * 72)
    print("  STRATEGY FIT ANALYSIS")
    print("  Why does the strategy work on stpRNG & R_25 but fail elsewhere?")
    print("=" * 72)
    print()

    results = []
    for sym in SYMBOLS:
        df = load_m15(sym)
        if df is None:
            print(f"  {sym:<12} -- no cache, skipping")
            continue
        r = analyze(sym, df)
        if r is None:
            print(f"  {sym:<12} -- not enough data")
            continue
        tag = " <-- CONFIRMED" if sym in ("stpRNG", "R_25") else ""
        print(f"  {sym:<12} analyzed ({r['candles']:,} candles){tag}")
        results.append(r)

    if not results:
        print("No data found.")
        return

    df_r = pd.DataFrame(results)

    print()
    print("=" * 72)
    print("  METRIC TABLE")
    print()
    print("  Legend:")
    print("  ATR-CV    : Volatility predictability (lower = more consistent)")
    print("  Body%     : Mean candle body ratio (higher = more conviction)")
    print("  Body50%   : % candles closing with 50%+ body (strategy needs this)")
    print("  Stack%    : % time EMA stack fully aligned (all 3 EMAs in order)")
    print("  StackRun  : Avg consecutive candles in an EMA stack run")
    print("  Spike%    : % candles with range > 5xATR (boom/crash events)")
    print("  SigRate   : % M15 candles showing STRONG_BUY/SELL trend bias")
    print("  DirAcc    : When STRONG bias fires, does price follow? (1hr fwd)")
    print()

    hdr = f"  {'Symbol':<12} {'ATR-CV':>8} {'Body%':>7} {'Body50%':>8} {'Stack%':>7} {'StackRun':>9} {'Spike%':>7} {'SigRate':>8} {'DirAcc':>8}"
    print(hdr)
    print("  " + "-" * 70)

    # Sort: confirmed first, then by directional accuracy
    confirmed = df_r[df_r["symbol"].isin(["stpRNG", "R_25"])]
    rest = df_r[~df_r["symbol"].isin(["stpRNG", "R_25"])].sort_values("directional_acc", ascending=False)
    ordered = pd.concat([confirmed, rest], ignore_index=True)

    for _, row in ordered.iterrows():
        tag = " <--" if row["symbol"] in ("stpRNG", "R_25") else ""
        print(
            f"  {row['symbol']:<12} "
            f"{row['atr_cv']:>7.3f}  "
            f"{row['body_mean']:>6.1%}  "
            f"{row['body_pct_50']:>7.1%}  "
            f"{row['stack_pct']:>6.1%}  "
            f"{row['stack_avg_run']:>8.1f}  "
            f"{row['spike_pct']:>6.2%}  "
            f"{row['signal_rate']:>7.1%}  "
            f"{row['directional_acc']:>7.1%}"
            f"{tag}"
        )

    print()
    print("=" * 72)
    print("  DIAGNOSIS")
    print()

    conf = df_r[df_r["symbol"].isin(["stpRNG", "R_25"])]
    fail = df_r[~df_r["symbol"].isin(["stpRNG", "R_25"])]

    print(f"  CONFIRMED SYMBOLS (stpRNG, R_25) averages:")
    print(f"    ATR-CV     : {conf['atr_cv'].mean():.3f}  (lower = more predictable volatility)")
    print(f"    Body 50%+  : {conf['body_pct_50'].mean():.1%}  (candles with conviction)")
    print(f"    Stack %    : {conf['stack_pct'].mean():.1%}  (time in clean EMA alignment)")
    print(f"    Stack Run  : {conf['stack_avg_run'].mean():.1f}   (avg candles held in stack)")
    print(f"    Spike %    : {conf['spike_pct'].mean():.2%}  (disruptive spike events)")
    print(f"    Dir Acc    : {conf['directional_acc'].mean():.1%}  (trend signal follow-through)")
    print()
    print(f"  FAILING SYMBOLS averages:")
    print(f"    ATR-CV     : {fail['atr_cv'].mean():.3f}")
    print(f"    Body 50%+  : {fail['body_pct_50'].mean():.1%}")
    print(f"    Stack %    : {fail['stack_pct'].mean():.1%}")
    print(f"    Stack Run  : {fail['stack_avg_run'].mean():.1f}")
    print(f"    Spike %    : {fail['spike_pct'].mean():.2%}")
    print(f"    Dir Acc    : {fail['directional_acc'].mean():.1%}")
    print()

    # Find any failing symbol that is close to confirmed on all metrics
    print("  CLOSEST TO CONFIRMED (potential candidates):")
    print("  Score = weighted similarity to stpRNG+R_25 profile")
    conf_profile = {
        "atr_cv": conf["atr_cv"].mean(),
        "body_pct_50": conf["body_pct_50"].mean(),
        "stack_pct": conf["stack_pct"].mean(),
        "stack_avg_run": conf["stack_avg_run"].mean(),
        "spike_pct": conf["spike_pct"].mean(),
        "directional_acc": conf["directional_acc"].mean(),
    }

    candidates = []
    for _, row in fail.iterrows():
        # Distance from confirmed profile (normalized)
        score = 0
        score += abs(row["atr_cv"] - conf_profile["atr_cv"]) / (conf_profile["atr_cv"] + 0.001)
        score += abs(row["body_pct_50"] - conf_profile["body_pct_50"]) / (conf_profile["body_pct_50"] + 0.001)
        score += abs(row["stack_pct"] - conf_profile["stack_pct"]) / (conf_profile["stack_pct"] + 0.001)
        score += abs(row["stack_avg_run"] - conf_profile["stack_avg_run"]) / (conf_profile["stack_avg_run"] + 0.001)
        score += abs(row["spike_pct"] - conf_profile["spike_pct"]) / (conf_profile["spike_pct"] + 0.01)
        score += abs(row["directional_acc"] - conf_profile["directional_acc"]) / (conf_profile["directional_acc"] + 0.001)
        candidates.append((row["symbol"], score))

    candidates.sort(key=lambda x: x[1])
    print()
    for sym, sc in candidates[:5]:
        row = df_r[df_r["symbol"] == sym].iloc[0]
        print(f"    {sym:<12}  dist={sc:.3f}  DirAcc={row['directional_acc']:.1%}  "
              f"ATR-CV={row['atr_cv']:.3f}  Spike={row['spike_pct']:.2%}")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()

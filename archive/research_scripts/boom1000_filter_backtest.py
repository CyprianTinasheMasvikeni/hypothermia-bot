"""
BOOM1000 Trend Filter Backtest
================================
Tests whether adding trend filters improves the spike-reversion SELL edge.
BOOM1000 spikes UP — we SELL expecting reversion DOWN.

Filters tested:
  S0   Baseline              — no filter, no cooldown (raw spike reversion)
  S1   M5 EMA8 > EMA21       — SELL only when M5 short-term trend is bullish (current live)
  S2   M5 close > EMA21      — SELL only when price above M5 medium MA
  S3   M5 close > EMA50      — SELL only when price above M5 slow MA
  S4   H1 close > H1 EMA21   — SELL only when H1 is bullish (spike = uptrend overextension)
  S5   H1 close < H1 EMA21   — SELL only when H1 is bearish (spike = dead-cat bounce)
  S6   H1 EMA8 > H1 EMA21    — SELL only when H1 short-term trend bullish
  S7   H1 EMA8 < H1 EMA21    — SELL only when H1 short-term trend bearish
  S8   H1 EMA21 slope rising  — SELL only when H1 EMA21 is accelerating up
  S9   S1 + S4  (M5 bull + H1 bull)
  S10  S1 + S5  (M5 bull + H1 bear)
  S11  S1 + CD12 cooldown     — current filter + 12-candle spike cluster cooldown

Walk-forward: IS = first 6 months, OOS = last 2 months.
"""

import sys, io, math, random
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
random.seed(42); np.random.seed(42)

DATA_DIR    = Path(__file__).resolve().parent / "data"
ATR_PERIOD  = 14
SPIKE_MULT  = 2.5
COOLDOWN    = 12
HOLD        = 24
MAX_DAY     = 6
CHANDELIER  = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R   = 2.0


# ── Data loading ──────────────────────────────────────────────────────────────
def load_m5():
    df = pd.read_csv(DATA_DIR / "cache_BOOM1000_M5.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)

    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(ATR_PERIOD).mean()
    body           = df["close"] - df["open"]   # positive = up candle (boom spike)
    df["is_spike"] = body > SPIKE_MULT * df["atr"]

    df["ema8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    return df.dropna(subset=["atr", "ema50"]).reset_index(drop=True)


def load_h1():
    df = pd.read_csv(DATA_DIR / "cache_BOOM1000_H1.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)

    df["h1_ema8"]        = df["close"].ewm(span=8,  adjust=False).mean()
    df["h1_ema21"]       = df["close"].ewm(span=21, adjust=False).mean()
    df["h1_ema50"]       = df["close"].ewm(span=50, adjust=False).mean()
    df["h1_ema21_slope"] = df["h1_ema21"].diff()
    df["h1_close"]       = df["close"]

    return df.dropna(subset=["h1_ema50"]).reset_index(drop=True)


def merge_h1_into_m5(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Attach last completed H1 bar values to each M5 bar — no lookahead."""
    h1_sorted = h1.sort_values("time").reset_index(drop=True)
    m5_out    = m5.copy()

    cols = ["h1_ema8", "h1_ema21", "h1_ema50", "h1_ema21_slope", "h1_close"]
    for c in cols:
        m5_out[c] = np.nan

    h1_idx       = 0
    last_h1_vals = {c: np.nan for c in cols}

    for i, row in m5_out.iterrows():
        while h1_idx < len(h1_sorted) - 1 and h1_sorted.iloc[h1_idx + 1]["time"] <= row["time"]:
            h1_idx += 1
        if h1_sorted.iloc[h1_idx]["time"] <= row["time"]:
            for c in cols:
                last_h1_vals[c] = h1_sorted.iloc[h1_idx][c]
        for c in cols:
            m5_out.at[i, c] = last_h1_vals[c]

    return m5_out.dropna(subset=["h1_ema50"]).reset_index(drop=True)


# ── Trade simulator — SELL direction ──────────────────────────────────────────
def run_sell(entry: float, atr: float, fwd: pd.DataFrame):
    sl     = entry + atr          # SL above entry (price going up = against us)
    size   = 1.0
    partial_done = False
    locked_r     = 0.0
    trough = entry                # track lowest price seen (trough = our profit direction)

    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_boom = (cl - op) > SPIKE_MULT * row["atr"]  # up spike while we're short = gap risk

        # SL hit — price spiked above our stop
        if hi >= sl:
            exit_r = size * ((entry - cl) / atr) + locked_r if is_boom else size * (-1.0) + locked_r
            return round(exit_r, 3), "SL"

        # Partial close at 2R (price dropped 2R from entry)
        if not partial_done and lo <= entry - atr * PARTIAL_R:
            locked_r     += size * 0.5 * PARTIAL_R
            size         *= 0.5
            partial_done  = True

        # Update trough (lowest point — where profit accumulates for SELL)
        trough   = min(trough, lo)
        trough_r = (entry - trough) / atr

        # Chandelier trailing stop — above trough
        cm  = 3.0
        for mr, tm in CHANDELIER:
            if trough_r >= mr:
                cm = tm
        csl = trough + atr * cm   # stop level above trough
        if hi >= csl:
            return round(size * ((entry - csl) / atr) + locked_r, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    return round(size * ((entry - last) / atr) + locked_r, 3), "TIME"


# ── Single backtest run ───────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, filter_fn=None, use_cooldown=False) -> pd.DataFrame:
    traded          = set()
    trades_per_day: dict = {}
    trades          = []
    lsi             = -999    # last spike index (for cooldown)

    for idx in range(len(df) - HOLD - 1):
        if idx in traded:
            continue
        row = df.iloc[idx]

        if not row["is_spike"]:
            continue

        # 12-candle spike cluster cooldown (optional)
        if use_cooldown:
            if (idx - lsi) <= COOLDOWN:
                lsi = idx
                continue
            lsi = idx
        else:
            lsi = idx

        date  = str(row["time"].date())
        month = date[:7]
        dow   = pd.Timestamp(date).day_name()

        if trades_per_day.get(date, 0) >= MAX_DAY:
            continue

        # Apply trend filter (evaluated at spike bar — no lookahead)
        if filter_fn is not None and not filter_fn(row):
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else float(row["close"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_sell(entry, atr, fwd)
        trades_per_day[date] = trades_per_day.get(date, 0) + 1

        trades.append({
            "date":   date,
            "month":  month,
            "dow":    dow,
            "r":      r,
            "result": "WIN" if r > 0 else "LOSS",
            "reason": reason,
        })
        for k in range(idx, idx + HOLD + 1):
            traded.add(k)

    return pd.DataFrame(trades)


# ── Stats ─────────────────────────────────────────────────────────────────────
def stats(t: pd.DataFrame, label: str):
    if t.empty:
        print(f"  {label:<38} NO TRADES")
        return None

    n    = len(t)
    w    = (t["result"] == "WIN").sum()
    wr   = w / n
    rs   = t["r"].values
    gw   = t[t["r"] > 0]["r"].sum()
    gl   = abs(t[t["r"] < 0]["r"].sum())
    pf   = gw / gl if gl > 0 else 0.0
    avgr = rs.mean()
    stdr = rs.std()

    streak = ms = 0
    for res in t["result"]:
        streak = streak + 1 if res == "LOSS" else 0
        ms = max(ms, streak)

    months   = t.groupby("month")["r"].sum()
    prof_m   = (months > 0).sum()
    t_stat   = (avgr / (stdr / math.sqrt(n))) if stdr > 0 else 0

    return {
        "label":        label,
        "n":            n,
        "wr":           wr,
        "pf":           pf,
        "avg_r":        avgr,
        "std_r":        stdr,
        "t_stat":       t_stat,
        "max_streak":   ms,
        "prof_months":  f"{prof_m}/{len(months)}",
        "trades":       t,
    }


def print_stats(s):
    if s is None:
        return
    print(f"\n  {'='*70}")
    print(f"  {s['label']}")
    print(f"  {'='*70}")
    print(f"  Trades         : {s['n']}")
    print(f"  Win Rate       : {s['wr']*100:.1f}%")
    print(f"  Profit Factor  : {s['pf']:.3f}")
    print(f"  Avg R/trade    : {s['avg_r']:+.3f}R")
    print(f"  Std R          : {s['std_r']:.3f}R")
    print(f"  T-statistic    : {s['t_stat']:.2f}")
    print(f"  Max loss streak: {s['max_streak']}")
    print(f"  Prof months    : {s['prof_months']}")

    t      = s["trades"]
    months = t.groupby("month")["r"].sum()
    print(f"\n  Monthly P&L (R):")
    for m, r in months.items():
        bar  = "#" * int(abs(r) * 2) if abs(r) > 0.1 else ""
        sign = "+" if r >= 0 else "-"
        print(f"    {m}  {sign}{abs(r):6.2f}R  {'['+bar+']' if bar else ''}")

    print(f"\n  Exit breakdown:")
    for reason, cnt in t["reason"].value_counts().items():
        pct = cnt / len(t) * 100
        avg = t[t["reason"] == reason]["r"].mean()
        print(f"    {reason:<12} {cnt:>4} ({pct:4.1f}%)  avg={avg:+.3f}R")


def compare_table(results: list):
    print(f"\n\n{'='*82}")
    print("  SIDE-BY-SIDE COMPARISON  (BOOM1000 · SELL after spike)")
    print(f"{'='*82}")
    print(f"  {'Strategy':<38} {'N':>5} {'WR':>6} {'PF':>6} {'AvgR':>7} {'T':>5} {'MaxLS':>6} {'ProfM':>7}")
    print(f"  {'-'*78}")
    for s in results:
        if s is None:
            continue
        star = " ◀" if s["pf"] == max(x["pf"] for x in results if x) else ""
        print(f"  {s['label']:<38} {s['n']:>5} {s['wr']*100:>5.1f}% {s['pf']:>6.3f} "
              f"{s['avg_r']:>+7.3f} {s['t_stat']:>5.2f} {s['max_streak']:>6} {s['prof_months']:>7}{star}")


def walk_forward(df: pd.DataFrame, filter_fn, label: str, use_cooldown=False):
    dates   = pd.to_datetime(df["time"])
    cutoff  = dates.min() + pd.DateOffset(months=6)
    is_df   = df[dates < cutoff].reset_index(drop=True)
    oos_df  = df[dates >= cutoff].reset_index(drop=True)

    is_t    = backtest(is_df,  filter_fn, use_cooldown)
    oos_t   = backtest(oos_df, filter_fn, use_cooldown)

    is_s    = stats(is_t,  f"{label} IS")
    oos_s   = stats(oos_t, f"{label} OOS")
    return is_s, oos_s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading BOOM1000 data...")
    m5 = load_m5()
    h1 = load_h1()
    print(f"M5 bars: {len(m5):,}  |  H1 bars: {len(h1):,}")
    print("Merging H1 into M5 (no lookahead)...")
    df = merge_h1_into_m5(m5, h1)
    print(f"Merged rows: {len(df):,}  |  Period: {df['time'].min().date()} to {df['time'].max().date()}")

    # ── Spike summary ──────────────────────────────────────────────────────────
    total_spikes = df["is_spike"].sum()
    print(f"\nTotal boom spikes in dataset: {total_spikes:,}  ({total_spikes/len(df)*100:.2f}% of bars)")

    print("\n\nRunning backtests for all filter strategies...")

    filters = [
        # label,                                filter_fn,                                                                    cooldown
        ("S0   Baseline (no filter, no CD)",    None,                                                                         False),
        ("S1   M5 EMA8 > EMA21 [LIVE]",         lambda r: r["ema8"] > r["ema21"],                                             False),
        ("S2   M5 close > EMA21",               lambda r: r["close"] > r["ema21"],                                            False),
        ("S3   M5 close > EMA50",               lambda r: r["close"] > r["ema50"],                                            False),
        ("S4   H1 close > H1 EMA21 (H1 bull)",  lambda r: r["h1_close"] > r["h1_ema21"],                                     False),
        ("S5   H1 close < H1 EMA21 (H1 bear)",  lambda r: r["h1_close"] < r["h1_ema21"],                                     False),
        ("S6   H1 EMA8 > H1 EMA21",             lambda r: r["h1_ema8"] > r["h1_ema21"],                                      False),
        ("S7   H1 EMA8 < H1 EMA21",             lambda r: r["h1_ema8"] < r["h1_ema21"],                                      False),
        ("S8   H1 EMA21 slope rising",          lambda r: r["h1_ema21_slope"] > 0,                                           False),
        ("S9   S1 + S4 (M5 bull + H1 bull)",    lambda r: r["ema8"] > r["ema21"] and r["h1_close"] > r["h1_ema21"],          False),
        ("S10  S1 + S5 (M5 bull + H1 bear)",    lambda r: r["ema8"] > r["ema21"] and r["h1_close"] < r["h1_ema21"],          False),
        ("S11  S1 + CD12 cooldown",              lambda r: r["ema8"] > r["ema21"],                                             True),
        ("S12  S4 + CD12",                       lambda r: r["h1_close"] > r["h1_ema21"],                                     True),
        ("S13  S9 + CD12 (S1+S4+CD12)",         lambda r: r["ema8"] > r["ema21"] and r["h1_close"] > r["h1_ema21"],          True),
    ]

    all_results = []
    for label, fn, cd in filters:
        t = backtest(df, fn, cd)
        s = stats(t, label)
        all_results.append(s)

    # Print full stats for every strategy
    for s in all_results:
        print_stats(s)

    compare_table(all_results)

    # ── Walk-forward validation on top candidates ──────────────────────────────
    print(f"\n\n{'='*82}")
    print("  WALK-FORWARD VALIDATION  (IS = first 6 months | OOS = last 2 months)")
    print(f"{'='*82}")

    wf_candidates = filters[:10]   # all single-filter strategies
    wf_results    = []
    for label, fn, cd in wf_candidates:
        is_s, oos_s = walk_forward(df, fn, label.split()[0], cd)
        wf_results.append((label, is_s, oos_s))

    print(f"\n  {'Strategy':<38} {'IS N':>5} {'IS PF':>7} {'IS WR':>6} {'OOS N':>6} {'OOS PF':>8} {'OOS WR':>7}")
    print(f"  {'-'*80}")
    for label, is_s, oos_s in wf_results:
        if is_s and oos_s:
            print(f"  {label:<38} {is_s['n']:>5} {is_s['pf']:>7.3f} {is_s['wr']*100:>5.1f}%  "
                  f"{oos_s['n']:>5} {oos_s['pf']:>7.3f}  {oos_s['wr']*100:>6.1f}%")

    # ── Best filter deep-dive ──────────────────────────────────────────────────
    valid   = [s for s in all_results if s is not None]
    best    = max(valid, key=lambda s: s["pf"])
    second  = sorted(valid, key=lambda s: s["pf"], reverse=True)[1]

    print(f"\n\n{'='*82}")
    print(f"  WINNER: {best['label']}")
    print(f"  RUNNER-UP: {second['label']}")
    print(f"{'='*82}")
    print(f"\n  Winner vs Baseline comparison:")
    baseline = all_results[0]
    if baseline:
        print(f"    Trades  : {baseline['n']:>5}  →  {best['n']:>5}  "
              f"({(best['n']-baseline['n'])/baseline['n']*100:+.1f}%  trades removed)")
        print(f"    Win Rate: {baseline['wr']*100:>5.1f}%  →  {best['wr']*100:>5.1f}%")
        print(f"    PF      : {baseline['pf']:>5.3f}  →  {best['pf']:>5.3f}")
        print(f"    Avg R   : {baseline['avg_r']:>+6.3f}R  →  {best['avg_r']:>+6.3f}R")
        print(f"    T-stat  : {baseline['t_stat']:>5.2f}  →  {best['t_stat']:>5.2f}")

    # ── Monthly breakdown of winner ────────────────────────────────────────────
    print(f"\n  Monthly detail — {best['label']}:")
    bt = best["trades"]
    monthly = bt.groupby("month").agg(
        n=("r", "count"), total_r=("r", "sum"),
        wr=("result", lambda x: (x == "WIN").mean())
    )
    print(f"  {'Month':<10} {'N':>4} {'Total R':>8} {'WR':>6}")
    print(f"  {'-'*32}")
    for m, row2 in monthly.iterrows():
        bar  = "#" * int(abs(row2.total_r) * 2) if abs(row2.total_r) > 0.1 else ""
        sign = "+" if row2.total_r >= 0 else "-"
        print(f"  {m:<10} {int(row2.n):>4} {sign}{abs(row2.total_r):>6.2f}R  {row2.wr*100:>5.1f}%  "
              f"{'['+bar+']' if bar else ''}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()

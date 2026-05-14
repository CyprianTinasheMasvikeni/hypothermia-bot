"""
CRASH1000 Trend Filter Backtest
================================
Tests whether adding a trend filter improves the spike-reversion edge.

Filters tested:
  S0  Baseline       — no filter (current live strategy)
  S1  M5 EMA8>EMA21  — only BUY when short-term M5 trend is bullish
  S2  M5 close>EMA21 — only BUY when price above M5 medium MA
  S3  M5 close>EMA50 — only BUY when price above M5 slow MA
  S4  H1 EMA8>EMA21  — only BUY when H1 short-term trend is bullish
  S5  H1 close>EMA21 — only BUY when price above H1 medium MA
  S6  H1 close>EMA50 — only BUY when price above H1 slow MA
  S7  H1 EMA21 slope — only BUY when H1 EMA21 is rising (positive slope)
  S8  M5 NOT downtrend — only BUY when M5 is NOT making lower-lows (last 3 bars)

All filters are applied AT the spike candle, using only data available at that time.
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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_m5():
    df = pd.read_csv(DATA_DIR / "cache_CRASH1000_M5.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)

    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]   = tr.rolling(ATR_PERIOD).mean()
    body        = df["open"] - df["close"]   # positive = down candle
    df["is_spike"] = body > SPIKE_MULT * df["atr"]

    df["ema8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Lower-low detector: is this candle's low below the previous 3 lows?
    df["lower_low"] = df["low"] < df["low"].rolling(3).min().shift(1)

    return df.dropna(subset=["atr", "ema50"]).reset_index(drop=True)


def load_h1():
    df = pd.read_csv(DATA_DIR / "cache_CRASH1000_H1.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)

    df["h1_ema8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["h1_ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["h1_ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["h1_ema21_slope"] = df["h1_ema21"].diff()
    df["h1_close"] = df["close"]

    return df.dropna(subset=["h1_ema50"]).reset_index(drop=True)


def merge_h1_into_m5(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Attach completed H1 bar values to each M5 bar (no lookahead)."""
    h1_sorted = h1.sort_values("time").reset_index(drop=True)
    m5_out = m5.copy()

    cols = ["h1_ema8", "h1_ema21", "h1_ema50", "h1_ema21_slope", "h1_close"]
    for c in cols:
        m5_out[c] = np.nan

    h1_idx = 0
    last_h1_vals = {c: np.nan for c in cols}

    for i, row in m5_out.iterrows():
        # Advance H1 pointer to last bar whose open <= m5 bar time
        while h1_idx < len(h1_sorted) - 1 and h1_sorted.iloc[h1_idx + 1]["time"] <= row["time"]:
            h1_idx += 1
        if h1_sorted.iloc[h1_idx]["time"] <= row["time"]:
            for c in cols:
                last_h1_vals[c] = h1_sorted.iloc[h1_idx][c]
        for c in cols:
            m5_out.at[i, c] = last_h1_vals[c]

    return m5_out.dropna(subset=["h1_ema50"]).reset_index(drop=True)


# ── Trade simulator ───────────────────────────────────────────────────────────
def run_buy(entry: float, atr: float, fwd: pd.DataFrame):
    sl = entry - atr
    size = 1.0; partial_done = False; locked_r = 0.0
    peak = entry

    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_crash = (op - cl) > SPIKE_MULT * row["atr"]

        if lo <= sl:
            exit_r = size * ((cl - entry) / atr) + locked_r if is_crash else size * (-1.0) + locked_r
            return round(exit_r, 3), "SL"

        if not partial_done and hi >= entry + atr * 2.0:
            locked_r += size * 0.5 * 2.0
            size *= 0.5
            partial_done = True

        peak = max(peak, hi)
        peak_r = (peak - entry) / atr
        cm = 3.0
        for mr, tm in CHANDELIER:
            if peak_r >= mr:
                cm = tm
        csl = peak - atr * cm
        if lo <= csl:
            return round(size * ((csl - entry) / atr) + locked_r, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    return round(size * ((last - entry) / atr) + locked_r, 3), "TIME"


# ── Single backtest run ───────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, filter_fn=None) -> pd.DataFrame:
    traded = set()
    trades_per_day: dict = {}
    trades = []
    lsi = -999

    for idx in range(len(df) - HOLD - 1):
        if idx in traded:
            continue
        row = df.iloc[idx]

        if not row["is_spike"]:
            continue
        if (idx - lsi) <= COOLDOWN:
            lsi = idx
            continue
        lsi = idx

        date  = str(row["time"].date())
        month = date[:7]
        dow   = pd.Timestamp(date).day_name()

        if trades_per_day.get(date, 0) >= MAX_DAY:
            continue

        # Apply trend filter (evaluated at spike bar, no lookahead)
        if filter_fn is not None and not filter_fn(row):
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else float(row["close"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_buy(entry, atr, fwd)
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
        print(f"  {label:<35} NO TRADES")
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

    months = t.groupby("month")["r"].sum()
    prof_m = (months > 0).sum()

    t_stat = (avgr / (stdr / math.sqrt(n))) if stdr > 0 else 0

    return {
        "label":    label,
        "n":        n,
        "wr":       wr,
        "pf":       pf,
        "avg_r":    avgr,
        "std_r":    stdr,
        "t_stat":   t_stat,
        "max_streak": ms,
        "prof_months": f"{prof_m}/{len(months)}",
        "trades": t,
    }


def print_stats(s):
    if s is None:
        return
    print(f"\n  {'='*68}")
    print(f"  {s['label']}")
    print(f"  {'='*68}")
    print(f"  Trades        : {s['n']}")
    print(f"  Win Rate      : {s['wr']*100:.1f}%")
    print(f"  Profit Factor : {s['pf']:.3f}")
    print(f"  Avg R/trade   : {s['avg_r']:+.3f}R")
    print(f"  Std R         : {s['std_r']:.3f}R")
    print(f"  T-statistic   : {s['t_stat']:.2f}")
    print(f"  Max loss streak: {s['max_streak']}")
    print(f"  Prof months   : {s['prof_months']}")

    t = s["trades"]
    months = t.groupby("month")["r"].sum()
    print(f"\n  Monthly P&L (R):")
    for m, r in months.items():
        bar = "#" * int(abs(r) * 2) if abs(r) > 0.1 else ""
        sign = "+" if r >= 0 else "-"
        print(f"    {m}  {sign}{abs(r):6.2f}R  {'['+bar+']' if bar else ''}")


def compare_table(results: list):
    print(f"\n\n{'='*80}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Strategy':<35} {'N':>5} {'WR':>6} {'PF':>6} {'AvgR':>7} {'T':>5} {'MaxLS':>6} {'ProfM':>7}")
    print(f"  {'-'*75}")
    for s in results:
        if s is None:
            continue
        print(f"  {s['label']:<35} {s['n']:>5} {s['wr']*100:>5.1f}% {s['pf']:>6.3f} "
              f"{s['avg_r']:>+7.3f} {s['t_stat']:>5.2f} {s['max_streak']:>6} {s['prof_months']:>7}")


def walk_forward(df: pd.DataFrame, filter_fn, label: str):
    """Split IS (first 6 months) vs OOS (last 2 months) by date."""
    dates = pd.to_datetime(df["time"])
    cutoff = dates.min() + pd.DateOffset(months=6)
    is_df  = df[dates < cutoff].reset_index(drop=True)
    oos_df = df[dates >= cutoff].reset_index(drop=True)

    is_t  = backtest(is_df,  filter_fn)
    oos_t = backtest(oos_df, filter_fn)

    is_s  = stats(is_t,  f"{label} IS")
    oos_s = stats(oos_t, f"{label} OOS")
    return is_s, oos_s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    m5 = load_m5()
    h1 = load_h1()
    print(f"M5 bars: {len(m5):,}  |  H1 bars: {len(h1):,}")
    print("Merging H1 into M5 (no lookahead)...")
    df = merge_h1_into_m5(m5, h1)
    print(f"Merged rows: {len(df):,}  |  Period: {df['time'].min().date()} to {df['time'].max().date()}")

    print("\n\nRunning backtests for all filter strategies...")

    filters = [
        ("S0  Baseline (no filter)",            None),
        ("S1  M5 EMA8 > EMA21",                 lambda r: r["ema8"] > r["ema21"]),
        ("S2  M5 close > EMA21",                lambda r: r["close"] > r["ema21"]),
        ("S3  M5 close > EMA50",                lambda r: r["close"] > r["ema50"]),
        ("S4  H1 EMA8 > H1 EMA21",              lambda r: r["h1_ema8"] > r["h1_ema21"]),
        ("S5  H1 close > H1 EMA21",             lambda r: r["h1_close"] > r["h1_ema21"]),
        ("S6  H1 close > H1 EMA50",             lambda r: r["h1_close"] > r["h1_ema50"]),
        ("S7  H1 EMA21 slope rising",           lambda r: r["h1_ema21_slope"] > 0),
        ("S8  M5 NOT lower-low bar",            lambda r: not r["lower_low"]),
        ("S4+S1 H1 bull + M5 EMA8>21",         lambda r: r["h1_ema8"] > r["h1_ema21"] and r["ema8"] > r["ema21"]),
        ("S5+S1 H1 close>EMA21 + M5 EMA8>21",  lambda r: r["h1_close"] > r["h1_ema21"] and r["ema8"] > r["ema21"]),
        ("S6+S1 H1 close>EMA50 + M5 EMA8>21",  lambda r: r["h1_close"] > r["h1_ema50"] and r["ema8"] > r["ema21"]),
    ]

    all_results = []
    for label, fn in filters:
        t = backtest(df, fn)
        s = stats(t, label)
        all_results.append(s)

    # Print individual stats for top results
    for s in all_results:
        print_stats(s)

    compare_table(all_results)

    # Walk-forward on best candidates
    print(f"\n\n{'='*80}")
    print("  WALK-FORWARD VALIDATION (IS=6mo | OOS=2mo)")
    print(f"{'='*80}")

    wf_results = []
    for label, fn in filters[:8]:   # top 8
        is_s, oos_s = walk_forward(df, fn, label.split()[0])
        wf_results.append((label, is_s, oos_s))

    print(f"\n  {'Strategy':<35} {'IS PF':>7} {'IS WR':>6} {'OOS PF':>8} {'OOS WR':>7} {'OOS N':>6}")
    print(f"  {'-'*72}")
    for label, is_s, oos_s in wf_results:
        if is_s and oos_s:
            print(f"  {label:<35} {is_s['pf']:>7.3f} {is_s['wr']*100:>5.1f}%  {oos_s['pf']:>7.3f}  {oos_s['wr']*100:>6.1f}% {oos_s['n']:>6}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()

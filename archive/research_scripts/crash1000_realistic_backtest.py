"""
CRASH1000 — Realistic Backtest
================================
Fixes the gap-risk problem missed by the original backtest:

Original:  any SL hit = exactly -1R  (assumes clean execution)
Realistic: if SL triggered by a crash spike → exit at candle CLOSE
           (simulates price gapping through SL in one tick)

Also tests spike cluster filters:
  cooldown=0  → enter immediately after any spike (original)
  cooldown=N  → skip entry if a spike occurred in last N candles
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR         = Path(__file__).resolve().parent
HOLD_CANDLES     = 24
ATR_PERIOD       = 14
THRESHOLD        = 2.5
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
MAX_TRADES_DAY   = 6


def load():
    df = pd.read_csv(BASE_DIR / "data" / "cache_CRASH1000_M5.csv", parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)),
                   abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(ATR_PERIOD).mean()
    body           = df["close"] - df["open"]
    df["is_spike"] = (-body) > THRESHOLD * df["atr"]
    return df.dropna(subset=["atr"]).reset_index(drop=True)


def run_trade(entry, atr, fwd, realistic_gaps):
    """
    Simulate one trade.
    realistic_gaps=True  → crash spike during hold gaps through SL
    realistic_gaps=False → original behaviour (always -1R on SL)
    """
    sl           = entry - atr
    size         = 1.0
    partial_done = False
    locked_r     = 0.0
    cur_size     = size
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi   = row["high"]
        lo   = row["low"]
        op   = row["open"]
        cl   = row["close"]
        body = cl - op

        if lo <= sl:
            if realistic_gaps and (-body) > THRESHOLD * row["atr"]:
                # Crash spike gapped through SL — exit at candle close
                actual_r = cur_size * ((cl - entry) / atr) + locked_r
            else:
                # Normal SL hit — clean -1R execution
                actual_r = cur_size * (-1.0) + locked_r
            return round(actual_r, 3), "SL"

        if not partial_done:
            if hi >= entry + atr * PARTIAL_R:
                locked_r     = cur_size * PARTIAL_PCT * PARTIAL_R
                cur_size    *= (1 - PARTIAL_PCT)
                partial_done = True

        peak_price = max(peak_price, hi)
        peak_r     = (peak_price - entry) / atr if atr > 0 else 0

        cm = CHANDELIER_TIERS[0][1]
        for mr, tm in CHANDELIER_TIERS:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - atr * cm
        if lo <= csl:
            r = cur_size * ((csl - entry) / atr) + locked_r
            return round(r, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    r = cur_size * ((last - entry) / atr) + locked_r
    return round(r, 3), "TIME"


def backtest(df, realistic_gaps, cooldown):
    """
    cooldown = min candles since last spike before we can enter again
    0 = no cooldown (original behaviour)
    """
    traded_idx      = set()
    trades_by_day   = {}
    trades          = []
    last_spike_idx  = -999

    for idx in range(len(df) - HOLD_CANDLES - 1):
        if idx in traded_idx:
            continue

        row  = df.iloc[idx]
        date = str(row["time"].date())

        if not row["is_spike"]:
            continue

        # Spike cluster filter
        if cooldown > 0 and (idx - last_spike_idx) <= cooldown:
            last_spike_idx = idx
            continue

        last_spike_idx = idx

        if trades_by_day.get(date, 0) >= MAX_TRADES_DAY:
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"])
        fwd   = df.iloc[idx + 1 : idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_trade(entry, atr, fwd, realistic_gaps)
        trades_by_day[date] = trades_by_day.get(date, 0) + 1
        trades.append({
            "date":   date,
            "month":  date[:7],
            "r":      r,
            "result": "WIN" if r > 0 else "LOSS",
            "reason": reason,
        })
        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return pd.DataFrame(trades)


def stats(trades, label):
    if trades.empty:
        return None
    total  = len(trades)
    wins   = (trades["result"] == "WIN").sum()
    wr     = wins / total
    gw     = trades[trades["r"] > 0]["r"].sum()
    gl     = abs(trades[trades["r"] < 0]["r"].sum())
    pf     = gw / gl if gl > 0 else 0
    avg_r  = trades["r"].mean()

    streak = max_streak = 0
    for r in trades["result"]:
        if r == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    avg_r_loss = trades[trades["r"] < 0]["r"].mean() if (trades["r"] < 0).any() else 0
    monthly    = trades.groupby("month")["r"].sum()
    prof_months = (monthly > 0).sum()

    return {
        "label":       label,
        "trades":      total,
        "wr":          wr,
        "pf":          pf,
        "avg_r":       avg_r,
        "avg_loss_r":  avg_r_loss,
        "max_streak":  max_streak,
        "prof_months": prof_months,
        "total_months":len(monthly),
        "monthly":     monthly,
        "worst_month": monthly.min(),
        "avg_month":   monthly.mean(),
    }


def main():
    print()
    print("=" * 75)
    print("  CRASH1000 M5 — REALISTIC BACKTEST  (gap risk + cooldown filters)")
    print("=" * 75)

    df = load()
    print(f"\n  Data: {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}"
          f"  ({len(df):,} bars)\n")

    configs = [
        # (realistic_gaps, cooldown, label)
        (False, 0,  "ORIGINAL    (no gap, no cooldown)"),
        (True,  0,  "GAP ONLY    (gap risk, no cooldown)"),
        (True,  3,  "GAP+CD3     (gap risk, 3-candle cooldown)"),
        (True,  6,  "GAP+CD6     (gap risk, 6-candle cooldown)"),
        (True,  12, "GAP+CD12    (gap risk, 12-candle cooldown)"),
        (True,  24, "GAP+CD24    (gap risk, 24-candle cooldown)"),
    ]

    all_stats = []
    for rg, cd, label in configs:
        t = backtest(df, rg, cd)
        s = stats(t, label)
        if s:
            all_stats.append(s)

    # ── Table 1: Core metrics ────────────────────────────────────────────────
    print("CORE METRICS")
    print(f"  {'Config':<38} {'Trades':>7} {'WR':>7} {'PF':>6} {'AvgR':>8} "
          f"{'AvgLoss':>9} {'Streak':>7}")
    print(f"  {'-'*80}")
    for s in all_stats:
        print(f"  {s['label']:<38} {s['trades']:>7} {s['wr']:>6.1%} "
              f"{s['pf']:>6.2f} {s['avg_r']:>+7.3f}R "
              f"{s['avg_loss_r']:>+8.3f}R {s['max_streak']:>7}")

    # ── Table 2: Monthly ────────────────────────────────────────────────────
    print()
    print("MONTHLY PERFORMANCE")
    print(f"  {'Config':<38} {'AvgMonth':>10} {'WorstMonth':>12} {'ProfMonths':>12}")
    print(f"  {'-'*75}")
    for s in all_stats:
        print(f"  {s['label']:<38} {s['avg_month']:>+9.1f}R "
              f"{s['worst_month']:>+11.1f}R "
              f"  {s['prof_months']}/{s['total_months']}")

    # ── Monthly detail comparison ─────────────────────────────────────────────
    orig  = next((s for s in all_stats if "ORIGINAL" in s["label"]), None)
    best  = max((s for s in all_stats if "GAP" in s["label"]),
                key=lambda x: x["pf"], default=None)

    if orig and best:
        print()
        print(f"MONTHLY DETAIL — Original vs Best Realistic ({best['label'].strip()})")
        months = sorted(set(list(orig["monthly"].index) + list(best["monthly"].index)))
        print(f"  {'Month':<9} {'Original':>10} {'Realistic':>11} {'Diff':>8}")
        print(f"  {'-'*42}")
        for m in months:
            r_orig = orig["monthly"].get(m, 0)
            r_best = best["monthly"].get(m, 0)
            diff   = r_best - r_orig
            flag   = " WORSE" if diff < -2 else (" better" if diff > 2 else "")
            print(f"  {m:<9} {r_orig:>+9.1f}R {r_best:>+10.1f}R {diff:>+7.1f}R{flag}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print()
    print("VERDICT")
    print(f"  {'Config':<38} {'Edge?':>10}")
    print(f"  {'-'*52}")
    for s in all_stats:
        if s["pf"] >= 1.5:
            verdict = "STRONG EDGE"
        elif s["pf"] >= 1.2:
            verdict = "WEAK EDGE"
        elif s["pf"] >= 1.0:
            verdict = "BREAKEVEN"
        else:
            verdict = "LOSING"
        print(f"  {s['label']:<38} {verdict}")
    print()


if __name__ == "__main__":
    main()

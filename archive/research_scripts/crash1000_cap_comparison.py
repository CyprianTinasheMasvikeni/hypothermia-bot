"""
CRASH1000 — Trade Cap Comparison
Does the 6-trade/day cap help or hurt?
Runs the same backtest at cap=1,2,3,6,10,unlimited and compares.
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
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df.dropna(subset=["atr"]).reset_index(drop=True)


def run_trade(entry, atr, fwd):
    sl           = entry - atr
    size         = 1.0
    partial_done = False
    locked_r     = 0.0
    cur_size     = size
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]

        if lo <= sl:
            r = cur_size * (-1.0) + locked_r
            return round(r, 3), "SL"

        if not partial_done:
            pp = entry + atr * PARTIAL_R
            if hi >= pp:
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


def backtest(df, max_trades_day):
    body = df["close"] - df["open"]
    df   = df.copy()
    df["is_spike"] = (-body) > THRESHOLD * df["atr"]

    traded_idx    = set()
    trades_by_day = {}
    trades        = []

    for idx in range(len(df) - HOLD_CANDLES - 1):
        if idx in traded_idx:
            continue
        row  = df.iloc[idx]
        date = str(df.iloc[idx]["time"].date())
        if not row["is_spike"]:
            continue
        if max_trades_day is not None and trades_by_day.get(date, 0) >= max_trades_day:
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"])
        fwd   = df.iloc[idx + 1 : idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_trade(entry, atr, fwd)
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


def stats(trades, cap_label):
    if trades.empty:
        return None

    total  = len(trades)
    wins   = (trades["result"] == "WIN").sum()
    wr     = wins / total
    avg_r  = trades["r"].mean()
    gw     = trades[trades["r"] > 0]["r"].sum()
    gl     = abs(trades[trades["r"] < 0]["r"].sum())
    pf     = gw / gl if gl > 0 else 0

    # Daily stats
    daily = trades.groupby("date")["r"].agg(["sum", "count"]).reset_index()
    avg_trades_day = daily["count"].mean()
    max_trades_day = daily["count"].max()
    worst_day_r    = daily["sum"].min()
    best_day_r     = daily["sum"].max()
    avg_day_r      = daily["sum"].mean()
    pct_green_days = (daily["sum"] > 0).mean()

    # Losing streak
    streak = max_streak = 0
    for r in trades["result"]:
        if r == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Monthly
    monthly     = trades.groupby("month")["r"].sum()
    prof_months = (monthly > 0).sum()
    avg_month_r = monthly.mean()
    worst_month = monthly.min()

    return {
        "cap":             cap_label,
        "total_trades":    total,
        "wr":              wr,
        "pf":              pf,
        "avg_r":           avg_r,
        "avg_trades_day":  avg_trades_day,
        "max_trades_day":  max_trades_day,
        "avg_day_r":       avg_day_r,
        "worst_day_r":     worst_day_r,
        "best_day_r":      best_day_r,
        "pct_green_days":  pct_green_days,
        "max_streak":      max_streak,
        "prof_months":     prof_months,
        "total_months":    len(monthly),
        "avg_month_r":     avg_month_r,
        "worst_month_r":   worst_month,
        "monthly":         monthly,
        "daily":           daily,
    }


def main():
    print()
    print("=" * 70)
    print("  CRASH1000 M5 — TRADE CAP COMPARISON  (threshold=2.5x ATR)")
    print("=" * 70)

    df   = load()
    caps = [1, 2, 3, 6, 10, None]
    all_stats = []

    for cap in caps:
        label  = str(cap) if cap is not None else "UNLIMITED"
        trades = backtest(df, cap)
        s      = stats(trades, label)
        if s:
            all_stats.append(s)

    # ── Table 1: Core edge metrics ────────────────────────────────────────────
    print()
    print("CORE EDGE METRICS")
    print(f"  {'Cap':>10} {'Trades':>7} {'WR':>7} {'PF':>6} {'AvgR':>7} {'MaxStreak':>10}")
    print(f"  {'-'*55}")
    for s in all_stats:
        flag = " <-- current" if s["cap"] == "6" else ""
        print(f"  {s['cap']:>10} {s['total_trades']:>7} {s['wr']:>6.1%} "
              f"{s['pf']:>6.2f} {s['avg_r']:>+6.3f}R {s['max_streak']:>10}{flag}")

    # ── Table 2: Daily breakdown ──────────────────────────────────────────────
    print()
    print("DAILY BREAKDOWN (in R — multiply by your risk $ per trade)")
    print(f"  {'Cap':>10} {'Avg/day':>9} {'Max/day':>9} {'Worst day':>10} "
          f"{'Best day':>9} {'Green days':>11}")
    print(f"  {'-'*65}")
    for s in all_stats:
        flag = " <--" if s["cap"] == "6" else ""
        print(f"  {s['cap']:>10} {s['avg_trades_day']:>8.1f} {s['max_trades_day']:>9} "
              f"{s['worst_day_r']:>+9.2f}R {s['best_day_r']:>+8.2f}R "
              f"{s['pct_green_days']:>10.1%}{flag}")

    # ── Table 3: Monthly breakdown ────────────────────────────────────────────
    print()
    print("MONTHLY BREAKDOWN")
    print(f"  {'Cap':>10} {'AvgMonth R':>12} {'WorstMonth R':>14} "
          f"{'ProfitMonths':>14} {'AvgMonth $1risk':>16}")
    print(f"  {'-'*70}")
    for s in all_stats:
        flag = " <--" if s["cap"] == "6" else ""
        print(f"  {s['cap']:>10} {s['avg_month_r']:>+11.1f}R {s['worst_month_r']:>+13.1f}R "
              f"  {s['prof_months']}/{s['total_months']}{' ':>8}"
              f"  ${s['avg_month_r']:>+.2f}{flag}")

    # ── Table 4: Worst day deep dive ──────────────────────────────────────────
    print()
    print("WORST DAY SURVIVAL ($50 account, $1 risk per trade)")
    print(f"  {'Cap':>10} {'Worst day R':>12} {'Worst day $':>13} "
          f"{'% of $50':>10} {'Survive?':>10}")
    print(f"  {'-'*60}")
    for s in all_stats:
        worst_dollar = s["worst_day_r"] * 1.0
        pct_of_50    = abs(worst_dollar) / 50 * 100
        survive      = "YES" if 50 + worst_dollar > 0 else "NO"
        flag = " <--" if s["cap"] == "6" else ""
        print(f"  {s['cap']:>10} {s['worst_day_r']:>+11.2f}R {worst_dollar:>+12.2f}  "
              f"{pct_of_50:>8.1f}%  {survive:>8}{flag}")

    # ── Monthly detail for cap=6 vs unlimited ─────────────────────────────────
    print()
    print("MONTHLY R DETAIL — Cap=6 vs Unlimited")
    cap6  = next((s for s in all_stats if s["cap"] == "6"), None)
    capun = next((s for s in all_stats if s["cap"] == "UNLIMITED"), None)
    if cap6 and capun:
        months = sorted(set(list(cap6["monthly"].index) + list(capun["monthly"].index)))
        print(f"  {'Month':<9} {'Cap=6 R':>10} {'Unlimited R':>12} {'Diff':>8}")
        print(f"  {'-'*45}")
        for m in months:
            r6  = cap6["monthly"].get(m, 0)
            run = capun["monthly"].get(m, 0)
            diff = run - r6
            flag = " WORSE" if diff < 0 else (" better" if diff > 0 else "")
            print(f"  {m:<9} {r6:>+9.1f}R {run:>+11.1f}R {diff:>+7.1f}R{flag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    print("VERDICT")
    cap6  = next((s for s in all_stats if s["cap"] == "6"), None)
    capun = next((s for s in all_stats if s["cap"] == "UNLIMITED"), None)
    if cap6 and capun:
        pf_diff  = capun["pf"] - cap6["pf"]
        wr_diff  = capun["wr"] - cap6["wr"]
        r_diff   = capun["avg_r"] - cap6["avg_r"]
        wd_diff  = capun["worst_day_r"] - cap6["worst_day_r"]
        print(f"  Removing cap changes PF by      : {pf_diff:+.2f}  "
              f"({'better' if pf_diff > 0 else 'WORSE'})")
        print(f"  Removing cap changes WR by      : {wr_diff:>+.1%}  "
              f"({'better' if wr_diff > 0 else 'WORSE'})")
        print(f"  Removing cap changes avg R by   : {r_diff:>+.3f}R  "
              f"({'better' if r_diff > 0 else 'WORSE'})")
        print(f"  Removing cap changes worst day  : {wd_diff:>+.2f}R  "
              f"({'better' if wd_diff > 0 else 'WORSE'})")
        print()
        if capun["pf"] > cap6["pf"] and capun["avg_r"] > cap6["avg_r"]:
            print("  Cap=6 is FILTERING OUT good trades. Removing it makes the edge stronger.")
        elif capun["pf"] < cap6["pf"] or capun["avg_r"] < cap6["avg_r"]:
            print("  Cap=6 is PROTECTING you. Signals after trade 6 are lower quality.")
            print("  Removing the cap adds trades but dilutes the edge.")
        else:
            print("  Cap makes little difference to edge quality.")
        print()


if __name__ == "__main__":
    main()

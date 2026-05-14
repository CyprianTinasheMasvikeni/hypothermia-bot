#!/usr/bin/env python3
"""
Time-of-Day & Day-of-Week Edge Analysis — Deriv Synthetic Indices
==================================================================
Uses the cached 1-year dataset to find exactly which hours and
days are statistically profitable for each symbol.

Run: python time_analysis.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

CACHE_PATH = "quant_cache/raw_data_1yr.pkl"
SYMBOLS    = ["stpRNG", "R_10", "R_25", "R_50", "R_75", "R_100"]
MIN_OBS    = 30   # min candles per bucket to include in analysis

SESSIONS = {
    "Asian":    list(range(0, 7)),
    "London":   list(range(7, 16)),
    "NY":       list(range(12, 21)),
    "Overlap":  list(range(12, 16)),
    "Off-peak": list(range(21, 24)) + list(range(0, 7)),
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["close"] / df["close"].shift(1)).dropna()


def bucket_stats(returns: pd.Series, labels: pd.Series) -> pd.DataFrame:
    """For each unique label compute return statistics."""
    rows = []
    for label in sorted(labels.unique()):
        mask = labels == label
        r    = returns[mask].dropna()
        if len(r) < MIN_OBS:
            continue
        mean  = r.mean()
        std   = r.std()
        n     = len(r)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else 0
        pval  = 2 * stats.t.sf(abs(tstat), df=n - 1)
        wr    = (r > 0).mean()
        rows.append({
            "bucket":  label,
            "n":       n,
            "mean_ret": mean,
            "win_rate": wr,
            "t_stat":  tstat,
            "p_value": pval,
            "sig":     pval < 0.05 and abs(mean) > 0,
        })
    return pd.DataFrame(rows)


def sharpe_like(mean_ret, std_ret, periods_per_year=8760):
    if std_ret == 0:
        return 0
    return (mean_ret / std_ret) * np.sqrt(periods_per_year)


# ── Hour-of-Day Analysis ──────────────────────────────────────────────────────
def analyse_hours(symbol: str, df: pd.DataFrame):
    ret    = log_returns(df)
    hours  = df.index.hour[1:]   # shift to align with returns
    stats_ = bucket_stats(ret, pd.Series(hours, index=ret.index))

    print(f"\n  {symbol} — Hour-of-Day (UTC)  [{len(ret):,} H1 bars]")
    print(f"  {'Hour':>5} {'UTC Session':<12} {'N':>5} {'Mean Ret':>10} {'WinRate':>8} {'t-stat':>7} {'Sig':>4}")
    print(f"  {'-'*5} {'-'*12} {'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*4}")

    def session_tag(h):
        if h in range(12, 16):  return "Overlap"
        if h in range(7,  16):  return "London"
        if h in range(12, 21):  return "NY"
        if h in range(21, 24) or h in range(0, 7): return "Off-peak"
        return "Asian"

    best_hours  = []
    worst_hours = []

    for _, row in stats_.iterrows():
        h    = int(row["bucket"])
        tag  = session_tag(h)
        sig  = "***" if row["p_value"] < 0.01 else ("** " if row["p_value"] < 0.05 else "   ")
        mr   = row["mean_ret"]
        arrow = "+" if mr > 0 else "-"
        print(f"  {h:>5}  {tag:<12} {int(row['n']):>5}  {mr:>+10.6f}  {row['win_rate']:>7.1%}  {row['t_stat']:>7.2f}  {sig}")
        if row["sig"] and mr > 0:
            best_hours.append(h)
        if row["sig"] and mr < 0:
            worst_hours.append(h)

    return best_hours, worst_hours


# ── Session Summary ───────────────────────────────────────────────────────────
def analyse_sessions(symbol: str, df: pd.DataFrame):
    ret  = log_returns(df)
    hour = df.index.hour[1:]

    def get_session(h):
        if h in range(12, 16): return "Overlap(12-16)"
        if h in range(7,  16): return "London(7-16)"
        if h in range(12, 21): return "NY(12-21)"
        if h in range(0,   7): return "Asian(0-7)"
        return "Off-peak(21-24)"

    session_labels = pd.Series(
        [get_session(h) for h in hour], index=ret.index
    )
    stats_ = bucket_stats(ret, session_labels)

    print(f"\n  {symbol} — Session Summary")
    print(f"  {'Session':<18} {'N':>5} {'Mean Ret':>10} {'WinRate':>8} {'t-stat':>7} {'Sig':>4}")
    print(f"  {'-'*18} {'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*4}")
    for _, row in stats_.sort_values("mean_ret", ascending=False).iterrows():
        sig = "***" if row["p_value"] < 0.01 else ("** " if row["p_value"] < 0.05 else "   ")
        print(f"  {str(row['bucket']):<18} {int(row['n']):>5}  {row['mean_ret']:>+10.6f}  {row['win_rate']:>7.1%}  {row['t_stat']:>7.2f}  {sig}")


# ── Day-of-Week Analysis ──────────────────────────────────────────────────────
def analyse_days(symbol: str, df: pd.DataFrame):
    ret  = log_returns(df)
    dow  = df.index.dayofweek[1:]   # 0=Mon, 6=Sun
    stats_ = bucket_stats(ret, pd.Series(dow, index=ret.index))

    print(f"\n  {symbol} — Day of Week")
    print(f"  {'Day':<12} {'N':>5} {'Mean Ret':>10} {'WinRate':>8} {'t-stat':>7} {'Sig':>4}")
    print(f"  {'-'*12} {'-'*5} {'-'*10} {'-'*8} {'-'*7} {'-'*4}")
    for _, row in stats_.iterrows():
        day = DAYS[int(row["bucket"])]
        sig = "***" if row["p_value"] < 0.01 else ("** " if row["p_value"] < 0.05 else "   ")
        print(f"  {day:<12} {int(row['n']):>5}  {row['mean_ret']:>+10.6f}  {row['win_rate']:>7.1%}  {row['t_stat']:>7.2f}  {sig}")


# ── Best Hour × Day Combinations ─────────────────────────────────────────────
def analyse_hour_day(symbol: str, df: pd.DataFrame, best_hours: list):
    if not best_hours:
        return
    ret  = log_returns(df)
    hour = df.index.hour[1:]
    dow  = df.index.dayofweek[1:]

    mask   = pd.Series(hour, index=ret.index).isin(best_hours)
    r_best = ret[mask]
    r_rest = ret[~mask]

    if len(r_best) < MIN_OBS:
        return

    t, p = stats.ttest_ind(r_best, r_rest)
    print(f"\n  {symbol} — Best Hours ({best_hours}) vs Rest")
    print(f"  Best hours: mean={r_best.mean():+.6f} | WR={( r_best>0).mean():.1%} | n={len(r_best)}")
    print(f"  Rest:       mean={r_rest.mean():+.6f} | WR={(r_rest>0).mean():.1%} | n={len(r_rest)}")
    print(f"  t-stat={t:.2f} | p={p:.4f} {'<-- SIGNIFICANT' if p < 0.05 else ''}")


# ── M15 Within-Hour Granularity ───────────────────────────────────────────────
def analyse_m15_hours(symbol: str, df_m15: pd.DataFrame):
    """Use M15 data to see which quarter of the hour is best."""
    ret    = log_returns(df_m15)
    hour   = df_m15.index.hour[1:]
    minute = df_m15.index.minute[1:]
    label  = pd.Series(
        [f"{h:02d}:{m:02d}" for h, m in zip(hour, minute)],
        index=ret.index
    )
    # Aggregate by hour-bucket for cleaner view
    hour_s  = pd.Series(hour, index=ret.index)
    stats_  = bucket_stats(ret, hour_s)
    sig_pos = stats_[stats_["sig"] & (stats_["mean_ret"] > 0)]["bucket"].tolist()
    sig_neg = stats_[stats_["sig"] & (stats_["mean_ret"] < 0)]["bucket"].tolist()

    print(f"\n  {symbol} — M15 Hour-of-Day (top edges)")
    print(f"  Positive edge hours: {[int(h) for h in sig_pos]}")
    print(f"  Negative edge hours: {[int(h) for h in sig_neg]}")

    if sig_pos or sig_neg:
        print(f"  {'Hour':>5} {'N':>6} {'Mean Ret':>10} {'WinRate':>8} {'t-stat':>7}")
        print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*7}")
        notable = stats_[stats_["sig"]].sort_values("mean_ret", ascending=False)
        for _, row in notable.iterrows():
            print(f"  {int(row['bucket']):>5} {int(row['n']):>6}  {row['mean_ret']:>+10.6f}  {row['win_rate']:>7.1%}  {row['t_stat']:>7.2f}")


# ── Cross-Symbol Time Summary ─────────────────────────────────────────────────
def cross_symbol_time_summary(all_best: dict, all_worst: dict):
    print("\n" + "=" * 70)
    print("  CROSS-SYMBOL TIME SUMMARY")
    print("=" * 70)

    # Count how many symbols each hour is "best" vs "worst" for
    hour_score = {}
    for h in range(24):
        best_count  = sum(1 for v in all_best.values()  if h in v)
        worst_count = sum(1 for v in all_worst.values() if h in v)
        hour_score[h] = best_count - worst_count

    print("\n  Hour (UTC) | Score (positive = good across symbols)")
    print(f"  {'-'*10}   {'-'*40}")
    for h in range(24):
        score = hour_score[h]
        bar   = "#" * abs(score) if score != 0 else "."
        sign  = "+" if score > 0 else ("-" if score < 0 else " ")
        tag   = ""
        if h in range(12, 16): tag = " [Overlap]"
        elif h in range(7, 16): tag = " [London]"
        elif h in range(12, 21): tag = " [NY]"
        print(f"  {h:>4}h UTC    {sign}{bar:<10} ({score:+d}){tag}")

    print("\n  Symbol     | Best hours (UTC)       | Worst hours (UTC)")
    print(f"  {'-'*10}   {'-'*22}   {'-'*22}")
    for sym in SYMBOLS:
        best  = sorted(all_best.get(sym,  []))
        worst = sorted(all_worst.get(sym, []))
        print(f"  {sym:<10}   {str(best):<22}   {str(worst):<22}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  TIME-OF-DAY & DAY-OF-WEEK EDGE ANALYSIS")
    print("  Deriv Synthetic Indices — 1 Year of H1 + M15 Data")
    print("=" * 70)

    if not os.path.exists(CACHE_PATH):
        print(f"\n[!] Cache not found at {CACHE_PATH}")
        print("    Run quant_research.py first to pull data.")
        return

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    all_best  = {}
    all_worst = {}

    for symbol in SYMBOLS:
        if symbol not in data:
            continue

        df_h1  = data[symbol].get("H1")
        df_m15 = data[symbol].get("M15")

        if df_h1 is None or len(df_h1) < 100:
            print(f"\n  [!] {symbol}: insufficient H1 data, skipping")
            continue

        print("\n" + "=" * 70)
        print(f"  {symbol}  ({len(df_h1):,} H1 bars | {(df_h1.index[-1]-df_h1.index[0]).days} days)")
        print("=" * 70)

        # Session summary first
        analyse_sessions(symbol, df_h1)

        # Hour-by-hour
        best, worst = analyse_hours(symbol, df_h1)
        all_best[symbol]  = best
        all_worst[symbol] = worst

        # Day of week
        analyse_days(symbol, df_h1)

        # Best hours vs rest
        analyse_hour_day(symbol, df_h1, best)

        # M15 granularity
        if df_m15 is not None and len(df_m15) >= 100:
            analyse_m15_hours(symbol, df_m15)

    # Cross-symbol summary
    cross_symbol_time_summary(all_best, all_worst)

    print("\n\nDone.")


if __name__ == "__main__":
    main()

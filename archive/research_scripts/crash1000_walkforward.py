"""
Walk-Forward Validation — CRASH1000 M5 Spike Reversion
=======================================================
Split:
  In-Sample  (IS) : Sep 2025 – Jan 2026  (~5 months) — optimize threshold
  Out-of-Sample   : Feb 2026 – Apr 2026  (~2.5 months) — blind test

Question: Does the edge survive on data it was NOT optimized on?
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR       = Path(__file__).resolve().parent
OOS_CUTOFF     = "2026-02-01"
HOLD_CANDLES   = 24
MAX_TRADES_DAY = 6
ATR_PERIOD     = 14
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R      = 2.0
PARTIAL_PCT    = 0.50


# -- Data loading & ATR -------------------------------------------------------
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


# -- Single-trade simulator ---------------------------------------------------
def run_trade(entry, atr, fwd):
    """Returns R-multiple."""
    sl           = entry - atr
    size         = 1.0 / atr      # normalised so 1 ATR loss = 1R
    partial_done = False
    locked_pnl   = 0.0
    cur_size     = size
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]
        if lo <= sl:
            pnl = cur_size * (sl - entry) + locked_pnl
            return round(pnl * atr, 3)   # convert back to R

        if not partial_done:
            pp = entry + atr * PARTIAL_R
            if hi >= pp:
                locked_pnl   = cur_size * PARTIAL_PCT * (pp - entry)
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
            pnl = cur_size * (csl - entry) + locked_pnl
            return round(pnl * atr, 3)

    last = fwd.iloc[-1]["close"]
    pnl  = cur_size * (last - entry) + locked_pnl
    return round(pnl * atr, 3)


# -- Backtest on a slice of data ----------------------------------------------
def backtest(df_slice, threshold):
    body = df_slice["close"] - df_slice["open"]
    df_slice = df_slice.copy()
    df_slice["is_spike"] = (-body) > threshold * df_slice["atr"]
    df_slice["body_atr"] = abs(body) / df_slice["atr"]

    traded_idx    = set()
    trades_by_day = {}
    trades        = []

    for idx in range(len(df_slice) - HOLD_CANDLES - 1):
        if idx in traded_idx:
            continue
        row  = df_slice.iloc[idx]
        date = str(df_slice.iloc[idx]["time"].date())
        if not row["is_spike"]:
            continue
        if trades_by_day.get(date, 0) >= MAX_TRADES_DAY:
            continue

        atr   = row["atr"]
        entry = float(df_slice.iloc[idx + 1]["open"])
        fwd   = df_slice.iloc[idx + 1 : idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        r = run_trade(entry, atr, fwd)
        trades_by_day[date] = trades_by_day.get(date, 0) + 1
        trades.append({
            "date":   date,
            "month":  date[:7],
            "r":      r,
            "result": "WIN" if r > 0 else "LOSS",
        })
        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return pd.DataFrame(trades)


# -- Summary stats -------------------------------------------------------------
def summarize(trades, label):
    if trades.empty:
        print(f"  {label}: NO TRADES")
        return None

    total  = len(trades)
    wins   = (trades["result"] == "WIN").sum()
    wr     = wins / total
    avg_r  = trades["r"].mean()
    gw     = trades[trades["r"] > 0]["r"].sum()
    gl     = abs(trades[trades["r"] < 0]["r"].sum())
    pf     = gw / gl if gl > 0 else 0

    streak = max_streak = 0
    for r in trades["result"]:
        if r == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    monthly = trades.groupby("month")["r"].sum()
    prof_months = (monthly > 0).sum()

    days_range = f"{trades['date'].iloc[0]} to {trades['date'].iloc[-1]}"

    print(f"  Period          : {days_range}")
    print(f"  Trades          : {total}  |  Wins: {wins}  Losses: {total-wins}")
    print(f"  Win rate        : {wr:.1%}")
    print(f"  Profit factor   : {pf:.2f}")
    print(f"  Avg R / trade   : {avg_r:+.3f}R")
    print(f"  Max losing streak: {max_streak}")
    print(f"  Profitable months: {prof_months}/{len(monthly)}")
    print()
    print(f"  Monthly R breakdown:")
    for month, r in monthly.items():
        bar = "#" * int(abs(r) / 2) if r > 0 else "-" * int(abs(r) / 2)
        print(f"    {month}  {r:>+7.1f}R  {'[' + bar[:30] + ']' if r > 0 else '[' + bar[:30] + '] <-- RED'}")

    return {"total": total, "wr": wr, "pf": pf, "avg_r": avg_r,
            "max_streak": max_streak, "prof_months": prof_months,
            "total_months": len(monthly)}


# -- Main ---------------------------------------------------------------------
def main():
    print()
    print("=" * 65)
    print("  CRASH1000 M5 — WALK-FORWARD VALIDATION")
    print(f"  IS cutoff: Sep 2025 – Jan 2026 | OOS: Feb 2026 – Apr 2026")
    print("=" * 65)

    df = load()
    cutoff = pd.Timestamp(OOS_CUTOFF, tz="UTC")
    is_df  = df[df["time"] <  cutoff].reset_index(drop=True)
    oos_df = df[df["time"] >= cutoff].reset_index(drop=True)

    print(f"\n  Data split:")
    print(f"    IS  : {is_df['time'].iloc[0].date()} to {is_df['time'].iloc[-1].date()}  ({len(is_df):,} bars)")
    print(f"    OOS : {oos_df['time'].iloc[0].date()} to {oos_df['time'].iloc[-1].date()}  ({len(oos_df):,} bars)")

    # -- Phase 1: threshold sweep on IS data ----------------------------------
    print()
    print("-" * 65)
    print("  PHASE 1: In-Sample Threshold Sweep (find best threshold)")
    print("-" * 65)
    print(f"  {'Thresh':>7} {'Trades':>7} {'WR':>7} {'PF':>6} {'AvgR':>7}")
    print(f"  {'-'*40}")

    best_thresh = None
    best_pf     = 0.0

    for thresh in [1.5, 2.0, 2.5, 3.0, 3.5]:
        t = backtest(is_df, thresh)
        if t.empty or len(t) < 10:
            print(f"  {thresh:>7.1f} {'<10 trades':>40}")
            continue
        wins = (t["result"] == "WIN").sum()
        wr   = wins / len(t)
        gw   = t[t["r"] > 0]["r"].sum()
        gl   = abs(t[t["r"] < 0]["r"].sum())
        pf   = gw / gl if gl > 0 else 0
        ar   = t["r"].mean()
        flag = " <<< BEST" if pf > best_pf else ""
        print(f"  {thresh:>7.1f} {len(t):>7} {wr:>6.1%} {pf:>6.2f} {ar:>+6.3f}R{flag}")
        if pf > best_pf:
            best_pf     = pf
            best_thresh = thresh

    print(f"\n  Best threshold (IS): {best_thresh}x ATR  (PF={best_pf:.2f})")

    # -- Phase 2: Deep IS result at best threshold -----------------------------
    print()
    print("-" * 65)
    print(f"  PHASE 2: In-Sample Full Result  (threshold={best_thresh}x ATR)")
    print("-" * 65)
    is_trades = backtest(is_df, best_thresh)
    is_stats  = summarize(is_trades, "IS")

    # -- Phase 3: OOS blind test at IS-chosen threshold ------------------------
    print()
    print("-" * 65)
    print(f"  PHASE 3: Out-of-Sample BLIND TEST  (threshold={best_thresh}x ATR — FIXED)")
    print("  [Parameters locked from IS. This data was never seen before.]")
    print("-" * 65)
    oos_trades = backtest(oos_df, best_thresh)
    oos_stats  = summarize(oos_trades, "OOS")

    # -- Phase 4: IS vs OOS comparison ----------------------------------------
    print()
    print("-" * 65)
    print("  PHASE 4: IS vs OOS COMPARISON — Does the edge hold?")
    print("-" * 65)
    if is_stats and oos_stats:
        metrics = [
            ("Win rate",         f"{is_stats['wr']:.1%}",        f"{oos_stats['wr']:.1%}"),
            ("Profit factor",    f"{is_stats['pf']:.2f}",        f"{oos_stats['pf']:.2f}"),
            ("Avg R / trade",    f"{is_stats['avg_r']:+.3f}R",   f"{oos_stats['avg_r']:+.3f}R"),
            ("Max streak",       f"{is_stats['max_streak']}",     f"{oos_stats['max_streak']}"),
            ("Profitable months",f"{is_stats['prof_months']}/{is_stats['total_months']}",
                                 f"{oos_stats['prof_months']}/{oos_stats['total_months']}"),
        ]
        print(f"  {'Metric':<22} {'IS (trained)':>14} {'OOS (blind)':>14}  Verdict")
        print(f"  {'-'*60}")
        for name, iv, ov in metrics:
            try:
                iv_f = float(iv.replace("R","").replace("%",""))
                ov_f = float(ov.replace("R","").replace("%",""))
                drop = (iv_f - ov_f) / abs(iv_f) * 100 if iv_f != 0 else 0
                if name == "Win rate" or name == "Profit factor" or name == "Avg R / trade":
                    verdict = "OK" if drop < 25 else ("CAUTION" if drop < 50 else "RISK")
                else:
                    verdict = ""
            except Exception:
                verdict = ""
            print(f"  {name:<22} {iv:>14} {ov:>14}  {verdict}")

        print()
        pf_drop = (is_stats["pf"] - oos_stats["pf"]) / is_stats["pf"] * 100
        if oos_stats["pf"] >= 1.5:
            verdict = "EDGE CONFIRMED — OOS profitable with PF > 1.5"
        elif oos_stats["pf"] >= 1.0:
            verdict = "WEAK EDGE — OOS barely profitable, use caution"
        else:
            verdict = "EDGE FAILED — OOS is a LOSER, do NOT trade live"

        print(f"  PF degradation IS->OOS: {pf_drop:+.1f}%")
        print(f"  VERDICT: {verdict}")
        print()


if __name__ == "__main__":
    main()

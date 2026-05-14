"""
Deep Dive: CRASH1000 + BOOM1000 M5 Reversion Strategy
- No kill switch (see the REAL equity curve)
- Monthly breakdown (is the edge consistent or clustered?)
- Red flag checks
- All thresholds compared
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

START_BAL        = 10_000.0
RISK_PCT         = 0.05
SL_ATR_MULT      = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
HOLD_CANDLES     = 24    # M5 x24 = 2 hours max hold
MAX_TRADES_DAY   = 6     # same cap as stpRNG/R_25 bots
ACCOUNT_DD       = 0.15  # kill switch


def load(symbol, tf):
    p = BASE_DIR / "data" / f"cache_{symbol}_{tf}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def add_atr(df, period=14):
    d = df.copy()
    d["tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(abs(d["high"] - d["close"].shift(1)),
                   abs(d["low"]  - d["close"].shift(1))))
    d["atr"] = d["tr"].rolling(period).mean()
    return d.dropna(subset=["atr"]).reset_index(drop=True)


def detect_spikes(df, pair, threshold):
    d    = df.copy()
    body = d["close"] - d["open"]
    if pair.startswith("BOOM"):
        d["is_spike"] = body > threshold * d["atr"]
        d["trade_dir"] = "SELL"   # reversion = sell after boom spike
    else:
        d["is_spike"] = (-body) > threshold * d["atr"]
        d["trade_dir"] = "BUY"    # reversion = buy after crash spike
    d["body_atr"] = abs(body) / d["atr"]
    return d


def run_trade(entry, atr, dirn, fwd, balance):
    d    = 1 if dirn == "BUY" else -1
    risk = balance * RISK_PCT
    sl   = entry - d * atr * SL_ATR_MULT
    size = risk / (atr * SL_ATR_MULT) if atr > 0 else 0

    partial_done = False
    locked_pnl   = 0.0
    cur_size     = size
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]
        if (d == 1 and lo <= sl) or (d == -1 and hi >= sl):
            pnl = cur_size * d * (sl - entry) + locked_pnl
            return round(pnl, 2), peak_r, "SL"
        if not partial_done:
            pp = entry + d * atr * PARTIAL_R
            if (d == 1 and hi >= pp) or (d == -1 and lo <= pp):
                locked_pnl  = cur_size * PARTIAL_PCT * d * (pp - entry)
                cur_size   *= (1 - PARTIAL_PCT)
                partial_done = True
        peak_price = max(peak_price, hi) if d == 1 else min(peak_price, lo)
        peak_r     = abs(peak_price - entry) / atr if atr > 0 else 0
        cm = CHANDELIER_TIERS[0][1]
        for mr, tm in CHANDELIER_TIERS:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - d * atr * cm
        if (d == 1 and lo <= csl) or (d == -1 and hi >= csl):
            pnl = cur_size * d * (csl - entry) + locked_pnl
            return round(pnl, 2), peak_r, "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    pnl  = cur_size * d * (last - entry) + locked_pnl
    return round(pnl, 2), peak_r, "TIME"


def run_backtest(df, pair, threshold):
    d             = detect_spikes(df, pair, threshold)
    balance       = START_BAL
    peak_bal      = START_BAL
    mdd           = 0.0
    trades        = []
    traded_idx    = set()
    trades_by_day = {}
    fixed_risk    = START_BAL * RISK_PCT  # fixed dollar risk — no compounding inflation

    for idx in range(len(d) - HOLD_CANDLES - 1):
        if idx in traded_idx:
            continue
        row  = d.iloc[idx]
        date = str(d.iloc[idx]["time"].date())
        if not row["is_spike"]:
            continue
        if trades_by_day.get(date, 0) >= MAX_TRADES_DAY:
            continue

        atr   = row["atr"]
        entry = float(d.iloc[idx + 1]["open"])
        fwd   = d.iloc[idx + 1 : idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        dirn = row["trade_dir"]
        # Fixed risk base keeps returns realistic — stake sized to fixed $500 risk
        pnl, peak_r, reason = run_trade(entry, atr, dirn, fwd, fixed_risk / RISK_PCT)
        r_val = pnl / fixed_risk if fixed_risk > 0 else 0

        balance  += pnl
        peak_bal  = max(peak_bal, balance)
        mdd       = max(mdd, (peak_bal - balance) / peak_bal)
        trades_by_day[date] = trades_by_day.get(date, 0) + 1

        t = d.iloc[idx]["time"]
        trades.append({
            "num":      len(trades) + 1,
            "date":     date,
            "month":    date[:7],
            "time":     str(t),
            "dir":      dirn,
            "entry":    round(entry, 2),
            "atr":      round(atr, 2),
            "body_atr": round(row["body_atr"], 2),
            "pnl":      round(pnl, 2),
            "r":        round(r_val, 2),
            "peak_r":   round(peak_r, 2),
            "result":   "WIN" if pnl > 0 else "LOSS",
            "reason":   reason,
            "bal":      round(balance, 2),
        })

        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return trades, balance, mdd


def analyze(pair, tf, threshold, trades, final_bal, max_dd):
    if not trades:
        print("  No trades found.")
        return

    df_t  = pd.DataFrame(trades)
    total = len(df_t)
    wins  = (df_t["result"] == "WIN").sum()
    wr    = wins / total
    gw    = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gl    = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    pf    = gw / gl if gl > 0 else 0
    avg_r = df_t["r"].mean()
    ret   = (final_bal / START_BAL - 1) * 100

    sep = "=" * 68
    print(f"  {sep}")
    print(f"  {pair} {tf} | Spike threshold = {threshold}x ATR | NO KILL SWITCH")
    print(f"  {sep}")
    print(f"  Period       : {df_t['date'].iloc[0]} to {df_t['date'].iloc[-1]}")
    print(f"  Total trades : {total}  |  Wins: {wins}  Losses: {total-wins}")
    print(f"  Win rate     : {wr:.1%}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg R/trade  : {avg_r:+.2f}R")
    print(f"  Net return   : {ret:+.1f}%  (${final_bal - START_BAL:+,.0f})")
    print(f"  Max drawdown : {max_dd:.1%}")
    print()

    # ── Monthly breakdown ──────────────────────────────────────────────────
    print(f"  MONTHLY BREAKDOWN (is the edge consistent every month?):")
    monthly = df_t.groupby("month").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "wins":   (g["result"] == "WIN").sum(),
            "wr":     (g["result"] == "WIN").mean(),
            "pnl":    g["pnl"].sum(),
            "pf":     (g[g["pnl"]>0]["pnl"].sum() /
                       abs(g[g["pnl"]<0]["pnl"].sum())
                       if abs(g[g["pnl"]<0]["pnl"].sum()) > 0 else 0),
        })
    ).reset_index()

    print(f"  {'Month':<9} {'Tr':>3} {'W':>3} {'WR':>6} {'PF':>6}  {'PnL':>9}  Bar")
    print(f"  {'-'*58}")
    for _, row in monthly.iterrows():
        bar_len  = int(row["wr"] * 20)
        bar      = "#" * bar_len + "-" * (20 - bar_len)
        flag     = " W" if row["wr"] >= 0.5 else (" L" if row["wr"] < 0.35 else "  ")
        print(f"  {row['month']:<9} {int(row['trades']):>3} {int(row['wins']):>3} "
              f"{row['wr']:>5.1%} {row['pf']:>6.2f}  ${row['pnl']:>+8.0f}  [{bar}]{flag}")

    prof_months = (monthly["pnl"] > 0).sum()
    total_months = len(monthly)
    print(f"\n  Profitable months: {prof_months}/{total_months} "
          f"({prof_months/total_months:.0%})")
    print()

    # ── Full trade journal ─────────────────────────────────────────────────
    print(f"  FULL TRADE LIST:")
    print(f"  {'#':<4} {'Date':<12} {'Dir':<5} {'Entry':>8} {'ATR':>7} "
          f"{'BodyATR':>8} {'R':>6} {'PkR':>6} {'Result':<7} {'Reason':<12} {'Balance'}")
    print(f"  {'-'*82}")
    for _, row in df_t.iterrows():
        big = " <-- BIG" if row["peak_r"] >= 5 else ""
        print(f"  {int(row['num']):<4} {row['date']:<12} {row['dir']:<5} "
              f"{row['entry']:>8.2f} {row['atr']:>7.2f} "
              f"{row['body_atr']:>8.2f} {row['r']:>+5.2f}R {row['peak_r']:>5.2f}R  "
              f"{'WIN' if row['pnl']>0 else 'LOSS':<7} {row['reason']:<12} "
              f"${row['bal']:>10,.2f}{big}")
    print()

    # ── Red flag checks ────────────────────────────────────────────────────
    print(f"  RED FLAG CHECKS:")

    # 1. Top 3 trades dominating profit?
    total_pnl = df_t["pnl"].sum()
    top3_pnl  = df_t.nlargest(3, "pnl")["pnl"].sum()
    top3_pct  = top3_pnl / total_pnl * 100 if total_pnl > 0 else 0
    flag1 = "RISK" if top3_pct > 70 else "OK"
    print(f"  Top 3 trades = {top3_pct:.0f}% of total profit  [{flag1}]")

    # 2. Longest losing streak
    streak = max_streak = 0
    for r in df_t["result"]:
        if r == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    flag2 = "RISK" if max_streak >= 6 else "OK"
    print(f"  Longest losing streak : {max_streak} trades  [{flag2}]")

    # 3. Sample size check
    flag3 = "RISK" if total < 20 else ("CAUTION" if total < 40 else "OK")
    print(f"  Sample size           : {total} trades  [{flag3}]")

    # 4. Profitable months consistency
    flag4 = "OK" if prof_months >= total_months * 0.55 else "RISK"
    print(f"  Profitable months     : {prof_months}/{total_months}  [{flag4}]")

    # 5. Average spike size (are we only catching genuine spikes?)
    avg_body = df_t["body_atr"].mean()
    print(f"  Avg spike size        : {avg_body:.2f}x ATR")
    print()


def threshold_comparison(pair, tf, df):
    print(f"  THRESHOLD SWEEP — {pair} {tf} (no kill switch)")
    print(f"  {'Thresh':>7} {'Trades':>7} {'WR':>7} {'PF':>6} "
          f"{'AvgR':>6} {'Ret%':>7} {'MaxDD':>7}")
    print(f"  {'-'*55}")
    for thresh in [1.5, 2.0, 2.5, 3.0, 3.5]:
        trades, final_bal, mdd = run_backtest(df, pair, thresh)
        if not trades or len(trades) < 5:
            print(f"  {thresh:>7.1f} {'<5 trades':>50}")
            continue
        df_t = pd.DataFrame(trades)
        total = len(df_t)
        wins  = (df_t["result"] == "WIN").sum()
        wr    = wins / total
        gw    = df_t[df_t["pnl"] > 0]["pnl"].sum()
        gl    = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
        pf    = gw / gl if gl > 0 else 0
        avg_r = df_t["r"].mean()
        ret   = (final_bal / START_BAL - 1) * 100
        flag  = " <<< STRONG" if pf >= 1.5 and total >= 20 else \
                " << DECENT"  if pf >= 1.3 and total >= 10 else \
                " (losing)"   if pf < 1.0 else ""
        print(f"  {thresh:>7.1f} {total:>7} {wr:>6.1%} {pf:>6.2f} "
              f"{avg_r:>+5.2f}R {ret:>+6.1f}% {mdd*100:>6.1f}%{flag}")
    print()


def main():
    print()
    print("=" * 70)
    print("  DEEP DIVE: CRASH1000 + BOOM1000 M5 Reversion")
    print("  NO KILL SWITCH — real equity curve, no inflation")
    print("=" * 70)
    print()

    for pair in ["CRASH1000", "BOOM1000"]:
        df = load(pair, "M5")
        if df is None:
            print(f"  {pair} M5: no data")
            continue
        df = add_atr(df)
        print(f"\n{'='*70}")
        print(f"  {pair}")
        print(f"{'='*70}")
        print(f"  Data: {len(df):,} M5 bars | "
              f"{df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}")
        print()

        # Threshold sweep first
        threshold_comparison(pair, "M5", df)

        # Deep dive on best thresholds
        for thresh in [2.0, 2.5, 3.0]:
            trades, final_bal, mdd = run_backtest(df, pair, thresh)
            print()
            analyze(pair, "M5", thresh, trades, final_bal, mdd)
            print()


if __name__ == "__main__":
    main()

"""
Deep dive: AUD/USD London Breakout + USD/JPY Previous Day Break
- Full trade journal
- Year-by-year breakdown (does edge hold each year?)
- Monthly win rate (is it clustered or consistent?)
- Drawdown map
- Verdict
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "forex_cache"
sys.path.insert(0, str(BASE_DIR))

MULTIPLIER       = 100
SL_ATR_MULT      = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
RISK_PCT         = 0.05
ACCOUNT_DD       = 0.15
START_BAL        = 10_000.0


def load(pair):
    cache = CACHE_DIR / f"forex_{pair}_H1.csv"
    if not cache.exists():
        return None
    df = pd.read_csv(cache, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[df["time"].dt.weekday < 5].sort_values("time").reset_index(drop=True)


def add_indicators(df):
    d = df.copy()
    d["tr"] = np.maximum(d["high"] - d["low"],
               np.maximum(abs(d["high"] - d["close"].shift(1)),
                          abs(d["low"]  - d["close"].shift(1))))
    d["atr"] = d["tr"].rolling(14).mean()
    d["date"] = d["time"].dt.date
    daily = d.groupby("date").agg(
        d_high=("high","max"), d_low=("low","min"),
        d_open=("open","first"), d_close=("close","last")
    ).reset_index()
    daily["prev_high"]  = daily["d_high"].shift(1)
    daily["prev_low"]   = daily["d_low"].shift(1)
    daily["prev_close"] = daily["d_close"].shift(1)
    d = d.merge(daily[["date","prev_high","prev_low","prev_close"]], on="date", how="left")
    return d.dropna(subset=["atr"]).reset_index(drop=True)


def run_trade(entry, atr, dirn, fwd):
    d     = 1 if dirn == "BUY" else -1
    risk  = START_BAL * RISK_PCT  # fixed for per-trade R calc
    sf    = MULTIPLIER * SL_ATR_MULT * atr / entry
    stake = max(1.0, round(risk / sf, 2)) if sf > 0 else 1.0
    sl    = entry - d * atr * SL_ATR_MULT

    partial_done = False
    locked_pnl   = 0.0
    cur_stake    = stake
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]

        if (d == 1 and lo <= sl) or (d == -1 and hi >= sl):
            pnl = cur_stake * MULTIPLIER * d * (sl - entry) / entry + locked_pnl
            return pnl, peak_r, "SL"

        if not partial_done:
            pp = entry + d * atr * PARTIAL_R
            if (d == 1 and hi >= pp) or (d == -1 and lo <= pp):
                locked_pnl   = cur_stake * PARTIAL_PCT * MULTIPLIER * d * (pp - entry) / entry
                cur_stake   *= (1 - PARTIAL_PCT)
                partial_done = True

        peak_price = max(peak_price, hi) if d == 1 else min(peak_price, lo)
        peak_r     = abs(peak_price - entry) / atr if atr > 0 else 0

        cm  = CHANDELIER_TIERS[0][1]
        for mr, tm in CHANDELIER_TIERS:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - d * atr * cm
        if (d == 1 and lo <= csl) or (d == -1 and hi >= csl):
            pnl = cur_stake * MULTIPLIER * d * (csl - entry) / entry + locked_pnl
            return pnl, peak_r, "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    pnl  = cur_stake * MULTIPLIER * d * (last - entry) / entry + locked_pnl
    return pnl, peak_r, "TIME"


# ── AUDUSD: London Breakout ───────────────────────────────────────────────────
def get_audusd_trades(df):
    trades = []
    traded = set()
    for i in range(100, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < 7 or hour >= 10:
            continue
        if date in traded:
            continue
        atr = r["atr"]
        if atr <= 0 or pd.isna(atr):
            continue

        t           = r["time"]
        asian_start = t.normalize() - pd.Timedelta(hours=2)
        asian_end   = t.normalize() + pd.Timedelta(hours=7)
        asian       = df[(df["time"] >= asian_start) & (df["time"] < asian_end)]
        if len(asian) < 5:
            continue

        a_high = asian["high"].max()
        a_low  = asian["low"].min()
        rng    = a_high - a_low
        if rng < atr * 0.3 or rng > atr * 3.0:
            continue

        close = float(r["close"])
        dirn  = None
        if close > a_high:
            dirn = "BUY"
        elif close < a_low:
            dirn = "SELL"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 33].copy()
        if len(fwd) < 4:
            continue

        risk  = START_BAL * RISK_PCT
        pnl, peak_r, reason = run_trade(entry, atr, dirn, fwd)
        r_val = pnl / risk if risk > 0 else 0

        trades.append({
            "date": str(date), "dir": dirn, "entry": round(entry, 5),
            "atr": round(atr, 5), "pnl": round(pnl, 2),
            "r": round(r_val, 2), "peak_r": round(peak_r, 2),
            "result": "WIN" if pnl > 0 else "LOSS", "reason": reason,
            "month": str(date)[:7], "year": str(date)[:4],
        })
        traded.add(date)
    return trades


# ── USDJPY: Previous Day High/Low Break ──────────────────────────────────────
def get_usdjpy_trades(df):
    trades = []
    traded = set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < 7 or hour >= 12:
            continue
        if date in traded:
            continue
        atr    = r["atr"]
        prev_h = r["prev_high"]
        prev_l = r["prev_low"]
        if pd.isna(prev_h) or pd.isna(prev_l) or atr <= 0:
            continue

        close = r["close"]
        dirn  = None
        if close > prev_h and r["close"] > r["open"]:
            dirn = "BUY"
        elif close < prev_l and r["close"] < r["open"]:
            dirn = "SELL"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 25].copy()
        if len(fwd) < 4:
            continue

        risk = START_BAL * RISK_PCT
        pnl, peak_r, reason = run_trade(entry, atr, dirn, fwd)
        r_val = pnl / risk if risk > 0 else 0

        trades.append({
            "date": str(date), "dir": dirn, "entry": round(entry, 3),
            "atr": round(atr, 3), "pnl": round(pnl, 2),
            "r": round(r_val, 2), "peak_r": round(peak_r, 2),
            "result": "WIN" if pnl > 0 else "LOSS", "reason": reason,
            "month": str(date)[:7], "year": str(date)[:4],
        })
        traded.add(date)
    return trades


def analyze(pair, strategy_name, trades):
    if not trades:
        print("  No trades.")
        return

    df_t  = pd.DataFrame(trades)
    total = len(df_t)
    wins  = (df_t["result"] == "WIN").sum()
    wr    = wins / total
    avg_r = df_t["r"].mean()
    gross_w = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gross_l = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    pf    = gross_w / gross_l if gross_l > 0 else 0

    # Balance curve
    bal  = START_BAL
    peak = START_BAL
    mdd  = 0.0
    for pnl in df_t["pnl"]:
        bal  += pnl
        peak  = max(peak, bal)
        mdd   = max(mdd, (peak - bal) / peak)
    ret = (bal / START_BAL - 1) * 100

    print(f"  {'='*68}")
    print(f"  {pair} -- {strategy_name}")
    print(f"  {'='*68}")
    print(f"  Total trades : {total}  |  Wins: {wins}  Losses: {total-wins}")
    print(f"  Win rate     : {wr:.1%}")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg R/trade  : {avg_r:+.2f}R")
    print(f"  Net return   : {ret:+.1f}%  (${bal-START_BAL:+,.0f})")
    print(f"  Max drawdown : {mdd:.1%}")
    print()

    # Year by year
    print(f"  YEAR-BY-YEAR BREAKDOWN:")
    print(f"  {'Year':<6} {'Trades':>7} {'WR':>7} {'PF':>7} {'Ret%':>8}")
    print(f"  {'-'*38}")
    for yr, grp in df_t.groupby("year"):
        y_wins = (grp["result"] == "WIN").sum()
        y_wr   = y_wins / len(grp)
        y_gw   = grp[grp["pnl"] > 0]["pnl"].sum()
        y_gl   = abs(grp[grp["pnl"] < 0]["pnl"].sum())
        y_pf   = y_gw / y_gl if y_gl > 0 else 0
        y_ret  = grp["pnl"].sum() / START_BAL * 100
        flag   = " <-- CONSISTENT" if y_pf >= 1.20 and y_wr >= 0.45 else (" <-- WEAK" if y_pf < 1.0 else "")
        print(f"  {yr:<6} {len(grp):>7} {y_wr:>6.1%} {y_pf:>7.2f} {y_ret:>+7.1f}%{flag}")
    print()

    # Monthly win rate heatmap (text)
    print(f"  MONTHLY WIN RATE (is edge consistent or clustered?):")
    monthly = df_t.groupby("month").apply(
        lambda g: pd.Series({
            "trades": len(g),
            "wr": (g["result"] == "WIN").mean(),
            "pnl": g["pnl"].sum()
        })
    ).reset_index()
    print(f"  {'Month':<9} {'Tr':>4} {'WR':>7} {'PnL':>10}  Bar")
    print(f"  {'-'*50}")
    for _, row in monthly.iterrows():
        bar_len  = int(row["wr"] * 20)
        bar_fill = "#" * bar_len + "-" * (20 - bar_len)
        flag = " W" if row["wr"] >= 0.5 else (" L" if row["wr"] < 0.35 else "  ")
        print(f"  {row['month']:<9} {int(row['trades']):>4} {row['wr']:>6.1%} "
              f"  ${row['pnl']:>+7.0f}  [{bar_fill}]{flag}")
    print()

    # Full trade journal
    print(f"  FULL TRADE LIST:")
    print(f"  {'#':<4} {'Date':<12} {'Dir':<5} {'Entry':>9} {'ATR':>7} "
          f"{'R':>6} {'PeakR':>7} {'Result':<7} {'Reason'}")
    print(f"  {'-'*72}")
    bal = START_BAL
    for idx, row in df_t.iterrows():
        bal += row["pnl"]
        flag = " <--BIG" if row["peak_r"] >= 5 else ""
        print(f"  {idx+1:<4} {row['date']:<12} {row['dir']:<5} "
              f"{row['entry']:>9.4f} {row['atr']:>7.4f} "
              f"{row['r']:>+5.2f}R {row['peak_r']:>6.2f}R  "
              f"{'WIN' if row['pnl']>0 else 'LOSS':<7} {row['reason']}{flag}")
    print()

    # Red flag check
    print(f"  RED FLAG CHECKS:")
    # 1. Is the edge from just a few big trades?
    top3_pnl  = df_t.nlargest(3, "pnl")["pnl"].sum()
    total_pnl = df_t["pnl"].sum()
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
    print(f"  Longest losing streak: {max_streak} trades  [{flag2}]")

    # 3. Does WR vary wildly by direction?
    for dirn in ["BUY", "SELL"]:
        grp = df_t[df_t["dir"] == dirn]
        if len(grp) >= 5:
            dwr = (grp["result"] == "WIN").mean()
            print(f"  {dirn} trades: {len(grp)} trades, {dwr:.1%} WR")

    # 4. Is the sample consistent (not just one lucky period)?
    monthly_pf = []
    for _, grp in df_t.groupby("month"):
        if len(grp) >= 3:
            gw = grp[grp["pnl"] > 0]["pnl"].sum()
            gl = abs(grp[grp["pnl"] < 0]["pnl"].sum())
            if gl > 0:
                monthly_pf.append(gw / gl)
    profitable_months = sum(1 for x in monthly_pf if x >= 1.0)
    flag3 = "OK" if profitable_months >= len(monthly_pf) * 0.55 else "RISK"
    print(f"  Profitable months (PF>=1): {profitable_months}/{len(monthly_pf)}  [{flag3}]")
    print()


def main():
    print()
    print("=" * 70)
    print("  DEEP DIVE: AUD/USD London Breakout + USD/JPY Prev Day Break")
    print("  Verifying edge is real, consistent, and not curve-fitted")
    print("=" * 70)
    print()

    # AUD/USD
    df_aud = load("AUDUSD")
    if df_aud is not None:
        df_aud = add_indicators(df_aud)
        aud_trades = get_audusd_trades(df_aud)
        analyze("AUD/USD", "London Breakout (07:00-10:00 UTC)", aud_trades)
        pd.DataFrame(aud_trades).to_csv(
            BASE_DIR / "quant_cache" / "audusd_trades.csv", index=False)

    # USD/JPY
    df_jpy = load("USDJPY")
    if df_jpy is not None:
        df_jpy = add_indicators(df_jpy)
        jpy_trades = get_usdjpy_trades(df_jpy)
        analyze("USD/JPY", "Previous Day High/Low Break (07:00-12:00 UTC)", jpy_trades)
        pd.DataFrame(jpy_trades).to_csv(
            BASE_DIR / "quant_cache" / "usdjpy_trades.csv", index=False)


if __name__ == "__main__":
    main()

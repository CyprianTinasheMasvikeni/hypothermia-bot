"""
CRASH1000 Practical Analysis — $50 starting account
Questions to answer:
  1. Trades per day (average, best day, worst day distribution)
  2. PnL per day (average, worst day, best day)
  3. PnL per month
  4. Losing streak survival at $50
  5. What risk % actually keeps us alive?
  6. What minimum account do we actually need?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

HOLD_CANDLES   = 24
MAX_TRADES_DAY = 6
THRESHOLD      = 2.5    # best threshold from deep dive


def load(symbol, tf):
    p = BASE_DIR / "data" / f"cache_{symbol}_{tf}.csv"
    df = pd.read_csv(p, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def add_atr(df):
    d = df.copy()
    d["tr"] = np.maximum(
        d["high"] - d["low"],
        np.maximum(abs(d["high"] - d["close"].shift(1)),
                   abs(d["low"]  - d["close"].shift(1))))
    d["atr"] = d["tr"].rolling(14).mean()
    return d.dropna(subset=["atr"]).reset_index(drop=True)


def detect_spikes(df, threshold):
    d    = df.copy()
    body = d["close"] - d["open"]
    d["is_spike"] = (-body) > threshold * d["atr"]   # crash spike = big DOWN candle
    d["body_atr"] = abs(body) / d["atr"]
    return d


def run_trade(entry, atr, fwd, risk_dollars):
    """BUY reversion trade after crash spike. Returns R-multiple."""
    sl   = entry - atr          # SL = 1 ATR below entry
    size = risk_dollars / atr   # units (so 1 ATR loss = risk_dollars)

    CHANDELIER = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
    PARTIAL_R  = 2.0
    PARTIAL_PCT = 0.50

    partial_done = False
    locked_pnl   = 0.0
    cur_size     = size
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]

        if lo <= sl:
            pnl = cur_size * (sl - entry) + locked_pnl
            return round(pnl / risk_dollars, 3), "SL"

        if not partial_done:
            pp = entry + atr * PARTIAL_R
            if hi >= pp:
                locked_pnl   = cur_size * PARTIAL_PCT * (pp - entry)
                cur_size    *= (1 - PARTIAL_PCT)
                partial_done = True

        peak_price = max(peak_price, hi)
        peak_r     = (peak_price - entry) / atr if atr > 0 else 0

        cm = CHANDELIER[0][1]
        for mr, tm in CHANDELIER:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - atr * cm
        if lo <= csl:
            pnl = cur_size * (csl - entry) + locked_pnl
            return round(pnl / risk_dollars, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    pnl  = cur_size * (last - entry) + locked_pnl
    return round(pnl / risk_dollars, 3), "TIME"


def collect_trades(df, threshold):
    d             = detect_spikes(df, threshold)
    traded_idx    = set()
    trades_by_day = {}
    all_trades    = []

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

        r_mult, reason = run_trade(entry, atr, fwd, risk_dollars=1.0)  # R per $1 risk

        trades_by_day[date] = trades_by_day.get(date, 0) + 1
        all_trades.append({
            "date":    date,
            "month":   date[:7],
            "r_mult":  r_mult,
            "result":  "WIN" if r_mult > 0 else "LOSS",
            "reason":  reason,
            "atr":     round(atr, 2),
        })

        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return pd.DataFrame(all_trades)


def main():
    print()
    print("=" * 65)
    print("  CRASH1000 M5 — PRACTICAL ANALYSIS FOR $50 ACCOUNT")
    print("  Threshold=2.5x ATR | Max 6 trades/day | BUY after spike")
    print("=" * 65)

    df = load("CRASH1000", "M5")
    df = add_atr(df)
    trades = collect_trades(df, THRESHOLD)

    total_days   = trades["date"].nunique()
    total_months = trades["month"].nunique()
    total_trades = len(trades)
    wins         = (trades["result"] == "WIN").sum()
    wr           = wins / total_trades
    avg_r        = trades["r_mult"].mean()
    gw = trades[trades["r_mult"] > 0]["r_mult"].sum()
    gl = abs(trades[trades["r_mult"] < 0]["r_mult"].sum())
    pf = gw / gl if gl > 0 else 0

    # ── Section 1: Overall edge ───────────────────────────────────────────────
    print()
    print("1. EDGE SUMMARY (R-based, independent of account size)")
    print(f"   Period      : {trades['date'].iloc[0]} to {trades['date'].iloc[-1]}")
    print(f"   Total trades: {total_trades} over {total_days} trading days ({total_months} months)")
    print(f"   Win rate    : {wr:.1%}")
    print(f"   Profit factor: {pf:.2f}")
    print(f"   Avg R/trade : {avg_r:+.3f}R  (per $1 risked, expect ${avg_r:.3f} return)")

    # ── Section 2: Daily breakdown ────────────────────────────────────────────
    daily = trades.groupby("date").agg(
        trades     = ("r_mult", "count"),
        total_r    = ("r_mult", "sum"),
        wins       = ("result", lambda x: (x == "WIN").sum()),
        losses     = ("result", lambda x: (x == "LOSS").sum()),
    ).reset_index()

    print()
    print("2. DAILY TRADE DISTRIBUTION")
    for n in range(1, 8):
        cnt = (daily["trades"] == n).sum()
        pct = cnt / len(daily) * 100
        bar = "#" * int(pct / 2)
        print(f"   {n} trades/day : {cnt:>3} days ({pct:>4.1f}%)  {bar}")

    print()
    print("3. DAILY PnL DISTRIBUTION (in R units, multiply by your risk $)")
    print(f"   Average day  : {daily['total_r'].mean():+.2f}R")
    print(f"   Best day     : {daily['total_r'].max():+.2f}R")
    print(f"   Worst day    : {daily['total_r'].min():+.2f}R")
    print(f"   % days green : {(daily['total_r'] > 0).mean():.1%}")
    print(f"   % days red   : {(daily['total_r'] < 0).mean():.1%}")

    worst_days = daily.nsmallest(5, "total_r")[["date","trades","wins","losses","total_r"]]
    print()
    print("   5 worst days:")
    for _, row in worst_days.iterrows():
        print(f"   {row['date']}: {int(row['trades'])} trades, "
              f"{int(row['wins'])}W/{int(row['losses'])}L = {row['total_r']:+.2f}R")

    # ── Section 3: Monthly breakdown ─────────────────────────────────────────
    monthly = trades.groupby("month").agg(
        trades  = ("r_mult", "count"),
        total_r = ("r_mult", "sum"),
        wins    = ("result", lambda x: (x == "WIN").sum()),
    ).reset_index()
    monthly["wr"] = monthly["wins"] / monthly["trades"]

    print()
    print("4. MONTHLY BREAKDOWN (R per month)")
    print(f"   {'Month':<9} {'Trades':>7} {'Wins':>5} {'WR':>6} {'TotalR':>8}")
    print("   " + "-" * 42)
    for _, row in monthly.iterrows():
        print(f"   {row['month']:<9} {int(row['trades']):>7} "
              f"{int(row['wins']):>5} {row['wr']:>5.1%} {row['total_r']:>+7.1f}R")
    print(f"   {'AVERAGE':<9} {monthly['trades'].mean():>7.0f} "
          f"{'':>5} {monthly['wr'].mean():>5.1%} {monthly['total_r'].mean():>+7.1f}R")

    # ── Section 4: Losing streak analysis ────────────────────────────────────
    streak = 0
    max_streak = 0
    streaks = []
    for r in trades["result"]:
        if r == "LOSS":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0
    if streak > 0:
        streaks.append(streak)

    streak_counts = pd.Series(streaks).value_counts().sort_index()
    print()
    print("5. LOSING STREAK ANALYSIS")
    print(f"   Longest streak ever   : {max_streak} losses in a row")
    print(f"   Streaks of 5+  : {sum(1 for s in streaks if s >= 5)} times")
    print(f"   Streaks of 8+  : {sum(1 for s in streaks if s >= 8)} times")
    print(f"   Streaks of 10+ : {sum(1 for s in streaks if s >= 10)} times")
    print(f"   Streaks of 12+ : {sum(1 for s in streaks if s >= 12)} times")

    # ── Section 5: $50 account survival ──────────────────────────────────────
    print()
    print("6. $50 ACCOUNT — RISK % IMPACT")
    print(f"   {'Risk%':>6} {'$/trade':>8} {'Max streak loss':>16} {'Survive?':>10} "
          f"{'Avg day $':>10} {'Avg month $':>12}")
    print("   " + "-" * 68)

    for risk_pct in [0.01, 0.02, 0.03, 0.05]:
        risk_dollar    = 50 * risk_pct
        streak_loss    = max_streak * risk_dollar
        surviving_bal  = 50 - streak_loss
        survive        = "YES" if surviving_bal > 0 else "NO"
        survive_pct    = surviving_bal / 50 * 100
        avg_day_dollar = daily["total_r"].mean() * risk_dollar
        avg_mon_dollar = monthly["total_r"].mean() * risk_dollar
        print(f"   {risk_pct:>5.0%} {risk_dollar:>8.2f}  "
              f"  -{streak_loss:>7.2f} ({max_streak} losses)  "
              f"{survive:>6} ({surviving_bal:>+.1f})  "
              f"{avg_day_dollar:>+8.2f}   {avg_mon_dollar:>+9.2f}")

    print()
    print("7. MINIMUM ACCOUNT SIZE TO SURVIVE THE WORST STREAK")
    for risk_pct in [0.01, 0.02, 0.03, 0.05]:
        min_account = max_streak * (1 / risk_pct)
        print(f"   At {risk_pct:.0%} risk: need at least ${min_account:.0f} "
              f"to survive {max_streak}-loss streak")

    # ── Section 6: Realistic $50 growth projection ────────────────────────────
    print()
    print("8. REALISTIC GROWTH PROJECTION AT $50 (2% risk, compounding)")
    bal    = 50.0
    peak   = 50.0
    mdd    = 0.0
    month_start = bal

    print(f"   {'Month':<9} {'Trades':>7} {'WinR':>6} {'MonthR':>8} "
          f"{'Profit$':>9} {'Balance':>10} {'DD%':>6}")
    print("   " + "-" * 60)

    for _, row in monthly.iterrows():
        risk_dollar = bal * 0.02
        month_pnl   = row["total_r"] * risk_dollar
        bal        += month_pnl
        peak        = max(peak, bal)
        dd          = (peak - bal) / peak * 100
        mdd         = max(mdd, dd)
        print(f"   {row['month']:<9} {int(row['trades']):>7} "
              f"{row['wr']:>5.1%} {row['total_r']:>+7.1f}R "
              f"{month_pnl:>+8.2f}  ${bal:>8.2f}  {dd:>5.1f}%")

    print(f"\n   Final balance : ${bal:,.2f}  (started $50.00)")
    print(f"   Total return  : {(bal/50-1)*100:+.0f}%")
    print(f"   Max drawdown  : {mdd:.1f}%")

    print()
    print("9. KEY CONCERNS FOR LIVE TRADING AT $50")
    print(f"   a) Worst losing streak is {max_streak} trades.")
    print(f"      At 2% risk ($1/trade): costs ${max_streak*1.0:.0f} = {max_streak*2:.0f}% of $50 account")
    print(f"      At 5% risk ($2.5/trade): costs ${max_streak*2.5:.0f} = {max_streak*5:.0f}% of $50 account")
    print(f"   b) Worst single day: {daily['total_r'].min():+.2f}R")
    print(f"      At 2% risk: ${daily['total_r'].min()*1.0:+.2f} = {daily['total_r'].min()*2:+.0f}% of account")
    print(f"   c) Can Deriv place a CRASH1000 trade with only $1 risk? Need to verify min stake.")
    print(f"   d) The 2% risk recommendation gives avg ${monthly['total_r'].mean()*1.0:+.2f}/month profit")
    print(f"      which is {monthly['total_r'].mean()*2:+.0f}% monthly return on $50 — still very strong.")
    print()


if __name__ == "__main__":
    main()

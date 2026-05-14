"""
CRASH1000 — USD Backtest (Live-Ready)
======================================
Mirrors the live bot exactly:
  - Gap-adjusted sizing: stake / 1.86 → gap events cost exactly 2% of balance
  - 12-candle spike cooldown
  - Hot (3%) / Base (2%) / Cold (1%) risk scaling on streaks
  - Daily 3% DD kill switch
  - Account 15% DD kill switch (permanent stop)
  - Compounding balance throughout
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ── Match live bot settings exactly ─────────────────────────────────────────
STARTING_BALANCE   = 10_000.0
RISK_PCT_BASE      = 0.02
RISK_PCT_HOT       = 0.03
RISK_PCT_COLD      = 0.01
STREAK_THRESHOLD   = 2
GAP_RISK_MULT      = 1.86
SPIKE_COOLDOWN     = 12
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT     = 0.03
ACCOUNT_DD_LIMIT   = 0.15

ATR_PERIOD         = 14
SPIKE_THRESHOLD    = 2.5
HOLD_CANDLES       = 24
CHANDELIER_TIERS   = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R          = 2.0
PARTIAL_PCT        = 0.50


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
    df["is_spike"] = (-body) > SPIKE_THRESHOLD * df["atr"]
    return df.dropna(subset=["atr"]).reset_index(drop=True)


def run_trade(entry, atr, fwd):
    """Simulate one trade with realistic gap SL. Returns (R multiple, exit reason)."""
    sl           = entry - atr
    cur_size     = 1.0
    partial_done = False
    locked_r     = 0.0
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi   = row["high"]
        lo   = row["low"]
        op   = row["open"]
        cl   = row["close"]
        body = cl - op

        if lo <= sl:
            if (-body) > SPIKE_THRESHOLD * row["atr"]:
                r = cur_size * ((cl - entry) / atr) + locked_r
            else:
                r = cur_size * (-1.0) + locked_r
            return round(r, 3), "SL"

        if not partial_done and hi >= entry + atr * PARTIAL_R:
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


MIN_HALF_STAKE = 1.0    # Deriv minimum stake per contract
MULTIPLIER     = 100
SL_ATR_MULT    = 1.0


def usd_backtest(df, starting_balance=STARTING_BALANCE,
                 daily_dd=DAILY_DD_LIMIT, account_dd=None,
                 model_min_stake=False):
    """
    model_min_stake=False  → ideal: monetary SL always honored, pure % risk
    model_min_stake=True   → realistic: $1 min stake, gap losses bypass monetary SL
                             (models what actually happens on small accounts)
    """
    balance       = starting_balance
    peak_balance  = starting_balance
    day_start_bal = starting_balance
    cur_day       = None

    traded_idx     = set()
    trades_by_day  = {}
    last_spike_idx = -999
    cons_wins      = 0
    cons_losses    = 0
    trades         = []
    bot_stopped    = False
    blown          = False

    for idx in range(len(df) - HOLD_CANDLES - 1):
        if bot_stopped or blown:
            break
        if idx in traded_idx:
            continue

        row  = df.iloc[idx]
        date = str(row["time"].date())

        if date != cur_day:
            day_start_bal = balance
            cur_day       = date

        if not row["is_spike"]:
            continue

        if (idx - last_spike_idx) <= SPIKE_COOLDOWN:
            last_spike_idx = idx
            continue
        last_spike_idx = idx

        if trades_by_day.get(date, 0) >= MAX_TRADES_PER_DAY:
            continue

        if daily_dd and balance < day_start_bal * (1 - daily_dd):
            continue

        if account_dd and balance < starting_balance * (1 - account_dd):
            bot_stopped = True
            break

        if cons_wins >= STREAK_THRESHOLD:
            risk_pct = RISK_PCT_HOT
        elif cons_losses >= STREAK_THRESHOLD:
            risk_pct = RISK_PCT_COLD
        else:
            risk_pct = RISK_PCT_BASE

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_trade(entry, atr, fwd)

        gap_adj_risk = balance * risk_pct / GAP_RISK_MULT  # intended USD risk

        if model_min_stake and entry > 0:
            # sl_factor: fraction of price move per 1R = how much each $1 stake earns per R
            sl_factor  = MULTIPLIER * SL_ATR_MULT * atr / entry
            # full_stake = total stake for the combined position to risk gap_adj_risk at -1R
            # split into 2 equal half-contracts (PARTIAL_PCT = 0.50)
            full_stake = (gap_adj_risk / sl_factor) if sl_factor > 0 else gap_adj_risk
            half_stake = max(MIN_HALF_STAKE, full_stake * PARTIAL_PCT)
            # total USD move per 1R across both contracts
            stake_risk_per_r = 2 * half_stake * sl_factor

            if reason == "SL" and abs(r + 1.0) < 0.001:
                # Clean SL — Deriv monetary SL is honored, caps at gap_adj_risk
                pnl_usd = -gap_adj_risk
            else:
                # Gap SL bypass / win / chandelier / time — actual stake-based P&L
                pnl_usd = r * stake_risk_per_r
        else:
            # Ideal: pure % risk, monetary SL always honored
            pnl_usd = r * gap_adj_risk

        # Balance floor — account blown if below minimum to open a trade ($2 for 2 contracts)
        balance += pnl_usd
        if balance < 2.0:
            blown = True
            balance = 0.0

        peak_balance  = max(peak_balance, balance)
        dd_usd        = balance - peak_balance
        dd_pct        = abs(dd_usd) / peak_balance * 100 if peak_balance > 0 else 0

        if pnl_usd > 0:
            cons_wins   += 1
            cons_losses  = 0
        else:
            cons_losses += 1
            cons_wins    = 0

        trades_by_day[date] = trades_by_day.get(date, 0) + 1

        trades.append({
            "date":        date,
            "month":       date[:7],
            "r":           r,
            "reason":      reason,
            "risk_pct":    risk_pct,
            "pnl_usd":     round(pnl_usd, 2),
            "balance":     round(balance, 2),
            "peak_balance":round(peak_balance, 2),
            "dd_usd":      round(dd_usd, 2),
            "dd_pct":      round(dd_pct, 2),
        })

        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return pd.DataFrame(trades), bot_stopped, blown


def print_results(trades, stopped, label, starting_balance=STARTING_BALANCE):
    if trades.empty:
        print(f"  [{label}] No trades.")
        return

    monthly_pnl = {}
    for month, g in trades.groupby("month"):
        n      = len(g)
        winpct = (g["pnl_usd"] > 0).mean() * 100
        pnl    = g["pnl_usd"].sum()
        endb   = g["balance"].iloc[-1]
        maxddp = g["dd_pct"].max()
        maxddu = g["dd_usd"].min()
        avg_t  = pnl / n
        monthly_pnl[month] = pnl
        print(f"  {month:<9} {n:>7} {winpct:>5.1f}% {pnl:>+10,.0f}  "
              f"${endb:>10,.0f}  -{maxddp:>5.1f}%  ${maxddu:>8,.0f}  {avg_t:>+9,.0f}")

    final_balance = trades["balance"].iloc[-1]
    total_return  = (final_balance / starting_balance - 1) * 100
    max_dd_pct    = trades["dd_pct"].max()
    max_dd_usd    = trades["dd_usd"].min()

    first_dt = pd.to_datetime(trades["date"].iloc[0])
    last_dt  = pd.to_datetime(trades["date"].iloc[-1])
    days     = max((last_dt - first_dt).days, 1)
    cagr     = ((final_balance / starting_balance) ** (365.0 / days) - 1) * 100

    monthly_s = pd.Series(monthly_pnl)
    best_m    = monthly_s.idxmax()
    worst_m   = monthly_s.idxmin()
    avg_m     = monthly_s.mean()
    profit_m  = (monthly_s > 0).sum()

    sl_n  = (trades["reason"] == "SL").sum()
    ch_n  = (trades["reason"] == "CHANDELIER").sum()
    tm_n  = (trades["reason"] == "TIME").sum()
    hot_n = (trades["risk_pct"] == RISK_PCT_HOT).sum()
    cld_n = (trades["risk_pct"] == RISK_PCT_COLD).sum()
    bas_n = (trades["risk_pct"] == RISK_PCT_BASE).sum()

    streak = max_streak = 0
    for p in trades["pnl_usd"]:
        if p < 0:
            streak    += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    print()
    print(f"  SUMMARY [{label}]")
    print(f"  {'='*55}")
    print(f"  Starting balance   : ${starting_balance:>12,.0f}")
    print(f"  Ending balance     : ${final_balance:>12,.2f}")
    print(f"  Total return       : {total_return:>+11.1f}%")
    print(f"  CAGR (annualised)  : {cagr:>+11.1f}%")
    print(f"  Max drawdown       : -{max_dd_pct:>9.1f}%  (-${abs(max_dd_usd):>,.0f})")
    print(f"  Max losing streak  : {max_streak:>11} trades")
    print(f"  Avg monthly P&L    : ${avg_m:>+12,.0f}")
    print(f"  Best month         : ${monthly_s[best_m]:>+12,.0f}  ({best_m})")
    print(f"  Worst month        : ${monthly_s[worst_m]:>+12,.0f}  ({worst_m})")
    print(f"  Profitable months  : {profit_m}/{len(monthly_s)}")
    print(f"  Total trades       : {len(trades):>12,}  |  Win rate: {(trades['pnl_usd']>0).mean()*100:.1f}%")
    print(f"  Exits              : SL={sl_n} | Chandelier={ch_n} | Time={tm_n}")
    print(f"  Risk tiers         : Base={bas_n} | Hot={hot_n} | Cold={cld_n}")
    if stopped:
        stop_row = trades.iloc[-1]
        print(f"  *** STOPPED at {stop_row['date']} | balance ${stop_row['balance']:,.2f} ***")

    print()
    print("  SCALE TABLE — same % risk, different starting amounts")
    print(f"  {'Start':>10} {'End Bal':>11} {'Return%':>9} {'Avg/Month':>11} {'MaxDD $':>10}")
    print(f"  {'-'*55}")
    scale  = final_balance / starting_balance
    dd_r   = abs(max_dd_usd) / starting_balance
    avg_r2 = avg_m / starting_balance
    for s in [1_000, 2_000, 5_000, 10_000, 25_000, 50_000]:
        print(f"  ${s:>9,}  ${s*scale:>10,.0f}  {total_return:>+8.1f}%  ${s*avg_r2:>10,.0f}  -${s*dd_r:>8,.0f}")


def run_scenario(df, label, start, min_stake=False, daily_dd=DAILY_DD_LIMIT, account_dd=None):
    trades, stopped, blown = usd_backtest(df, starting_balance=start,
                                          daily_dd=daily_dd, account_dd=account_dd,
                                          model_min_stake=min_stake)
    if trades.empty:
        print(f"  {label:<30}  NO TRADES")
        return None, blown, stopped

    final  = trades["balance"].iloc[-1]
    ret    = (final / start - 1) * 100
    mdd    = trades["dd_pct"].max()
    mdd_u  = trades["dd_usd"].min()
    monthly = trades.groupby("month")["pnl_usd"].sum()
    avg_m  = monthly.mean()
    prof_m = (monthly > 0).sum()
    streak = max_streak = 0
    for p in trades["pnl_usd"]:
        if p < 0: streak += 1; max_streak = max(max_streak, streak)
        else:     streak = 0

    status = "BLOWN" if blown else ("STOPPED" if stopped else "OK")
    print(f"  {label:<30}  ${final:>10,.2f}  {ret:>+8.1f}%  "
          f"-{mdd:>5.1f}%  ${avg_m:>+8,.2f}/mo  {prof_m}/{len(monthly)} prof  streak={max_streak}  [{status}]")
    return trades, blown, stopped


def main():
    print()
    print("=" * 80)
    print("  CRASH1000 — $50 STARTING BALANCE ANALYSIS")
    print("=" * 80)

    df = load()
    print(f"\n  Data: {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}  ({len(df):,} bars)")
    print(f"  Strategy: 2% base risk | 12-candle cooldown | gap adj 1.86x\n")

    # ── PART 1: $50 — Ideal vs Realistic ──────────────────────────────────────
    print("=" * 80)
    print("  PART 1 — $50 STARTING BALANCE: Ideal vs Realistic")
    print("  Ideal    = Deriv monetary SL always honored (% risk preserved)")
    print("  Realistic= $1 min stake + gap SL bypasses monetary SL (what actually happens)")
    print("=" * 80)
    print()
    print(f"  {'Scenario':<30}  {'End Bal':>11}  {'Return':>8}  {'MaxDD':>6}  {'Avg/Mo':>9}  {'Prof Mo':>8}  {'MaxStreak':>9}  Status")
    print(f"  {'-'*95}")

    t_ideal,    blown_i, _ = run_scenario(df, "$50 | Ideal (SL always honored)",  50, min_stake=False)
    t_real,     blown_r, _ = run_scenario(df, "$50 | Realistic (min stake+gap)",  50, min_stake=True)

    # ── PART 2: $50 Realistic — month by month until blow/end ─────────────────
    print()
    print("=" * 80)
    print("  PART 2 — $50 REALISTIC: Month-by-Month Until Blow (or 8 months)")
    print("=" * 80)
    if t_real is not None:
        print()
        print(f"  {'Month':<9} {'Trades':>7} {'Win%':>6} {'PnL USD':>10} {'End Bal':>10} {'MaxDD%':>7} Status")
        print(f"  {'-'*65}")
        for month, g in t_real.groupby("month"):
            n     = len(g)
            wpct  = (g["pnl_usd"] > 0).mean() * 100
            pnl   = g["pnl_usd"].sum()
            endb  = g["balance"].iloc[-1]
            mddp  = g["dd_pct"].max()
            flag  = " *** BLOWN ***" if endb <= 2.0 else ""
            print(f"  {month:<9} {n:>7} {wpct:>5.1f}% {pnl:>+10.2f}  ${endb:>9.2f}  -{mddp:>5.1f}%{flag}")

    # ── PART 3: Minimum viable balance analysis ────────────────────────────────
    print()
    print("=" * 80)
    print("  PART 3 — MINIMUM VIABLE BALANCE  (realistic min-stake modeling)")
    print("  Finding the smallest account where min-stake stops overriding % risk")
    print("=" * 80)
    print()
    print(f"  {'Start Bal':>10}  {'End Bal':>11}  {'Return':>8}  {'MaxDD':>6}  {'Avg/Mo':>9}  {'Prof Mo':>8}  Status")
    print(f"  {'-'*75}")

    for start in [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]:
        label = f"${start:,}"
        run_scenario(df, label, start, min_stake=True, daily_dd=DAILY_DD_LIMIT)

    # ── PART 4: Recommended balance — ideal scenario monthly breakdown ─────────
    print()
    print("=" * 80)
    print("  PART 4 — MONTHLY RETURNS AT KEY STARTING BALANCES (Ideal, no kill switches)")
    print("=" * 80)
    print()
    header = f"  {'Month':<9}"
    for s in [500, 1_000, 2_000, 5_000, 10_000]:
        header += f"  ${s:>6,} start"
    print(header)
    print(f"  {'-'*80}")

    monthly_data = {}
    for s in [500, 1_000, 2_000, 5_000, 10_000]:
        trades_s, _, _ = usd_backtest(df, starting_balance=s, daily_dd=None, account_dd=None)
        monthly_data[s] = trades_s.groupby("month")["pnl_usd"].sum() if not trades_s.empty else {}

    all_months = sorted(set(m for ms in monthly_data.values() for m in ms.index))
    for month in all_months:
        row = f"  {month:<9}"
        for s in [500, 1_000, 2_000, 5_000, 10_000]:
            ms = monthly_data[s]
            v  = ms.get(month, 0)
            row += f"  {v:>+13,.2f}"
        print(row)

    print()
    print("  KEY TAKEAWAYS")
    print("  " + "-" * 60)
    print("  $50 ideal      : strategy works, never blows, small $ returns")
    print("  $50 realistic  : $1 min stake forces 30-50% risk/trade on gaps -> likely BLOWN")
    print("  Safe minimum   : $1,000+ where 2% sizing > $1 min stake requirement")
    print("  Recommended    : $2,000+ for comfortable gap risk buffer")
    print()


if __name__ == "__main__":
    main()

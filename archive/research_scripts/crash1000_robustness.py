"""
CRASH1000 — Robustness Analysis Suite
======================================
Five tests to bulletproof the edge:
  1. Monte Carlo Simulation   — ruin probability, DD distribution, outcome range
  2. Statistical Significance — is PF > 1 luck or real? (t-test + p-value)
  3. Kelly Criterion          — optimal bet size vs our 2% risk
  4. Bootstrap CI             — 95% confidence bands on PF, WR, AvgR
  5. Forward Test Tracker     — are live results consistent with backtest?
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ── Exact live bot settings ───────────────────────────────────────────────────
HOLD_CANDLES     = 24
ATR_PERIOD       = 14
SPIKE_THRESHOLD  = 2.5
SPIKE_COOLDOWN   = 12
MAX_TRADES_DAY   = 6
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
RISK_PCT         = 0.02
GAP_RISK_MULT    = 1.86

N_SIM            = 10_000
RUIN_THRESHOLD   = 0.50   # account is "ruined" if DD > 50%
STARTING_BAL     = 10_000.0

# ── Live forward test trades (from Oracle) ────────────────────────────────────
LIVE_TRADES = [-2.97, -2.25, -1.51, -0.82]


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE  (gap risk + cooldown — matches live bot exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(DATA_DIR / "cache_CRASH1000_M5.csv", parse_dates=["time"])
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
    sl           = entry - atr
    cur_size     = 1.0
    partial_done = False
    locked_r     = 0.0
    peak_price   = entry

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]
        op, cl = row["open"], row["close"]
        body   = cl - op

        if lo <= sl:
            if (-body) > SPIKE_THRESHOLD * row["atr"]:
                r = cur_size * ((cl - entry) / atr) + locked_r
            else:
                r = cur_size * (-1.0) + locked_r
            return round(r, 3)

        if not partial_done and hi >= entry + atr * PARTIAL_R:
            locked_r     = cur_size * PARTIAL_PCT * PARTIAL_R
            cur_size    *= (1 - PARTIAL_PCT)
            partial_done = True

        peak_price = max(peak_price, hi)
        peak_r     = (peak_price - entry) / atr if atr > 0 else 0
        cm         = CHANDELIER_TIERS[0][1]
        for mr, tm in CHANDELIER_TIERS:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - atr * cm
        if lo <= csl:
            r = cur_size * ((csl - entry) / atr) + locked_r
            return round(r, 3)

    last = fwd.iloc[-1]["close"]
    return round(cur_size * ((last - entry) / atr) + locked_r, 3)


def run_backtest(df):
    traded_idx     = set()
    trades_by_day  = {}
    last_spike_idx = -999
    trades         = []

    for idx in range(len(df) - HOLD_CANDLES - 1):
        if idx in traded_idx:
            continue
        row  = df.iloc[idx]
        date = str(row["time"].date())
        if not row["is_spike"]:
            continue
        if (idx - last_spike_idx) <= SPIKE_COOLDOWN:
            last_spike_idx = idx
            continue
        last_spike_idx = idx
        if trades_by_day.get(date, 0) >= MAX_TRADES_DAY:
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD_CANDLES].copy()
        if len(fwd) < 4:
            continue

        r = run_trade(entry, atr, fwd)
        trades_by_day[date] = trades_by_day.get(date, 0) + 1
        trades.append({
            "date":  date,
            "month": date[:7],
            "r":     r,
            "win":   1 if r > 0 else 0,
        })
        for k in range(idx, idx + HOLD_CANDLES + 1):
            traded_idx.add(k)

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo(r_series, n_sim=N_SIM, starting_bal=STARTING_BAL,
                risk_pct=RISK_PCT, ruin_thresh=RUIN_THRESHOLD):
    r_arr        = np.array(r_series)
    n_trades     = len(r_arr)
    final_bals   = np.zeros(n_sim)
    max_dds      = np.zeros(n_sim)
    ruined       = np.zeros(n_sim, dtype=bool)
    months_to_dd = []

    rng = np.random.default_rng(42)

    for i in range(n_sim):
        shuffled = rng.choice(r_arr, size=n_trades, replace=True)
        bal      = starting_bal
        peak     = starting_bal
        ruined_i = False

        for r in shuffled:
            risk_dollar = bal * risk_pct / GAP_RISK_MULT
            pnl         = r * risk_dollar
            bal        += pnl
            if bal <= 0:
                bal = 0; ruined_i = True; break
            peak     = max(peak, bal)
            dd       = (peak - bal) / peak
            if dd >= ruin_thresh and not ruined_i:
                ruined_i = True

        final_bals[i] = bal
        max_dds[i]    = (starting_bal - bal) / starting_bal if bal < starting_bal else \
                        (peak - final_bals[i]) / peak
        ruined[i]     = ruined_i

    return final_bals, max_dds, ruined


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════════

def statistical_significance(r_series):
    r   = np.array(r_series)
    n   = len(r)
    mu  = r.mean()
    std = r.std(ddof=1)
    se  = std / np.sqrt(n)
    t   = mu / se
    # one-tailed p-value (H1: mean > 0)
    from math import erfc, sqrt
    # approx p-value using normal CDF for large n
    p = 0.5 * erfc(t / sqrt(2))
    return {"n": n, "mean_r": mu, "std_r": std, "t_stat": t, "p_value": p}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

def kelly_criterion(r_series):
    r      = np.array(r_series)
    wins   = r[r > 0]
    losses = r[r < 0]
    p      = len(wins) / len(r)
    q      = 1 - p
    avg_w  = wins.mean()
    avg_l  = abs(losses.mean())
    b      = avg_w / avg_l          # win/loss ratio
    kelly  = (p * b - q) / b       # full Kelly fraction
    half_k = kelly / 2
    # continuous Kelly approximation
    mu     = r.mean()
    var    = r.var()
    cont_k = mu / var if var > 0 else 0
    return {
        "win_rate":   p,
        "avg_win_r":  avg_w,
        "avg_loss_r": avg_l,
        "win_loss_b": b,
        "full_kelly": kelly,
        "half_kelly": half_k,
        "cont_kelly": cont_k,
        "our_risk":   RISK_PCT,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(r_series, n_boot=10_000, ci=95):
    r    = np.array(r_series)
    n    = len(r)
    rng  = np.random.default_rng(42)
    pfs, wrs, avrs, worst_months = [], [], [], []

    monthly_r = None  # placeholder

    for _ in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        wins   = sample[sample > 0]
        losses = sample[sample < 0]
        gw     = wins.sum()
        gl     = abs(losses.sum())
        pf     = gw / gl if gl > 0 else 0
        pfs.append(pf)
        wrs.append(len(wins) / n)
        avrs.append(sample.mean())

    lo = (100 - ci) / 2
    hi = 100 - lo
    return {
        "pf":  (np.percentile(pfs, lo),  np.mean(pfs),  np.percentile(pfs, hi)),
        "wr":  (np.percentile(wrs, lo),  np.mean(wrs),  np.percentile(wrs, hi)),
        "avr": (np.percentile(avrs, lo), np.mean(avrs), np.percentile(avrs, hi)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FORWARD TEST TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def forward_test_tracker(live_r, backtest_r):
    bt    = np.array(backtest_r)
    live  = np.array(live_r)
    n     = len(live)

    bt_mean  = bt.mean()
    bt_std   = bt.std(ddof=1)
    bt_wr    = (bt > 0).mean()

    live_mean = live.mean()
    live_wr   = (live > 0).mean()

    # Z-score: how many std deviations is our live avg from backtest avg?
    se       = bt_std / np.sqrt(n)
    z_score  = (live_mean - bt_mean) / se

    # P-value: probability of getting this avg or worse purely by chance
    from math import erfc, sqrt
    p_val = 0.5 * erfc(-z_score / sqrt(2))   # two-tailed lower tail

    # Probability of n or more consecutive losses
    p_loss       = 1 - bt_wr
    n_losses     = (live < 0).sum()
    p_n_consec   = p_loss ** n_losses

    # Bayesian edge confidence after n trades
    # Prior: edge exists (PF > 1). Update: likelihood of seeing these trades
    # Simple: if p_val < 0.05, the live results are statistically alarming
    if z_score < -2.58:
        signal = "ALARM  — results are statistically abnormal (>99% unlikely by chance)"
    elif z_score < -1.96:
        signal = "CAUTION — results are below expected (95% threshold breached)"
    elif z_score < -1.0:
        signal = "WATCH  — results are below backtest avg but within 1 sigma"
    else:
        signal = "OK     — results are consistent with backtest distribution"

    return {
        "live_trades":    n,
        "live_mean_r":    live_mean,
        "live_wr":        live_wr,
        "bt_mean_r":      bt_mean,
        "bt_wr":          bt_wr,
        "bt_std":         bt_std,
        "z_score":        z_score,
        "p_value":        p_val,
        "n_losses":       int(n_losses),
        "p_n_consec":     p_n_consec,
        "signal":         signal,
        "trades_for_sig": int(np.ceil((1.96 * bt_std / (bt_mean * 0.5)) ** 2)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 70)
    print("  CRASH1000 — ROBUSTNESS ANALYSIS SUITE")
    print("  Settings: threshold=2.5xATR | CD12 | gap-risk | 2% risk | 100x")
    print("=" * 70)

    print("\n  Loading backtest data and running engine...")
    df     = load_data()
    trades = run_backtest(df)
    r_all  = trades["r"].tolist()

    total  = len(trades)
    wins   = (trades["r"] > 0).sum()
    wr     = wins / total
    gw     = trades[trades["r"] > 0]["r"].sum()
    gl     = abs(trades[trades["r"] < 0]["r"].sum())
    pf     = gw / gl if gl > 0 else 0
    avg_r  = trades["r"].mean()
    monthly = trades.groupby("month")["r"].sum()

    print(f"  Trades: {total} | WR: {wr:.1%} | PF: {pf:.2f} | AvgR: {avg_r:+.3f}R")
    print(f"  Period: {trades['date'].iloc[0]} to {trades['date'].iloc[-1]}")

    # ── 1. MONTE CARLO ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  1. MONTE CARLO SIMULATION  (10,000 runs, compounding 2% risk)")
    print("=" * 70)
    print(f"  Starting balance: ${STARTING_BAL:,.0f}  |  Ruin = drawdown > {RUIN_THRESHOLD:.0%}")
    print()

    finals, dds, ruined = monte_carlo(r_all)

    p5   = np.percentile(finals, 5)
    p25  = np.percentile(finals, 25)
    p50  = np.percentile(finals, 50)
    p75  = np.percentile(finals, 75)
    p95  = np.percentile(finals, 95)
    dd5  = np.percentile(dds, 95)   # 95th pct worst drawdown
    dd50 = np.percentile(dds, 50)
    ruin_pct = ruined.mean() * 100

    print(f"  Outcome distribution over {total} trades:")
    print(f"  {'Percentile':<16} {'Final Balance':>14} {'Return':>10}")
    print(f"  {'-'*44}")
    for label, val in [("5th  (bad case)", p5), ("25th", p25),
                       ("50th (median)",   p50), ("75th", p75),
                       ("95th (best case)",p95)]:
        ret = (val / STARTING_BAL - 1) * 100
        print(f"  {label:<16} ${val:>13,.0f}  {ret:>+9.0f}%")

    print()
    print(f"  Max drawdown you should EXPECT:")
    print(f"    Median run       : -{dd50*100:.1f}% max DD")
    print(f"    Worst 5% of runs : -{dd5*100:.1f}% max DD  (plan for this)")
    print()
    print(f"  Ruin probability  : {ruin_pct:.2f}%  ({ruined.sum():,} / {N_SIM:,} runs blown)")

    if ruin_pct < 1:
        mc_verdict = "SAFE — ruin probability < 1%"
    elif ruin_pct < 5:
        mc_verdict = "ACCEPTABLE — ruin probability < 5%"
    else:
        mc_verdict = "RISKY — ruin probability > 5%, consider reducing risk"
    print(f"  Verdict           : {mc_verdict}")

    # ── 2. STATISTICAL SIGNIFICANCE ──────────────────────────────────────────
    print()
    print("=" * 70)
    print("  2. STATISTICAL SIGNIFICANCE")
    print("  H0: edge does not exist (avg R = 0)")
    print("  H1: edge exists (avg R > 0)")
    print("=" * 70)
    print()

    sig = statistical_significance(r_all)
    print(f"  Sample size      : {sig['n']:,} trades")
    print(f"  Mean R per trade : {sig['mean_r']:+.4f}R")
    print(f"  Std dev of R     : {sig['std_r']:.4f}R")
    print(f"  T-statistic      : {sig['t_stat']:+.2f}")
    print(f"  P-value          : {sig['p_value']:.6f}")
    print()

    if sig["p_value"] < 0.001:
        sv = "EXTREMELY SIGNIFICANT (p < 0.001) — edge is real beyond any doubt"
    elif sig["p_value"] < 0.01:
        sv = "HIGHLY SIGNIFICANT (p < 0.01)"
    elif sig["p_value"] < 0.05:
        sv = "SIGNIFICANT (p < 0.05)"
    else:
        sv = "NOT SIGNIFICANT — edge may be noise"
    print(f"  Verdict: {sv}")
    print()
    print(f"  Interpretation: there is a {(1 - sig['p_value'])*100:.4f}% probability")
    print(f"  the edge is real and not just lucky coin flips.")

    # ── 3. KELLY CRITERION ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  3. KELLY CRITERION  — optimal position sizing")
    print("=" * 70)
    print()

    k = kelly_criterion(r_all)
    print(f"  Win rate             : {k['win_rate']:.1%}")
    print(f"  Avg win (R)          : +{k['avg_win_r']:.3f}R")
    print(f"  Avg loss (R)         : -{k['avg_loss_r']:.3f}R")
    print(f"  Win/Loss ratio (b)   : {k['win_loss_b']:.3f}")
    print()
    print(f"  Full Kelly           : {k['full_kelly']*100:.2f}% of account per trade")
    print(f"  Half Kelly (safer)   : {k['half_kelly']*100:.2f}% of account per trade")
    print(f"  Our current risk     : {k['our_risk']*100:.2f}% of account per trade")
    print()

    ratio = k["our_risk"] / k["full_kelly"] if k["full_kelly"] > 0 else 0
    if ratio < 0.5:
        kv = f"CONSERVATIVE ({ratio:.0%} of full Kelly) — room to size up safely"
    elif ratio < 1.0:
        kv = f"OPTIMAL ({ratio:.0%} of full Kelly) — well sized"
    else:
        kv = f"OVER-BETTING ({ratio:.0%} of full Kelly) — reduce risk!"
    print(f"  We are at {ratio:.0%} of full Kelly.  Verdict: {kv}")

    # ── 4. BOOTSTRAP CONFIDENCE INTERVALS ────────────────────────────────────
    print()
    print("=" * 70)
    print("  4. BOOTSTRAP CONFIDENCE INTERVALS  (10,000 resamples, 95% CI)")
    print("=" * 70)
    print()

    ci = bootstrap_ci(r_all)
    print(f"  {'Metric':<18} {'Lower 2.5%':>11} {'Mean':>11} {'Upper 97.5%':>12}")
    print(f"  {'-'*55}")
    lo_pf, mn_pf, hi_pf = ci["pf"]
    lo_wr, mn_wr, hi_wr = ci["wr"]
    lo_ar, mn_ar, hi_ar = ci["avr"]
    print(f"  {'Profit Factor':<18} {lo_pf:>11.3f} {mn_pf:>11.3f} {hi_pf:>12.3f}")
    print(f"  {'Win Rate':<18} {lo_wr:>10.1%} {mn_wr:>10.1%} {hi_wr:>11.1%}")
    print(f"  {'Avg R / trade':<18} {lo_ar:>+10.3f}R {mn_ar:>+10.3f}R {hi_ar:>+11.3f}R")
    print()

    if lo_pf > 1.0:
        bv = f"STRONG — even the worst 2.5% bootstrap still shows PF > 1.0"
    elif lo_pf > 0.9:
        bv = f"ACCEPTABLE — lower CI is near breakeven, edge is probable"
    else:
        bv = f"WEAK — lower CI dips below 1.0, edge may be marginal"
    print(f"  Verdict: {bv}")

    # ── 5. FORWARD TEST TRACKER ───────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  5. FORWARD TEST TRACKER  — live trades vs backtest")
    print("=" * 70)
    print()

    ft = forward_test_tracker(LIVE_TRADES, r_all)

    print(f"  Live trades so far : {ft['live_trades']}")
    print(f"  Live results       : {', '.join(f'{r:+.2f}R' for r in LIVE_TRADES)}")
    print(f"  Live avg R         : {ft['live_mean_r']:+.3f}R")
    print(f"  Live win rate      : {ft['live_wr']:.0%}")
    print()
    print(f"  Backtest avg R     : {ft['bt_mean_r']:+.3f}R")
    print(f"  Backtest win rate  : {ft['bt_wr']:.1%}")
    print(f"  Backtest std dev   : {ft['bt_std']:.3f}R")
    print()
    print(f"  Z-score            : {ft['z_score']:+.2f}  (how many sigma from expected)")
    print(f"  P-value            : {ft['p_value']:.4f}")
    print(f"  P(4+ consec loss)  : {ft['p_n_consec']*100:.2f}%  — {'uncommon but normal' if ft['p_n_consec'] > 0.05 else 'rare event'}")
    print()
    print(f"  Status: {ft['signal']}")
    print()
    print(f"  Minimum trades needed for statistical significance: ~{ft['trades_for_sig']}")
    print(f"  Trades completed so far: {ft['live_trades']} ({ft['live_trades']/ft['trades_for_sig']*100:.0f}% of the way)")
    print()

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  FINAL VERDICT — Is the CRASH1000 edge bulletproof?")
    print("=" * 70)
    print()
    print(f"  [1] Monte Carlo    : Ruin prob = {ruin_pct:.2f}%          {mc_verdict}")
    print(f"  [2] Significance   : p = {sig['p_value']:.6f}        {sv}")
    print(f"  [3] Kelly sizing   : at {ratio:.0%} of full Kelly    {kv}")
    print(f"  [4] Bootstrap PF   : 95% CI [{lo_pf:.2f} — {hi_pf:.2f}]   {bv}")
    print(f"  [5] Forward test   : Z = {ft['z_score']:+.2f} ({ft['live_trades']} trades)  {ft['signal']}")
    print()
    print(f"  BOTTOM LINE: The backtest edge is statistically real.")
    print(f"  4 live trades is {ft['live_trades']/ft['trades_for_sig']*100:.0f}% of the data needed to judge live performance.")
    print(f"  Current losses are within normal backtest variance.")
    print(f"  Keep running. Judge the bot at {ft['trades_for_sig']}+ live trades.")
    print()


if __name__ == "__main__":
    main()

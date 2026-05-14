"""
BOOM1000 Deep Analysis — vs CRASH1000
"""
import numpy as np, pandas as pd, sys, io, math, random
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR   = Path(__file__).resolve().parent / "data"
ATR_PERIOD = 14
THRESHOLD  = 2.5
HOLD       = 24
MAX_DAY    = 6
random.seed(42); np.random.seed(42)


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_boom(fname):
    df = pd.read_csv(BASE_DIR / fname, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]  = tr.rolling(ATR_PERIOD).mean()
    body       = df["close"] - df["open"]
    df["boom_spike"] = body > THRESHOLD * df["atr"]
    df["ema8"]  = df["close"].ewm(span=8,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    return df.dropna(subset=["atr"]).reset_index(drop=True)

def load_crash(fname):
    df = pd.read_csv(BASE_DIR / fname, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(ATR_PERIOD).mean()
    body           = df["close"] - df["open"]
    df["is_spike"] = (-body) > THRESHOLD * df["atr"]
    return df.dropna(subset=["atr"]).reset_index(drop=True)


# ── Trade simulators ───────────────────────────────────────────────────────────
CHANDELIER = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]

def run_sell(entry, atr, fwd):
    sl = entry + atr
    cur = 1.0; partial_done = False; locked_r = 0.0
    peak = entry; peak_r = 0.0
    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        spike = (cl - op) > THRESHOLD * row["atr"]
        if hi >= sl:
            r = cur * ((entry - cl) / atr) + locked_r if spike else cur * (-1.0) + locked_r
            return round(r, 3), "SL"
        if not partial_done and lo <= entry - atr * 2.0:
            locked_r = cur * 0.5 * 2.0; cur *= 0.5; partial_done = True
        peak = min(peak, lo); peak_r = (entry - peak) / atr
        cm = 3.0
        for mr, tm in CHANDELIER:
            if peak_r >= mr: cm = tm
        csl = peak + atr * cm
        if hi >= csl:
            return round(cur * ((entry - csl) / atr) + locked_r, 3), "CHANDELIER"
    last = fwd.iloc[-1]["close"]
    return round(cur * ((entry - last) / atr) + locked_r, 3), "TIME"

def run_buy(entry, atr, fwd):
    sl = entry - atr
    cur = 1.0; partial_done = False; locked_r = 0.0
    peak = entry; peak_r = 0.0
    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        spike = (-(cl - op)) > THRESHOLD * row["atr"]
        if lo <= sl:
            r = cur * ((cl - entry) / atr) + locked_r if spike else cur * (-1.0) + locked_r
            return round(r, 3), "SL"
        if not partial_done and hi >= entry + atr * 2.0:
            locked_r = cur * 0.5 * 2.0; cur *= 0.5; partial_done = True
        peak = max(peak, hi); peak_r = (peak - entry) / atr
        cm = 3.0
        for mr, tm in CHANDELIER:
            if peak_r >= mr: cm = tm
        csl = peak - atr * cm
        if lo <= csl:
            return round(cur * ((csl - entry) / atr) + locked_r, 3), "CHANDELIER"
    last = fwd.iloc[-1]["close"]
    return round(cur * ((last - entry) / atr) + locked_r, 3), "TIME"


# ── Backtests ──────────────────────────────────────────────────────────────────
def backtest_boom(df):
    sig = df["boom_spike"] & (df["ema8"] > df["ema21"]) & (df.index < len(df) - HOLD - 1)
    traded = set(); tbd = {}; trades = []
    for idx in df.index[sig]:
        if idx in traded:
            continue
        row = df.iloc[idx]
        date = str(row["time"].date()); month = date[:7]
        dow = pd.Timestamp(date).day_name()
        if tbd.get(date, 0) >= MAX_DAY:
            continue
        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else float(row["close"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD].copy()
        if len(fwd) < 4:
            continue
        r, reason = run_sell(entry, atr, fwd)
        tbd[date] = tbd.get(date, 0) + 1
        trades.append({"date": date, "month": month, "dow": dow, "r": r,
                       "result": "WIN" if r > 0 else "LOSS", "reason": reason})
        for k in range(idx, idx + HOLD + 1):
            traded.add(k)
    return pd.DataFrame(trades)

def backtest_crash(df):
    traded = set(); tbd = {}; trades = []; lsi = -999
    for idx in range(len(df) - HOLD - 1):
        if idx in traded:
            continue
        row = df.iloc[idx]
        date = str(row["time"].date()); month = date[:7]
        dow = pd.Timestamp(date).day_name()
        if not row["is_spike"]:
            continue
        if (idx - lsi) <= 12:
            lsi = idx; continue
        lsi = idx
        if tbd.get(date, 0) >= MAX_DAY:
            continue
        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"])
        fwd   = df.iloc[idx + 1: idx + 1 + HOLD].copy()
        if len(fwd) < 4:
            continue
        r, reason = run_buy(entry, atr, fwd)
        tbd[date] = tbd.get(date, 0) + 1
        trades.append({"date": date, "month": month, "dow": dow, "r": r,
                       "result": "WIN" if r > 0 else "LOSS", "reason": reason})
        for k in range(idx, idx + HOLD + 1):
            traded.add(k)
    return pd.DataFrame(trades)


# ── Deep analysis ──────────────────────────────────────────────────────────────
def deep_analysis(t, label, risk_pct=0.02, start_bal=10000, n_mc=10000):
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    n      = len(t)
    w      = (t["result"] == "WIN").sum()
    wr     = w / n
    r_arr  = t["r"].values
    gw     = t[t["r"] > 0]["r"].sum()
    gl     = abs(t[t["r"] < 0]["r"].sum())
    pf     = gw / gl if gl > 0 else 0
    avg_r  = r_arr.mean()
    std_r  = r_arr.std()
    avg_win  = t[t["r"] > 0]["r"].mean() if w > 0 else 0
    avg_loss = t[t["r"] < 0]["r"].mean() if (t["r"] < 0).any() else 0
    streak = ms = 0
    for r in t["result"]:
        streak = streak + 1 if r == "LOSS" else 0
        ms = max(ms, streak)
    days          = (pd.to_datetime(t["date"].max()) - pd.to_datetime(t["date"].min())).days + 1
    trading_days  = t["date"].nunique()
    months_count  = t["month"].nunique()

    print(f"\n  CORE EDGE METRICS")
    print(f"  {'Trades total':<28}: {n}")
    print(f"  {'Period':<28}: {t['date'].min()} to {t['date'].max()} ({months_count} months)")
    print(f"  {'Win Rate':<28}: {wr*100:.2f}%")
    print(f"  {'Profit Factor':<28}: {pf:.3f}")
    print(f"  {'Avg R per trade':<28}: {avg_r:+.4f}R")
    print(f"  {'Avg Win':<28}: {avg_win:+.3f}R")
    print(f"  {'Avg Loss':<28}: {avg_loss:+.3f}R")
    print(f"  {'Win/Loss ratio':<28}: {abs(avg_win / avg_loss):.2f}x")
    print(f"  {'Std R':<28}: {std_r:.3f}R")
    print(f"  {'Max consec losses':<28}: {ms}")

    tstat = avg_r / (std_r / math.sqrt(n)) if std_r > 0 else 0
    pval  = math.erfc(abs(tstat) / math.sqrt(2))
    sig_trades = int((1.96 * std_r / avg_r) ** 2) if avg_r > 0 else 9999
    print(f"\n  STATISTICAL SIGNIFICANCE")
    print(f"  {'T-statistic':<28}: {tstat:.3f}")
    pval_str = f"{pval:.8f}"
    sig_label = "HIGHLY SIGNIFICANT" if pval < 0.001 else ("SIGNIFICANT" if pval < 0.05 else "NOT SIG")
    print(f"  {'P-value':<28}: {pval_str}  ({sig_label})")
    print(f"  {'Trades for 95% sig':<28}: {sig_trades}")

    b = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    q = 1 - wr
    kelly_full = ((wr * b) - q) / b if b > 0 else 0
    kelly_ratio = risk_pct / kelly_full if kelly_full > 0 else 0
    kelly_label = "CONSERVATIVE" if kelly_ratio < 0.5 else ("OPTIMAL" if kelly_ratio < 1.0 else "OVERBETTING")
    print(f"\n  KELLY CRITERION")
    print(f"  {'Full Kelly':<28}: {kelly_full*100:.2f}%")
    print(f"  {'Half Kelly':<28}: {kelly_full/2*100:.2f}%")
    print(f"  {'Our risk (2%)':<28}: {risk_pct*100:.1f}% = {kelly_ratio*100:.0f}% of full Kelly  ({kelly_label})")

    daily = t.groupby("date").agg(trades=("r","count"), daily_r=("r","sum")).reset_index()
    print(f"\n  DAILY BREAKDOWN  ({trading_days} active days out of {days} calendar days)")
    print(f"  {'Avg trades / active day':<28}: {daily['trades'].mean():.2f}")
    print(f"  {'Avg R / active day':<28}: {daily['daily_r'].mean():+.3f}R")
    print(f"  {'Best single day':<28}: {daily['daily_r'].max():+.2f}R")
    print(f"  {'Worst single day':<28}: {daily['daily_r'].min():+.2f}R")
    print(f"  {'Profitable active days':<28}: {(daily['daily_r']>0).sum()}/{len(daily)}  = {(daily['daily_r']>0).mean()*100:.0f}%")
    print(f"  {'Idle days (0 trades)':<28}: {days - trading_days}  ({(days-trading_days)/days*100:.0f}% of calendar)")

    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = (t.groupby("dow")
             .agg(trades=("r","count"), avg_r=("r","mean"),
                  wr=("result", lambda x: (x=="WIN").sum()/len(x)))
             .reindex([d for d in dow_order if d in t["dow"].unique()]))
    print(f"\n  DAY-OF-WEEK BREAKDOWN")
    for d, row in dow.iterrows():
        bar = "#" * max(0, int(row["avg_r"] * 20))
        print(f"    {d:<12}: {row['trades']:>5} trades  AvgR={row['avg_r']:+.3f}R  WR={row['wr']*100:.0f}%  {bar}")

    monthly   = t.groupby("month")["r"].agg(["sum","count","mean"])
    prof_m    = (monthly["sum"] > 0).sum()
    OOS_CUT   = "2026-02"
    print(f"\n  MONTHLY BREAKDOWN  ({months_count} months, {prof_m} profitable = {prof_m/months_count*100:.0f}%)")
    print(f"  {'Month':<10} {'Trades':>7} {'R/month':>10} {'AvgR/tr':>10} {'WR':>7}  Bar")
    print(f"  {'-'*66}")
    for m in sorted(monthly.index):
        tm   = t[t["month"] == m]
        wr_m = (tm["result"] == "WIN").sum() / len(tm)
        r_m  = monthly.loc[m, "sum"]
        avg_m= monthly.loc[m, "mean"]
        bar  = "#" * int(abs(r_m) / 5)
        tag  = " [OOS]" if m >= OOS_CUT else ""
        print(f"  {m:<10} {len(tm):>7} {r_m:>+9.1f}R {avg_m:>+9.3f}R {wr_m*100:>6.0f}%  {'|'+bar if r_m>0 else bar+'|'}{tag}")
    print(f"  {'AVERAGE':<10} {monthly['count'].mean():>7.0f} {monthly['sum'].mean():>+9.1f}R {monthly['mean'].mean():>+9.3f}R")
    print(f"  Worst month: {monthly['sum'].min():+.1f}R  Best month: {monthly['sum'].max():+.1f}R")

    print(f"\n  EXIT REASON BREAKDOWN")
    reasons = t.groupby("reason")["r"].agg(["count","mean","sum"])
    for reason, row in reasons.iterrows():
        pct = row["count"] / n * 100
        print(f"    {reason:<14}: {row['count']:>5} ({pct:>4.0f}%)  AvgR={row['mean']:+.3f}R  TotalR={row['sum']:+.0f}R")

    print(f"\n  USD PROJECTION  ($10,000 start, 2% risk, compounding)")
    bal = start_bal; peak = bal
    for r in r_arr:
        bal  += r * (bal * risk_pct)
        peak  = max(peak, bal)
    total_ret = (bal - start_bal) / start_bal * 100
    print(f"  {'Final balance':<28}: ${bal:>12,.2f}  ({total_ret:+.0f}%)")
    print(f"  {'Monthly avg ($)':<28}: ${(bal-start_bal)/months_count:>10,.0f}/month avg")
    avg_r_day = daily["daily_r"].mean()
    print(f"  {'Avg $ on active day (2%)':<28}: ${avg_r_day * start_bal * risk_pct:>8,.2f}/day (on $10k)")

    print(f"\n  MONTE CARLO  ({n_mc:,} runs, 2% risk, compounding)")
    finals = []; max_dds = []
    for _ in range(n_mc):
        sample = np.random.choice(r_arr, size=n, replace=True)
        b = start_bal; pk = b; mdd = 0
        for r in sample:
            b  += r * (b * risk_pct)
            pk  = max(pk, b)
            mdd = max(mdd, (pk - b) / pk)
        finals.append(b); max_dds.append(mdd * 100)
    finals  = np.array(finals)
    max_dds = np.array(max_dds)
    ruin    = np.mean(finals < start_bal * 0.5) * 100
    p5, p25, p50, p75, p95 = np.percentile(finals, [5, 25, 50, 75, 95])
    print(f"  {'Ruin prob (balance < $5k)':<28}: {ruin:.2f}%")
    print(f"  {'5th percentile':<28}: ${p5:>12,.0f}")
    print(f"  {'25th percentile':<28}: ${p25:>12,.0f}")
    print(f"  {'Median (50th)':<28}: ${p50:>12,.0f}  (+{(p50/start_bal-1)*100:.0f}%)")
    print(f"  {'75th percentile':<28}: ${p75:>12,.0f}")
    print(f"  {'95th percentile':<28}: ${p95:>12,.0f}")
    print(f"  {'Avg max drawdown':<28}: {np.mean(max_dds):.1f}%  (worst 5%: {np.percentile(max_dds,95):.1f}%)")

    return {
        "n": n, "wr": wr, "pf": pf, "avg_r": avg_r, "std_r": std_r,
        "kelly_full": kelly_full, "kelly_ratio": kelly_ratio,
        "ruin": ruin, "p50": p50, "p5": p5, "p95": p95,
        "tstat": tstat, "pval": pval, "sig_trades": sig_trades, "ms": ms,
        "avg_day_r": daily["daily_r"].mean(),
        "avg_trades_day": daily["trades"].mean(),
        "prof_days_pct": (daily["daily_r"] > 0).mean() * 100,
        "months": months_count, "prof_months": prof_m,
        "avg_month_r": monthly["sum"].mean(),
        "worst_month": monthly["sum"].min(),
        "avg_win": avg_win, "avg_loss": avg_loss,
        "final_bal": bal,
    }


def main():
    df_b = load_boom("cache_BOOM1000_M5.csv")
    df_c = load_crash("cache_CRASH1000_M5.csv")

    tb = backtest_boom(df_b)
    tc = backtest_crash(df_c)

    sb = deep_analysis(tb, "BOOM1000  — Spike SELL + EMA8 > EMA21  (NEW STRATEGY)")
    sc = deep_analysis(tc, "CRASH1000 — Spike BUY  + CD12           (LIVE STRATEGY)")

    print(f"\n{'='*72}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Metric':<32} {'BOOM1000':>15} {'CRASH1000':>15}")
    print(f"  {'-'*64}")
    rows = [
        ("Trades total",          f"{sb['n']}",               f"{sc['n']}"),
        ("Win Rate",              f"{sb['wr']*100:.2f}%",      f"{sc['wr']*100:.2f}%"),
        ("Profit Factor",         f"{sb['pf']:.3f}",           f"{sc['pf']:.3f}"),
        ("Avg R / trade",         f"{sb['avg_r']:+.4f}R",      f"{sc['avg_r']:+.4f}R"),
        ("Avg Win",               f"{sb['avg_win']:+.3f}R",    f"{sc['avg_win']:+.3f}R"),
        ("Avg Loss",              f"{sb['avg_loss']:+.3f}R",   f"{sc['avg_loss']:+.3f}R"),
        ("Avg R / active day",    f"{sb['avg_day_r']:+.3f}R",  f"{sc['avg_day_r']:+.3f}R"),
        ("Avg trades / day",      f"{sb['avg_trades_day']:.2f}", f"{sc['avg_trades_day']:.2f}"),
        ("Profitable days %",     f"{sb['prof_days_pct']:.0f}%", f"{sc['prof_days_pct']:.0f}%"),
        ("Avg R / month",         f"{sb['avg_month_r']:+.1f}R", f"{sc['avg_month_r']:+.1f}R"),
        ("Worst month",           f"{sb['worst_month']:+.1f}R", f"{sc['worst_month']:+.1f}R"),
        ("Profitable months",     f"{sb['prof_months']}/{sb['months']}", f"{sc['prof_months']}/{sc['months']}"),
        ("T-statistic",           f"{sb['tstat']:.3f}",        f"{sc['tstat']:.3f}"),
        ("P-value",               f"{sb['pval']:.6f}",         f"{sc['pval']:.6f}"),
        ("Trades for 95% sig",    f"{sb['sig_trades']}",       f"{sc['sig_trades']}"),
        ("Full Kelly %",          f"{sb['kelly_full']*100:.2f}%", f"{sc['kelly_full']*100:.2f}%"),
        ("Our risk = X% of Kelly",f"{sb['kelly_ratio']*100:.0f}%",f"{sc['kelly_ratio']*100:.0f}%"),
        ("Max consec losses",     f"{sb['ms']}",               f"{sc['ms']}"),
        ("Ruin probability",      f"{sb['ruin']:.2f}%",        f"{sc['ruin']:.2f}%"),
        ("MC Median balance",     f"${sb['p50']:,.0f}",        f"${sc['p50']:,.0f}"),
        ("MC 5th percentile",     f"${sb['p5']:,.0f}",         f"${sc['p5']:,.0f}"),
        ("MC 95th percentile",    f"${sb['p95']:,.0f}",        f"${sc['p95']:,.0f}"),
        ("Final balance (linear)",f"${sb['final_bal']:,.0f}",  f"${sc['final_bal']:,.0f}"),
    ]
    for m, b, c in rows:
        print(f"  {m:<32} {b:>15} {c:>15}")

    print(f"\n  VERDICT")
    print(f"  BOOM1000  : PF={sb['pf']:.2f}  T={sb['tstat']:.2f}  {sb['prof_months']}/{sb['months']} months  Ruin={sb['ruin']:.1f}%  Kelly={sb['kelly_ratio']*100:.0f}% conservative")
    print(f"  CRASH1000 : PF={sc['pf']:.2f}  T={sc['tstat']:.2f}  {sc['prof_months']}/{sc['months']} months  Ruin={sc['ruin']:.1f}%  Kelly={sc['kelly_ratio']*100:.0f}% conservative")
    print()


if __name__ == "__main__":
    main()

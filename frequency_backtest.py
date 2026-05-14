"""
frequency_backtest.py — Trade Frequency & Signal Selectivity Analysis
======================================================================
Answers: How many trades do CRASH1000 and BOOM1000 actually take per month?
         Is the signal selective enough, or are we overtrading?

Reports per month:
  - Spikes detected
  - Passed filter (trend direction OK)
  - Blocked by cooldown
  - Blocked by daily cap
  - Trades actually taken
  - Win rate + PF that month

Uses exact same signal logic as the live bots / parameter_sensitivity.py
"""

import sys, io, math
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"

BASE = {
    "spike_mult":    2.5,
    "atr_period":    14,
    "hold":          24,
    "h1_ema_period": 21,
    "partial_r":     2.0,
    "chand_init":    3.0,
    "max_day":       6,
    "ema_fast":      8,
    "ema_slow":      21,
}
COOLDOWN = 12


def load_raw(symbol: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"cache_{symbol}_{gran}.csv"
    df   = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def build_merged(raw_m5: pd.DataFrame, raw_h1: pd.DataFrame) -> pd.DataFrame:
    h1 = raw_h1.copy().sort_values("time").reset_index(drop=True)
    p  = BASE["h1_ema_period"]
    h1[f"h1_ema{p}"] = h1["close"].ewm(span=p, adjust=False).mean()
    h1.rename(columns={"close": "h1_close"}, inplace=True)
    keep     = ["time", "h1_close", f"h1_ema{p}"]
    h1_slim  = h1[keep].dropna().sort_values("time").reset_index(drop=True)
    m5       = raw_m5.copy().sort_values("time").reset_index(drop=True)
    merged   = pd.merge_asof(m5, h1_slim, on="time", direction="backward")
    return merged.dropna(subset=["h1_close"]).reset_index(drop=True)


def compute_crash(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(BASE["atr_period"]).mean()
    body           = df["open"] - df["close"]
    df["is_spike"] = body > BASE["spike_mult"] * df["atr"]
    col            = f"h1_ema{BASE['h1_ema_period']}"
    df["filter"]   = df["h1_close"] > df[col]
    return df.dropna(subset=["atr", col]).reset_index(drop=True)


def compute_boom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(BASE["atr_period"]).mean()
    body           = df["close"] - df["open"]
    df["is_spike"] = body > BASE["spike_mult"] * df["atr"]
    df["ema_f"]    = df["close"].ewm(span=BASE["ema_fast"],  adjust=False).mean()
    df["ema_s"]    = df["close"].ewm(span=BASE["ema_slow"],  adjust=False).mean()
    col            = f"h1_ema{BASE['h1_ema_period']}"
    df["filter"]   = (df["ema_f"] > df["ema_s"]) & (df["h1_close"] < df[col])
    return df.dropna(subset=["atr", col, "ema_f", "ema_s"]).reset_index(drop=True)


def _chand_tiers(init: float):
    return [(0.0, init), (2.0, init - 0.5), (4.0, init - 1.0)]


def run_buy(entry, atr, fwd):
    sl = entry - atr
    size = 1.0; partial_done = False; locked_r = 0.0; peak = entry
    tiers = _chand_tiers(BASE["chand_init"])
    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_gap = (op - cl) > BASE["spike_mult"] * row["atr"]
        if lo <= sl:
            r = size * ((cl - entry) / atr) + locked_r if is_gap else size * (-1.0) + locked_r
            return round(r, 3), "SL"
        if not partial_done and hi >= entry + atr * BASE["partial_r"]:
            locked_r += size * 0.5 * BASE["partial_r"]; size *= 0.5; partial_done = True
        peak   = max(peak, hi); peak_r = (peak - entry) / atr
        cm     = tiers[0][1]
        for mr, tm in tiers:
            if peak_r >= mr: cm = tm
        csl = peak - atr * cm
        if lo <= csl:
            return round(size * ((csl - entry) / atr) + locked_r, 3), "CHANDELIER"
    last = fwd.iloc[-1]["close"]
    return round(size * ((last - entry) / atr) + locked_r, 3), "TIME"


def run_sell(entry, atr, fwd):
    sl = entry + atr
    size = 1.0; partial_done = False; locked_r = 0.0; trough = entry
    tiers = _chand_tiers(BASE["chand_init"])
    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_gap = (cl - op) > BASE["spike_mult"] * row["atr"]
        if hi >= sl:
            r = size * ((entry - cl) / atr) + locked_r if is_gap else size * (-1.0) + locked_r
            return round(r, 3), "SL"
        if not partial_done and lo <= entry - atr * BASE["partial_r"]:
            locked_r += size * 0.5 * BASE["partial_r"]; size *= 0.5; partial_done = True
        trough   = min(trough, lo); trough_r = (entry - trough) / atr
        cm       = tiers[0][1]
        for mr, tm in tiers:
            if trough_r >= mr: cm = tm
        csl = trough + atr * cm
        if hi >= csl:
            return round(size * ((entry - csl) / atr) + locked_r, 3), "CHANDELIER"
    last = fwd.iloc[-1]["close"]
    return round(size * ((entry - last) / atr) + locked_r, 3), "TIME"


def run_frequency(df: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (trades_df, skips_df).
    skips_df tracks every spike that was detected but NOT traded, with reason.
    """
    traded = set(); tpd: dict = {}; lsi = -999
    trades = []
    skips  = []

    for idx in range(len(df) - BASE["hold"] - 1):
        if idx in traded:
            continue
        row = df.iloc[idx]
        if not row["is_spike"]:
            continue

        date  = str(row["time"].date())
        month = date[:7]
        hour  = row["time"].hour

        # Cooldown block
        if (idx - lsi) <= COOLDOWN:
            lsi = idx
            skips.append({"month": month, "date": date, "hour": hour, "reason": "COOLDOWN"})
            continue
        lsi = idx

        # Filter block
        if not row["filter"]:
            skips.append({"month": month, "date": date, "hour": hour, "reason": "FILTER"})
            continue

        # Daily cap block
        if tpd.get(date, 0) >= BASE["max_day"]:
            skips.append({"month": month, "date": date, "hour": hour, "reason": "DAILY_CAP"})
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else float(row["close"])
        fwd   = df.iloc[idx + 1: idx + 1 + BASE["hold"]].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_buy(entry, atr, fwd) if direction == "BUY" else run_sell(entry, atr, fwd)
        tpd[date] = tpd.get(date, 0) + 1
        trades.append({"month": month, "date": date, "r": r,
                        "result": "W" if r > 0 else "L", "exit": reason})
        for k in range(idx, idx + BASE["hold"] + 1):
            traded.add(k)

    return pd.DataFrame(trades), pd.DataFrame(skips)


def report(symbol: str, direction: str, trades: pd.DataFrame, skips: pd.DataFrame, df: pd.DataFrame):
    print(f"\n{'='*80}")
    print(f"  {symbol}  ({direction})")
    print(f"  Data: {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"  Total spikes raw : {int(df['is_spike'].sum()):,}")
    print(f"  Total traded     : {len(trades)}")
    if not trades.empty:
        wins = trades[trades["r"] > 0]
        losses = trades[trades["r"] < 0]
        wr  = len(wins) / len(trades) * 100
        pf  = wins["r"].sum() / abs(losses["r"].sum()) if len(losses) > 0 else float("inf")
        avg = trades["r"].mean()
        print(f"  Overall WR       : {wr:.1f}%")
        print(f"  Overall PF       : {pf:.3f}")
        print(f"  Avg R per trade  : {avg:+.3f}")
    print(f"{'='*80}")

    if not trades.empty:
        print(f"\n  {'Month':<10} {'Trades':>7} {'Avg/Day':>8} {'Wins':>6} {'Losses':>7} {'WR%':>6} {'PF':>7} {'AvgR':>7} | {'Skips':>6}  FILTER  COOLDOWN  DAYCAP")
        print(f"  {'─'*95}")

        # trading days per month from data
        df["date_str"] = df["time"].dt.date.astype(str)
        trading_days_per_month = df.groupby(df["time"].dt.to_period("M"))["date_str"].nunique()

        months = sorted(set(trades["month"].tolist() + (skips["month"].tolist() if not skips.empty else [])))
        total_t = total_w = total_l = 0
        total_wins_r = total_losses_r = 0.0

        for m in months:
            mt = trades[trades["month"] == m]
            ms = skips[skips["month"] == m] if not skips.empty else pd.DataFrame()

            n  = len(mt)
            w  = len(mt[mt["r"] > 0])
            l  = len(mt[mt["r"] < 0])
            wr = w / n * 100 if n > 0 else 0
            w_r = mt[mt["r"] > 0]["r"].sum()
            l_r = abs(mt[mt["r"] < 0]["r"].sum())
            pf  = w_r / l_r if l_r > 0 else float("inf")
            avg = mt["r"].mean() if n > 0 else 0

            period = pd.Period(m, freq="M")
            tdays  = trading_days_per_month.get(period, 1)
            apd    = n / tdays if tdays > 0 else 0

            sk_filt = len(ms[ms["reason"] == "FILTER"])     if not ms.empty else 0
            sk_cool = len(ms[ms["reason"] == "COOLDOWN"])   if not ms.empty else 0
            sk_cap  = len(ms[ms["reason"] == "DAILY_CAP"])  if not ms.empty else 0
            sk_tot  = len(ms)

            pf_str  = f"{pf:.2f}" if pf != float("inf") else "  ∞  "
            print(f"  {m:<10} {n:>7} {apd:>8.1f} {w:>6} {l:>7} {wr:>5.0f}% {pf_str:>7} {avg:>+7.3f} | {sk_tot:>6}  {sk_filt:>6}  {sk_cool:>8}  {sk_cap:>6}")

            total_t  += n; total_w += w; total_l += l
            total_wins_r += w_r; total_losses_r += l_r

        pf_tot  = total_wins_r / total_losses_r if total_losses_r > 0 else float("inf")
        wr_tot  = total_w / total_t * 100 if total_t > 0 else 0
        n_months = len(months)
        print(f"  {'─'*95}")
        print(f"  {'TOTAL':<10} {total_t:>7} {total_t/n_months:>8.1f} {total_w:>6} {total_l:>7} {wr_tot:>5.0f}% {pf_tot:>7.2f}")
        print(f"\n  Avg trades/month : {total_t/n_months:.1f}")

    # Exit reason breakdown
    if not trades.empty:
        print(f"\n  Exit reasons:")
        for ex, cnt in trades["exit"].value_counts().items():
            pct = cnt / len(trades) * 100
            avg_r = trades[trades["exit"] == ex]["r"].mean()
            print(f"    {ex:<12} {cnt:>4}  ({pct:.0f}%)   avg R {avg_r:+.3f}")

    # Skip reason summary
    if not skips.empty:
        print(f"\n  Skipped spikes breakdown:")
        for reason, cnt in skips["reason"].value_counts().items():
            print(f"    {reason:<12} {cnt:>5} spikes skipped")


def main():
    print()
    print("=" * 80)
    print("  TRADE FREQUENCY & SIGNAL SELECTIVITY BACKTEST")
    print("  Parameters: production baseline (live bot values)")
    print("=" * 80)
    print(f"\n  SPIKE_MULT={BASE['spike_mult']} | ATR={BASE['atr_period']} | "
          f"HOLD={BASE['hold']}bars | H1_EMA={BASE['h1_ema_period']} | "
          f"MAX_DAY={BASE['max_day']} | COOLDOWN={COOLDOWN}bars")

    print("\n  Loading data...")
    c_m5 = load_raw("CRASH1000", "M5")
    c_h1 = load_raw("CRASH1000", "H1")
    b_m5 = load_raw("BOOM1000",  "M5")
    b_h1 = load_raw("BOOM1000",  "H1")
    print(f"  CRASH1000: {len(c_m5):,} M5 bars | {len(c_h1):,} H1 bars")
    print(f"  BOOM1000 : {len(b_m5):,} M5 bars | {len(b_h1):,} H1 bars")

    # CRASH1000
    print("\n  Running CRASH1000...")
    c_merged = build_merged(c_m5, c_h1)
    c_df     = compute_crash(c_merged)
    c_trades, c_skips = run_frequency(c_df, "BUY")
    report("CRASH1000", "BUY", c_trades, c_skips, c_df)

    # BOOM1000
    print("\n  Running BOOM1000...")
    b_merged = build_merged(b_m5, b_h1)
    b_df     = compute_boom(b_merged)
    b_trades, b_skips = run_frequency(b_df, "SELL")
    report("BOOM1000", "SELL", b_trades, b_skips, b_df)

    # Combined summary
    print(f"\n\n{'='*80}")
    print("  COMBINED SUMMARY")
    print(f"{'='*80}")
    total = len(c_trades) + len(b_trades)
    months_c = c_trades["month"].nunique() if not c_trades.empty else 1
    months_b = b_trades["month"].nunique() if not b_trades.empty else 1
    print(f"  CRASH1000 : {len(c_trades):>4} trades over {months_c} months = {len(c_trades)/months_c:.1f}/month")
    print(f"  BOOM1000  : {len(b_trades):>4} trades over {months_b} months = {len(b_trades)/months_b:.1f}/month")
    print(f"  Combined  : {total:>4} trades")

    if not c_trades.empty:
        cw = c_trades[c_trades["r"] > 0]; cl = c_trades[c_trades["r"] < 0]
        cpf = cw["r"].sum() / abs(cl["r"].sum()) if len(cl) > 0 else 0
        print(f"\n  CRASH1000 PF: {cpf:.3f}  WR: {len(cw)/len(c_trades)*100:.1f}%")
    if not b_trades.empty:
        bw = b_trades[b_trades["r"] > 0]; bl = b_trades[b_trades["r"] < 0]
        bpf = bw["r"].sum() / abs(bl["r"].sum()) if len(bl) > 0 else 0
        print(f"  BOOM1000  PF: {bpf:.3f}  WR: {len(bw)/len(b_trades)*100:.1f}%")

    print()


if __name__ == "__main__":
    main()

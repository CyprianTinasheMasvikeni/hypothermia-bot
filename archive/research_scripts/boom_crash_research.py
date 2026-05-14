"""
Boom/Crash Spike Strategy Research — Jim Simons approach
Phase 1: Characterize the spike (size, frequency, post-spike behavior)
Phase 2: Backtest reversion strategy across all TFs and all pairs
Phase 3: Verdict — where is the edge?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent

# ── Settings ──────────────────────────────────────────────────────────────────
START_BAL        = 10_000.0
RISK_PCT         = 0.05
SL_ATR_MULT      = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
ACCOUNT_DD       = 0.15

# Spike detection thresholds to sweep
SPIKE_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

BOOM_PAIRS  = ["BOOM300N", "BOOM500",  "BOOM1000"]
CRASH_PAIRS = ["CRASH300N","CRASH500", "CRASH900", "CRASH1000"]
ALL_PAIRS   = BOOM_PAIRS + CRASH_PAIRS
TIMEFRAMES  = ["M5", "M15", "H1"]


# ── Data ──────────────────────────────────────────────────────────────────────
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


# ── Spike detection ───────────────────────────────────────────────────────────
def detect_spikes(df, pair, threshold):
    """
    Boom spike:  close - open > threshold * ATR  (sudden up-move)
    Crash spike: open - close > threshold * ATR  (sudden down-move)
    Returns df with 'is_spike' and 'spike_dir' columns.
    """
    d = df.copy()
    is_boom  = pair.startswith("BOOM")
    body     = d["close"] - d["open"]

    if is_boom:
        d["is_spike"]  = body > threshold * d["atr"]
        d["spike_dir"] = "UP"
    else:
        d["is_spike"]  = (-body) > threshold * d["atr"]
        d["spike_dir"] = "DOWN"

    d["body_atr_ratio"] = abs(body) / d["atr"]
    return d


# ── Post-spike characterization ───────────────────────────────────────────────
def characterize_post_spike(df, pair, threshold, hold_candles=20):
    """
    For every spike, measure:
    - Spike magnitude (ATR multiples)
    - Reversion at 1, 5, 10, 20 candles
    - Max reversion within hold_candles
    """
    d          = detect_spikes(df, pair, threshold)
    is_boom    = pair.startswith("BOOM")
    spike_rows = d[d["is_spike"]].index.tolist()

    records = []
    for idx in spike_rows:
        if idx + hold_candles >= len(d):
            continue
        spike_close = d.loc[idx, "close"]
        spike_open  = d.loc[idx, "open"]
        atr         = d.loc[idx, "atr"]
        spike_mag   = abs(spike_close - spike_open) / atr

        fwd = d.loc[idx+1 : idx+hold_candles]
        if len(fwd) < 5:
            continue

        # Reversion = move in opposite direction of spike
        # Boom spike UP -> reversion = (spike_close - future_low) / atr
        # Crash spike DOWN -> reversion = (future_high - spike_close) / atr
        if is_boom:
            rev_at  = [(spike_close - fwd.iloc[:n]["low"].min()) / atr
                       for n in [1, 5, 10, 20] if n <= len(fwd)]
            max_rev = (spike_close - fwd["low"].min()) / atr
            max_ext = (fwd["high"].max() - spike_close) / atr  # continuation
        else:
            rev_at  = [(fwd.iloc[:n]["high"].max() - spike_close) / atr
                       for n in [1, 5, 10, 20] if n <= len(fwd)]
            max_rev = (fwd["high"].max() - spike_close) / atr
            max_ext = (spike_close - fwd["low"].min()) / atr   # continuation

        records.append({
            "spike_mag":  round(spike_mag, 2),
            "rev_1":      round(rev_at[0], 2) if len(rev_at) > 0 else 0,
            "rev_5":      round(rev_at[1], 2) if len(rev_at) > 1 else 0,
            "rev_10":     round(rev_at[2], 2) if len(rev_at) > 2 else 0,
            "max_rev":    round(max_rev, 2),
            "max_ext":    round(max_ext, 2),
        })

    return pd.DataFrame(records)


# ── Trade engine ──────────────────────────────────────────────────────────────
def run_trade(entry, atr, dirn, fwd, balance):
    d    = 1 if dirn == "BUY" else -1
    risk = balance * RISK_PCT
    sl   = entry - d * atr * SL_ATR_MULT
    # Units: risk / (SL distance in price)
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


# ── Backtest one config ───────────────────────────────────────────────────────
def backtest(df, pair, threshold, trade_dir, hold_candles):
    """
    trade_dir: "REVERSION" or "CONTINUATION"
    After a spike, enter in reversion or continuation direction.
    """
    d        = detect_spikes(df, pair, threshold)
    is_boom  = pair.startswith("BOOM")
    balance  = START_BAL
    peak_bal = START_BAL
    mdd      = 0.0
    trades   = []
    traded_idx = set()

    for idx in range(len(d) - hold_candles - 1):
        if idx in traded_idx:
            continue
        row = d.iloc[idx]
        if not row["is_spike"]:
            continue

        atr   = row["atr"]
        entry = float(d.iloc[idx + 1]["open"])
        fwd   = d.iloc[idx + 1 : idx + 1 + hold_candles].copy()
        if len(fwd) < 4:
            continue

        # Direction logic
        if trade_dir == "REVERSION":
            dirn = "SELL" if is_boom else "BUY"
        else:  # CONTINUATION
            dirn = "BUY" if is_boom else "SELL"

        pnl, peak_r, reason = run_trade(entry, atr, dirn, fwd, balance)
        r_val   = pnl / (balance * RISK_PCT) if balance > 0 else 0
        balance += pnl
        peak_bal = max(peak_bal, balance)
        mdd      = max(mdd, (peak_bal - balance) / peak_bal)

        trades.append({
            "date":   str(d.iloc[idx]["time"].date()),
            "pnl":    pnl,
            "r":      round(r_val, 2),
            "peak_r": round(peak_r, 2),
            "result": "WIN" if pnl > 0 else "LOSS",
            "reason": reason,
        })

        # Mark next hold_candles as blocked (no overlapping trades)
        for k in range(idx, idx + hold_candles + 1):
            traded_idx.add(k)

        if mdd >= ACCOUNT_DD:
            break

    if not trades:
        return None

    df_t   = pd.DataFrame(trades)
    total  = len(df_t)
    wins   = (df_t["result"] == "WIN").sum()
    gw     = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gl     = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    pf     = gw / gl if gl > 0 else 0
    avg_r  = df_t["r"].mean()
    ret    = (balance / START_BAL - 1) * 100

    return {
        "trades": total, "wr": wins/total, "pf": pf,
        "avg_r": avg_r, "ret": ret, "mdd": mdd * 100,
        "final_bal": balance,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 72)
    print("  BOOM / CRASH SPIKE RESEARCH  — Jim Simons mode")
    print("  Goal: find the mechanical post-spike edge across all TFs")
    print("=" * 72)

    # ── Phase 1: Characterize spikes ─────────────────────────────────────────
    print()
    print("PHASE 1 — SPIKE CHARACTERIZATION (threshold = 2.0x ATR, H1 data)")
    print(f"  {'Pair':<12} {'Spikes':>7} {'AvgMag':>8} {'Rev@1c':>8} {'Rev@5c':>8} "
          f"{'MaxRev':>8} {'MaxExt':>8}  Verdict")
    print("  " + "-" * 75)

    for pair in ALL_PAIRS:
        df = load(pair, "H1")
        if df is None:
            print(f"  {pair:<12}  no H1 data")
            continue
        df = add_atr(df)
        stats = characterize_post_spike(df, pair, threshold=2.0, hold_candles=24)
        if stats.empty:
            print(f"  {pair:<12}  no spikes detected")
            continue

        avg_mag = stats["spike_mag"].mean()
        rev1    = stats["rev_1"].mean()
        rev5    = stats["rev_5"].mean()
        maxrev  = stats["max_rev"].mean()
        maxext  = stats["max_ext"].mean()
        n       = len(stats)

        # Verdict: reversion > continuation = reversion edge exists
        verdict = "REVERT" if maxrev > maxext else "CONTINUE"
        quality = "STRONG" if maxrev > 1.5 and maxrev > maxext * 1.3 else \
                  "MODERATE" if maxrev > 1.0 else "WEAK"

        print(f"  {pair:<12} {n:>7} {avg_mag:>8.2f} {rev1:>8.2f} {rev5:>8.2f} "
              f"{maxrev:>8.2f} {maxext:>8.2f}  {verdict} [{quality}]")

    # ── Phase 2: Spike frequency across TFs ──────────────────────────────────
    print()
    print("PHASE 2 — SPIKE FREQUENCY & SIZE BY TIMEFRAME (threshold = 2.0x ATR)")
    print(f"  {'Pair':<12} {'TF':<5} {'Total':>6} {'Spikes':>7} {'Freq%':>7} "
          f"{'AvgMag':>8} {'ATR-CV':>8}")
    print("  " + "-" * 60)

    for pair in ALL_PAIRS:
        for tf in TIMEFRAMES:
            df = load(pair, tf)
            if df is None:
                continue
            df = add_atr(df)
            d  = detect_spikes(df, pair, 2.0)
            n_spikes = d["is_spike"].sum()
            freq_pct = n_spikes / len(d) * 100
            avg_mag  = d[d["is_spike"]]["body_atr_ratio"].mean() if n_spikes > 0 else 0
            atr_cv   = d["atr"].std() / d["atr"].mean() if d["atr"].mean() > 0 else 0
            print(f"  {pair:<12} {tf:<5} {len(d):>6} {n_spikes:>7} {freq_pct:>6.1f}% "
                  f"{avg_mag:>8.2f} {atr_cv:>8.3f}")

    # ── Phase 3: Strategy backtest ────────────────────────────────────────────
    print()
    print("PHASE 3 — BACKTEST: REVERSION vs CONTINUATION (best threshold per TF)")
    print("  Using chandelier exit system, 1x ATR SL, 15% account DD kill switch")
    print()

    best_results = []

    for pair in ALL_PAIRS:
        for tf in TIMEFRAMES:
            df = load(pair, tf)
            if df is None:
                continue
            df = add_atr(df)

            hold = {"M5": 24, "M15": 16, "H1": 10}[tf]

            for thresh in [2.0, 2.5, 3.0]:
                for direction in ["REVERSION", "CONTINUATION"]:
                    r = backtest(df, pair, thresh, direction, hold)
                    if r is None:
                        continue
                    if r["trades"] < 5:
                        continue
                    r.update({"pair": pair, "tf": tf,
                               "thresh": thresh, "dir": direction})
                    best_results.append(r)

    # Print all results sorted by PF
    print(f"  {'Pair':<12} {'TF':<5} {'Thr':>4} {'Dir':<12} "
          f"{'Tr':>4} {'WR':>6} {'PF':>6} {'AvgR':>6} {'Ret%':>7} {'MDD%':>6}  Flag")
    print("  " + "-" * 82)

    for r in sorted(best_results, key=lambda x: x["pf"], reverse=True)[:40]:
        flag = ""
        if r["pf"] >= 1.5 and r["trades"] >= 15:
            flag = " <<< STRONG"
        elif r["pf"] >= 1.3 and r["trades"] >= 10:
            flag = " << DECENT"
        elif r["pf"] < 1.0:
            flag = " (losing)"
        print(f"  {r['pair']:<12} {r['tf']:<5} {r['thresh']:>4.1f} {r['dir']:<12} "
              f"{r['trades']:>4} {r['wr']:>5.1%} {r['pf']:>6.2f} "
              f"{r['avg_r']:>+5.2f}R {r['ret']:>+6.1f}% {r['mdd']:>5.1f}%{flag}")

    # ── Phase 4: Top candidates summary ──────────────────────────────────────
    print()
    print("TOP CANDIDATES (PF >= 1.3, trades >= 10):")
    top = [r for r in best_results if r["pf"] >= 1.3 and r["trades"] >= 10]
    top.sort(key=lambda x: (x["pf"], x["trades"]), reverse=True)
    if top:
        for r in top[:8]:
            print(f"  {r['pair']} {r['tf']} thresh={r['thresh']} {r['dir']}: "
                  f"PF={r['pf']:.2f}, WR={r['wr']:.1%}, {r['trades']} trades, "
                  f"ret={r['ret']:+.1f}%, MDD={r['mdd']:.1f}%")
    else:
        print("  None cleared the threshold yet.")

    print()
    print("NEXT STEP: pick the strongest candidate and run a deep dive")
    print("(year-by-year, monthly breakdown, red flag checks)")
    print()


if __name__ == "__main__":
    main()

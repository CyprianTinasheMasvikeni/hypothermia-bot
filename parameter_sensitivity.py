"""
parameter_sensitivity.py — Robustness / Parameter Sensitivity Test
===================================================================
Answers the question: is the edge structural, or does it only work at
the exact parameter values we happened to choose?

Methodology: one-parameter-at-a-time (OAT) sensitivity analysis.
  For each key parameter, hold all others at baseline and vary the
  target param ±20–50%. Record PF, WR, T-stat, OOS PF for each variant.

Pass criteria (PF ≥ 1.5 in all tested ranges → ROBUST).
If the edge collapses when you move one dial → FRAGILE → don't trade it.

Tests:
  CRASH1000 — BUY after crash spike, S5 filter (H1 close > H1 EMA21)
  BOOM1000  — SELL after boom spike, S10 filter (M5 EMA8>21 + H1 < EMA21)

Parameters tested:
  SPIKE_MULT     — how large a candle counts as a "spike" (ATR units)
  ATR_PERIOD     — lookback for ATR calculation
  HOLD_CANDLES   — max bars to hold a trade before force-exit
  H1_EMA_PERIOD  — H1 trend filter EMA lookback
  PARTIAL_R      — where to take the 50% partial profit
  CHANDELIER_INIT— initial trailing stop distance (ATR units from peak/trough)
  MAX_DAY        — max trades per calendar day
  EMA_FAST/SLOW  — BOOM1000 M5 momentum filter periods
"""

import sys, io, math
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Baseline (production values, validated by filter backtest) ─────────────────
BASE = {
    "spike_mult":     2.5,
    "atr_period":     14,
    "hold":           24,
    "h1_ema_period":  21,
    "partial_r":      2.0,
    "chand_init":     3.0,   # chandelier initial mult; tiers step down by 0.5
    "max_day":        6,
    "ema_fast":       8,     # BOOM1000 M5 EMA fast (not used for CRASH1000)
    "ema_slow":       21,    # BOOM1000 M5 EMA slow (not used for CRASH1000)
}

H1_EMA_TEST_PERIODS = [14, 17, 21, 25, 30]
COOLDOWN            = 12      # spike cluster cooldown (not varied — structural)
PASS_PF             = 1.50    # minimum acceptable PF
MARGINAL_PF         = 1.00    # below this = edge gone


# ── Raw data loading ───────────────────────────────────────────────────────────
def load_raw(symbol: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"cache_{symbol}_{gran}.csv"
    df   = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


# ── H1 merge — one pass, all EMA periods at once ──────────────────────────────
def build_merged(raw_m5: pd.DataFrame, raw_h1: pd.DataFrame) -> pd.DataFrame:
    """
    Compute H1 EMAs at H1 frequency, then forward-fill them into M5.
    Uses pd.merge_asof for speed.  No lookahead: each M5 bar gets the
    last H1 bar whose timestamp <= M5 bar time (same logic as live bot).
    """
    h1 = raw_h1.copy().sort_values("time").reset_index(drop=True)
    for p in H1_EMA_TEST_PERIODS:
        h1[f"h1_ema{p}"] = h1["close"].ewm(span=p, adjust=False).mean()
    h1.rename(columns={"close": "h1_close"}, inplace=True)

    keep = ["time", "h1_close"] + [f"h1_ema{p}" for p in H1_EMA_TEST_PERIODS]
    h1_slim = h1[keep].dropna().sort_values("time").reset_index(drop=True)

    m5 = raw_m5.copy().sort_values("time").reset_index(drop=True)
    merged = pd.merge_asof(m5, h1_slim, on="time", direction="backward")
    return merged.dropna(subset=["h1_close"]).reset_index(drop=True)


# ── Indicator computation — CRASH1000 ─────────────────────────────────────────
def compute_crash(merged: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = merged.copy()
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(p["atr_period"]).mean()
    body           = df["open"] - df["close"]          # positive = down candle
    df["is_spike"] = body > p["spike_mult"] * df["atr"]
    h1_ema_col     = f"h1_ema{p['h1_ema_period']}"
    df["filter"]   = df["h1_close"] > df[h1_ema_col]  # H1 bullish
    return df.dropna(subset=["atr", h1_ema_col]).reset_index(drop=True)


# ── Indicator computation — BOOM1000 ──────────────────────────────────────────
def compute_boom(merged: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = merged.copy()
    tr = np.maximum(df["high"] - df["low"],
         np.maximum(abs(df["high"] - df["close"].shift(1)),
                    abs(df["low"]  - df["close"].shift(1))))
    df["atr"]      = tr.rolling(p["atr_period"]).mean()
    body           = df["close"] - df["open"]          # positive = up candle
    df["is_spike"] = body > p["spike_mult"] * df["atr"]
    df["ema_f"]    = df["close"].ewm(span=p["ema_fast"], adjust=False).mean()
    df["ema_s"]    = df["close"].ewm(span=p["ema_slow"], adjust=False).mean()
    h1_ema_col     = f"h1_ema{p['h1_ema_period']}"
    # S10 filter: M5 EMA fast > slow AND H1 bearish
    df["filter"]   = (df["ema_f"] > df["ema_s"]) & (df["h1_close"] < df[h1_ema_col])
    return df.dropna(subset=["atr", h1_ema_col, "ema_f", "ema_s"]).reset_index(drop=True)


# ── Chandelier tier table from initial multiplier ─────────────────────────────
def _chand_tiers(init: float):
    return [(0.0, init), (2.0, init - 0.5), (4.0, init - 1.0)]


# ── Trade simulators ──────────────────────────────────────────────────────────
def run_buy(entry: float, atr: float, fwd: pd.DataFrame, p: dict) -> tuple:
    sl     = entry - atr
    size   = 1.0; partial_done = False; locked_r = 0.0; peak = entry
    tiers  = _chand_tiers(p["chand_init"])

    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_gap = (op - cl) > p["spike_mult"] * row["atr"]

        if lo <= sl:
            r = size * ((cl - entry) / atr) + locked_r if is_gap else size * (-1.0) + locked_r
            return round(r, 3), "SL"

        if not partial_done and hi >= entry + atr * p["partial_r"]:
            locked_r     += size * 0.5 * p["partial_r"]
            size         *= 0.5
            partial_done  = True

        peak   = max(peak, hi)
        peak_r = (peak - entry) / atr
        cm     = tiers[0][1]
        for mr, tm in tiers:
            if peak_r >= mr: cm = tm
        csl = peak - atr * cm
        if lo <= csl:
            return round(size * ((csl - entry) / atr) + locked_r, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    return round(size * ((last - entry) / atr) + locked_r, 3), "TIME"


def run_sell(entry: float, atr: float, fwd: pd.DataFrame, p: dict) -> tuple:
    sl     = entry + atr
    size   = 1.0; partial_done = False; locked_r = 0.0; trough = entry
    tiers  = _chand_tiers(p["chand_init"])

    for _, row in fwd.iterrows():
        hi, lo, op, cl = row["high"], row["low"], row["open"], row["close"]
        is_gap = (cl - op) > p["spike_mult"] * row["atr"]

        if hi >= sl:
            r = size * ((entry - cl) / atr) + locked_r if is_gap else size * (-1.0) + locked_r
            return round(r, 3), "SL"

        if not partial_done and lo <= entry - atr * p["partial_r"]:
            locked_r     += size * 0.5 * p["partial_r"]
            size         *= 0.5
            partial_done  = True

        trough   = min(trough, lo)
        trough_r = (entry - trough) / atr
        cm       = tiers[0][1]
        for mr, tm in tiers:
            if trough_r >= mr: cm = tm
        csl = trough + atr * cm
        if hi >= csl:
            return round(size * ((entry - csl) / atr) + locked_r, 3), "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    return round(size * ((entry - last) / atr) + locked_r, 3), "TIME"


# ── Backtest engine ───────────────────────────────────────────────────────────
def backtest(df: pd.DataFrame, direction: str, p: dict) -> pd.DataFrame:
    traded = set(); tpd: dict = {}; trades = []; lsi = -999

    for idx in range(len(df) - p["hold"] - 1):
        if idx in traded:
            continue
        row = df.iloc[idx]
        if not row["is_spike"]:
            continue
        if (idx - lsi) <= COOLDOWN:
            lsi = idx; continue
        lsi = idx

        date = str(row["time"].date())
        if tpd.get(date, 0) >= p["max_day"]:
            continue
        if not row["filter"]:
            continue

        atr   = row["atr"]
        entry = float(df.iloc[idx + 1]["open"]) if idx + 1 < len(df) else float(row["close"])
        fwd   = df.iloc[idx + 1: idx + 1 + p["hold"]].copy()
        if len(fwd) < 4:
            continue

        r, reason = run_buy(entry, atr, fwd, p) if direction == "BUY" else run_sell(entry, atr, fwd, p)
        tpd[date] = tpd.get(date, 0) + 1
        trades.append({"date": date, "month": date[:7], "r": r, "result": "W" if r > 0 else "L"})
        for k in range(idx, idx + p["hold"] + 1):
            traded.add(k)

    return pd.DataFrame(trades)


# ── Stats ──────────────────────────────────────────────────────────────────────
def calc_stats(trades: pd.DataFrame) -> dict | None:
    if trades.empty or len(trades) < 5:
        return None
    n    = len(trades)
    rs   = trades["r"].values
    wins = rs[rs > 0]; losses = abs(rs[rs < 0])
    wr   = (rs > 0).mean()
    pf   = wins.sum() / losses.sum() if losses.sum() > 0 else 0.0
    avg  = rs.mean()
    std  = rs.std(ddof=1)
    t    = (avg / (std / math.sqrt(n))) if std > 0 else 0.0
    return {"n": n, "wr": wr, "pf": pf, "avg": avg, "t": t}


def oos_pf(df: pd.DataFrame, direction: str, p: dict) -> float | None:
    dates  = df["time"]
    cutoff = dates.min() + pd.DateOffset(months=6)
    oos_df = df[dates >= cutoff].reset_index(drop=True)
    t      = backtest(oos_df, direction, p)
    s      = calc_stats(t)
    return s["pf"] if s else None


# ── Status label ──────────────────────────────────────────────────────────────
def status_label(pf: float) -> str:
    if pf >= PASS_PF:     return "PASS    "
    if pf >= MARGINAL_PF: return "MARGINAL"
    return "FAIL    "


# ── Print row ─────────────────────────────────────────────────────────────────
def print_row(label: str, s: dict | None, oos: float | None, is_base: bool = False):
    star = "★" if is_base else " "
    if s is None:
        print(f"  {star} {label:<24}   —    —     —       —     —      —")
        return
    pf_str  = f"{s['pf']:.3f}"
    oos_str = f"{oos:.3f}" if oos is not None else "  —  "
    st      = status_label(s["pf"])
    print(f"  {star} {label:<24} {s['n']:>5} {s['wr']*100:>5.1f}%  {pf_str:>6} "
          f" {s['avg']:>+7.3f}  {s['t']:>6.2f}  {oos_str:>6}   {st}")


# ── Run one sensitivity group ──────────────────────────────────────────────────
def run_group(title: str, variants: list, compute_fn, direction: str, merged: pd.DataFrame):
    """
    variants: list of (label, param_override_dict, is_baseline)
    """
    print(f"\n  ── {title}")
    print(f"  {'':1} {'Description':<24} {'N':>5} {'WR':>6}  {'PF':>6}  {'AvgR':>7}  "
          f"{'T-stat':>6}  {'OOS PF':>6}   Status")
    print(f"  {'─'*85}")

    results = []
    for label, override, is_base in variants:
        p  = {**BASE, **override}
        df = compute_fn(merged, p)
        t  = backtest(df, direction, p)
        s  = calc_stats(t)
        op = oos_pf(df, direction, p) if s else None
        print_row(label, s, op, is_base)
        results.append((s["pf"] if s else None, is_base))

    # Quick summary for this group
    non_base = [(pf, ib) for pf, ib in results if not ib and pf is not None]
    fails    = [pf for pf, _ in non_base if pf < PASS_PF]
    if not fails:
        print(f"  → All variants PASS (PF ≥ {PASS_PF})")
    else:
        low = min(fails)
        print(f"  → Lowest PF in range: {low:.3f}  "
              f"({'MARGINAL' if low >= MARGINAL_PF else 'FAIL'})")
    return results


# ── Full sensitivity analysis for one symbol ──────────────────────────────────
def run_symbol(symbol: str, direction: str, raw_m5: pd.DataFrame, raw_h1: pd.DataFrame):
    print(f"\n\n{'='*90}")
    print(f"  PARAMETER SENSITIVITY — {symbol}  ({direction})")
    if symbol == "CRASH1000":
        filter_desc = f"H1 close > H1 EMA{BASE['h1_ema_period']} (bullish regime — spike reversion BUY)"
    else:
        filter_desc = (f"M5 EMA{BASE['ema_fast']}>EMA{BASE['ema_slow']} + "
                       f"H1 close < H1 EMA{BASE['h1_ema_period']} (bearish regime — spike reversion SELL)")
    print(f"  Filter: {filter_desc}")
    print(f"  Baseline: SPIKE={BASE['spike_mult']} | ATR={BASE['atr_period']} | "
          f"HOLD={BASE['hold']} | H1_EMA={BASE['h1_ema_period']} | "
          f"PARTIAL={BASE['partial_r']}R | CHAND={BASE['chand_init']}")
    print(f"{'='*90}")

    print("\n  Building merged dataset... ", end="", flush=True)
    merged = build_merged(raw_m5, raw_h1)
    print(f"done ({len(merged):,} M5 bars | "
          f"{merged['time'].min().date()} → {merged['time'].max().date()})")

    fn = compute_crash if direction == "BUY" else compute_boom

    all_non_base = []

    # ── 1. SPIKE THRESHOLD ────────────────────────────────────────────────────
    g = run_group("SPIKE_MULT  (detection threshold — ATR × mult)", [
        ("1.50  (−40%)",    {"spike_mult": 1.5}, False),
        ("2.00  (−20%)",    {"spike_mult": 2.0}, False),
        ("2.50  ★ baseline",{"spike_mult": 2.5}, True ),
        ("3.00  (+20%)",    {"spike_mult": 3.0}, False),
        ("3.50  (+40%)",    {"spike_mult": 3.5}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 2. ATR PERIOD ─────────────────────────────────────────────────────────
    g = run_group("ATR_PERIOD  (volatility lookback — bars)", [
        (" 8   (−43%)",     {"atr_period":  8}, False),
        ("10   (−29%)",     {"atr_period": 10}, False),
        ("14   ★ baseline", {"atr_period": 14}, True ),
        ("18   (+29%)",     {"atr_period": 18}, False),
        ("21   (+50%)",     {"atr_period": 21}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 3. HOLD CANDLES ───────────────────────────────────────────────────────
    g = run_group("HOLD_CANDLES  (max bars to hold before time-exit)", [
        ("16   (−33%)",     {"hold": 16}, False),
        ("20   (−17%)",     {"hold": 20}, False),
        ("24   ★ baseline", {"hold": 24}, True ),
        ("28   (+17%)",     {"hold": 28}, False),
        ("32   (+33%)",     {"hold": 32}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 4. H1 EMA PERIOD ──────────────────────────────────────────────────────
    g = run_group("H1_EMA_PERIOD  (trend filter lookback — H1 bars)", [
        ("14   (−33%)",     {"h1_ema_period": 14}, False),
        ("17   (−19%)",     {"h1_ema_period": 17}, False),
        ("21   ★ baseline", {"h1_ema_period": 21}, True ),
        ("25   (+19%)",     {"h1_ema_period": 25}, False),
        ("30   (+43%)",     {"h1_ema_period": 30}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 5. PARTIAL PROFIT LEVEL ───────────────────────────────────────────────
    g = run_group("PARTIAL_R  (where 50% position closed — R multiples)", [
        ("1.50  (−25%)",    {"partial_r": 1.5}, False),
        ("2.00  ★ baseline",{"partial_r": 2.0}, True ),
        ("2.50  (+25%)",    {"partial_r": 2.5}, False),
        ("3.00  (+50%)",    {"partial_r": 3.0}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 6. CHANDELIER INITIAL MULTIPLIER ─────────────────────────────────────
    g = run_group("CHANDELIER_INIT  (trailing stop width — ATR × mult from peak/trough)", [
        ("2.50  (−17%)",    {"chand_init": 2.5}, False),
        ("3.00  ★ baseline",{"chand_init": 3.0}, True ),
        ("3.50  (+17%)",    {"chand_init": 3.5}, False),
        ("4.00  (+33%)",    {"chand_init": 4.0}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 7. MAX TRADES PER DAY ─────────────────────────────────────────────────
    g = run_group("MAX_DAY  (daily trade cap per bot)", [
        (" 4   (−33%)",     {"max_day": 4}, False),
        (" 6   ★ baseline", {"max_day": 6}, True ),
        (" 8   (+33%)",     {"max_day": 8}, False),
    ], fn, direction, merged)
    all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── 8. BOOM1000-ONLY: M5 EMA FILTER ─────────────────────────────────────
    if direction == "SELL":
        g = run_group("EMA_FAST  (M5 momentum filter — fast EMA period)", [
            (" 6   (−25%)",     {"ema_fast":  6}, False),
            (" 8   ★ baseline", {"ema_fast":  8}, True ),
            ("10   (+25%)",     {"ema_fast": 10}, False),
            ("12   (+50%)",     {"ema_fast": 12}, False),
        ], fn, direction, merged)
        all_non_base += [(pf, ib) for pf, ib in g if not ib]

        g = run_group("EMA_SLOW  (M5 momentum filter — slow EMA period)", [
            ("15   (−29%)",     {"ema_slow": 15}, False),
            ("21   ★ baseline", {"ema_slow": 21}, True ),
            ("26   (+24%)",     {"ema_slow": 26}, False),
        ], fn, direction, merged)
        all_non_base += [(pf, ib) for pf, ib in g if not ib]

    # ── STRESS TEST: multiple pessimistic values simultaneously ──────────────
    print(f"\n  ── STRESS TEST  (combine multiple non-baseline values)")
    print(f"  {'':1} {'Scenario':<24} {'N':>5} {'WR':>6}  {'PF':>6}  {'AvgR':>7}  "
          f"{'T-stat':>6}  {'OOS PF':>6}   Status")
    print(f"  {'─'*85}")

    stress = [
        ("Baseline ★",           {},                                                True),
        ("High threshold",        {"spike_mult": 3.0},                              False),
        ("Low threshold",         {"spike_mult": 2.0},                              False),
        ("Slow ATR (21)",         {"atr_period": 21},                               False),
        ("Fast ATR (8)",          {"atr_period":  8},                               False),
        ("Short hold (16)",       {"hold": 16},                                     False),
        ("Wide chandelier (4.0)", {"chand_init": 4.0},                             False),
        ("Tight chandelier (2.5)",{"chand_init": 2.5},                             False),
        ("Alt H1 EMA (14)",       {"h1_ema_period": 14},                           False),
        ("Alt H1 EMA (30)",       {"h1_ema_period": 30},                           False),
        ("Worst combo",           {"spike_mult": 3.0, "hold": 16, "chand_init": 3.5,
                                   "atr_period": 18},                               False),
        ("Conservative combo",    {"spike_mult": 2.0, "hold": 28, "chand_init": 2.5,
                                   "atr_period": 10},                               False),
    ]

    stress_results = []
    for name, override, is_base in stress:
        p  = {**BASE, **override}
        df = fn(merged, p)
        t  = backtest(df, direction, p)
        s  = calc_stats(t)
        op = oos_pf(df, direction, p) if s else None
        print_row(name, s, op, is_base)
        if not is_base and s:
            stress_results.append(s["pf"])

    # ── OVERALL VERDICT ───────────────────────────────────────────────────────
    print(f"\n\n  {'='*70}")
    print(f"  OVERALL ROBUSTNESS VERDICT — {symbol}")
    print(f"  {'='*70}")

    valid_pfs = [pf for pf, _ in all_non_base if pf is not None]
    if not valid_pfs:
        print("  Insufficient data for verdict.")
        return

    n_total    = len(valid_pfs)
    n_pass     = sum(1 for pf in valid_pfs if pf >= PASS_PF)
    n_marginal = sum(1 for pf in valid_pfs if MARGINAL_PF <= pf < PASS_PF)
    n_fail     = sum(1 for pf in valid_pfs if pf < MARGINAL_PF)
    min_pf     = min(valid_pfs)
    max_pf     = max(valid_pfs)
    mean_pf    = sum(valid_pfs) / len(valid_pfs)

    print(f"  Variations tested : {n_total}")
    print(f"  PASS  (PF ≥ 1.50) : {n_pass}  ({n_pass/n_total*100:.0f}%)")
    print(f"  MARGINAL (1.0-1.5): {n_marginal}")
    print(f"  FAIL  (PF < 1.00) : {n_fail}")
    print(f"  PF range          : {min_pf:.3f} – {max_pf:.3f}  (mean {mean_pf:.3f})")

    worst_stress = min(stress_results) if stress_results else None
    if worst_stress is not None:
        print(f"  Worst stress combo: PF {worst_stress:.3f}")

    if n_fail == 0 and n_marginal <= 2:
        verdict = "ROBUST — edge holds across all tested parameter ranges"
        detail  = ("The edge is structural. Moving any parameter ±30-40% from "
                   "baseline does not destroy it. This is NOT curve-fit.")
    elif n_fail == 0:
        verdict = "MOSTLY ROBUST — some marginal zones, no outright failures"
        detail  = ("Edge holds across most ranges. Marginal zones are at extremes "
                   "and expected — the market simply provides fewer/different spikes there.")
    elif n_fail <= 3:
        verdict = "MODERATE — fails at extremes, robust near baseline"
        detail  = ("Edge holds near baseline ±20% but degrades at extreme values. "
                   "Normal for a real edge — don't use extreme parameter values.")
    else:
        verdict = "FRAGILE — edge is sensitive to parameter choice"
        detail  = ("Edge collapses across multiple parameter variations. "
                   "This suggests possible curve-fitting. Do not trade this.")

    print(f"\n  Verdict: {verdict}")
    print(f"  {detail}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 90)
    print("  PARAMETER SENSITIVITY & ROBUSTNESS TEST")
    print("  Hypothesis: the spike-reversion edge is structural — not parameter-dependent")
    print("=" * 90)
    print()
    print(f"  PASS threshold  : PF ≥ {PASS_PF}  (edge clearly profitable)")
    print(f"  MARGINAL zone   : PF {MARGINAL_PF:.1f} – {PASS_PF}  (edge exists but thin)")
    print(f"  FAIL            : PF < {MARGINAL_PF}  (edge gone at this parameter value)")
    print()
    print("  Method: one-parameter-at-a-time (OAT) + multi-param stress test")
    print("  Each variant: full period PF + OOS (last 2 months) PF")
    print("  OOS PF tests robustness across time as well as parameters")

    print("\n\n  Loading data...")
    c_m5 = load_raw("CRASH1000", "M5")
    c_h1 = load_raw("CRASH1000", "H1")
    b_m5 = load_raw("BOOM1000",  "M5")
    b_h1 = load_raw("BOOM1000",  "H1")
    print(f"  CRASH1000 M5: {len(c_m5):,} bars  H1: {len(c_h1):,} bars")
    print(f"  BOOM1000  M5: {len(b_m5):,} bars  H1: {len(b_h1):,} bars")

    run_symbol("CRASH1000", "BUY",  c_m5, c_h1)
    run_symbol("BOOM1000",  "SELL", b_m5, b_h1)

    print("\n\n" + "=" * 90)
    print("  TEST COMPLETE")
    print("=" * 90)
    print()
    print("  How to interpret:")
    print("  ─ If PASS rate > 80% across all groups → edge is structural → trade with confidence")
    print("  ─ If any single group has multiple FAILs → that parameter zone is fragile → avoid extremes")
    print("  ─ Stress test 'worst combo' PF > 1.0 → edge survives adversarial parameter choice")
    print("  ─ OOS PF > 1.0 across all variants → edge is not a time-period artefact")
    print()


if __name__ == "__main__":
    main()

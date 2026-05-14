#!/usr/bin/env python3
"""
Walk-Forward Validation Framework — Deriv Synthetic Indices
============================================================
Proper out-of-sample testing to confirm time-of-day edges are real.

Method:
  1. Split 1-year data into N folds (rolling windows)
  2. In each fold: use first 75% (in-sample) to DISCOVER best hours
  3. Then test those hours on the remaining 25% (out-of-sample, never seen)
  4. If edge holds out-of-sample across ALL folds → it's real
  5. If it only works in-sample → it's overfitting, discard it

This is the standard quant fund approach to avoid curve-fitting.

Run: python walk_forward_validation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

CACHE_PATH   = "quant_cache/raw_data_1yr.pkl"
SYMBOLS      = ["stpRNG", "R_10", "R_25", "R_50", "R_75", "R_100"]
N_FOLDS      = 4       # number of walk-forward folds
IN_SAMPLE    = 0.75    # 75% in-sample, 25% out-of-sample per fold
MIN_OBS      = 20      # min observations per hour bucket
TOP_N_HOURS  = 3       # how many best hours to pick from in-sample
PVAL_THRESH  = 0.05

# ── Helpers ───────────────────────────────────────────────────────────────────
def log_returns(df):
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
    return r


def hour_stats(returns, hours):
    """Return mean return per hour, t-stat, p-value."""
    rows = []
    for h in range(24):
        r = returns[hours == h].dropna()
        if len(r) < MIN_OBS:
            continue
        mean  = r.mean()
        std   = r.std()
        n     = len(r)
        tstat = mean / (std / np.sqrt(n)) if std > 0 else 0
        pval  = 2 * stats.t.sf(abs(tstat), df=n - 1)
        rows.append({"hour": h, "mean": mean, "n": n, "tstat": tstat, "pval": pval})
    return pd.DataFrame(rows).set_index("hour") if rows else pd.DataFrame()


def pick_best_hours(df_insample, top_n=TOP_N_HOURS):
    """Find statistically best hours from in-sample data."""
    ret   = log_returns(df_insample)
    hours = pd.Series(df_insample.index.hour[1:], index=ret.index)
    stats_df = hour_stats(ret, hours)
    if stats_df.empty:
        return [], []
    # Best = positive mean, sorted by t-stat
    pos = stats_df[stats_df["mean"] > 0].sort_values("tstat", ascending=False)
    neg = stats_df[stats_df["mean"] < 0].sort_values("tstat", ascending=True)
    best  = pos.head(top_n).index.tolist()
    worst = neg.head(top_n).index.tolist()
    return best, worst


def evaluate_hours(df_oos, hours_to_use, strategy="best"):
    """
    Out-of-sample evaluation.
    Returns: (mean_ret_filtered, mean_ret_baseline, win_rate, n_obs, t_stat, p_value)
    """
    ret        = log_returns(df_oos)
    hour_index = pd.Series(df_oos.index.hour[1:], index=ret.index)

    mask_filtered = hour_index.isin(hours_to_use)
    r_filtered    = ret[mask_filtered]
    r_baseline    = ret[~mask_filtered]

    if len(r_filtered) < MIN_OBS:
        return None

    mean_f = r_filtered.mean()
    mean_b = r_baseline.mean()
    wr     = (r_filtered > 0).mean()
    n      = len(r_filtered)
    std    = r_filtered.std()
    tstat  = mean_f / (std / np.sqrt(n)) if std > 0 else 0
    pval   = 2 * stats.t.sf(abs(tstat), df=n - 1)

    return {
        "mean_filtered":  mean_f,
        "mean_baseline":  mean_b,
        "edge":           mean_f - mean_b,
        "win_rate":       wr,
        "n_obs":          n,
        "t_stat":         tstat,
        "p_value":        pval,
        "significant":    pval < PVAL_THRESH,
    }


# ── Walk-Forward Engine ───────────────────────────────────────────────────────
def walk_forward(symbol, df):
    n      = len(df)
    fold_size = n // N_FOLDS
    results = []

    for fold in range(N_FOLDS):
        start = fold * fold_size
        end   = start + fold_size if fold < N_FOLDS - 1 else n

        df_fold  = df.iloc[start:end]
        split    = int(len(df_fold) * IN_SAMPLE)
        df_in    = df_fold.iloc[:split]
        df_out   = df_fold.iloc[split:]

        if len(df_in) < 200 or len(df_out) < 50:
            continue

        date_in_start  = df_in.index[0].strftime("%Y-%m-%d")
        date_in_end    = df_in.index[-1].strftime("%Y-%m-%d")
        date_out_start = df_out.index[0].strftime("%Y-%m-%d")
        date_out_end   = df_out.index[-1].strftime("%Y-%m-%d")

        # Discover best/worst hours on in-sample
        best_hours, worst_hours = pick_best_hours(df_in, top_n=TOP_N_HOURS)

        # Evaluate on out-of-sample
        res_best  = evaluate_hours(df_out, best_hours,  "best")  if best_hours  else None
        res_worst = evaluate_hours(df_out, worst_hours, "worst") if worst_hours else None

        results.append({
            "fold":          fold + 1,
            "in_sample":     f"{date_in_start} to {date_in_end}",
            "out_sample":    f"{date_out_start} to {date_out_end}",
            "best_hours_is": best_hours,
            "worst_hours_is":worst_hours,
            "best_oos":      res_best,
            "worst_oos":     res_worst,
            "df_in":         df_in,
            "df_out":        df_out,
        })

    return results


def print_walk_forward(symbol, df, results):
    print(f"\n{'='*70}")
    print(f"  {symbol} — WALK-FORWARD VALIDATION  ({len(df):,} H1 bars, {N_FOLDS} folds)")
    print(f"{'='*70}")

    best_oos_all  = []
    worst_oos_all = []

    for r in results:
        print(f"\n  Fold {r['fold']}")
        print(f"    In-sample:      {r['in_sample']}")
        print(f"    Out-of-sample:  {r['out_sample']}")

        # In-sample rediscovery
        df_in = r["df_in"]
        ret_in = log_returns(df_in)
        hours_in = pd.Series(df_in.index.hour[1:], index=ret_in.index)
        stats_in = hour_stats(ret_in, hours_in)

        print(f"\n    In-sample best hours discovered: {r['best_hours_is']}")
        if not stats_in.empty and r["best_hours_is"]:
            for h in r["best_hours_is"]:
                if h in stats_in.index:
                    row = stats_in.loc[h]
                    sig = "***" if row["pval"] < 0.01 else ("**" if row["pval"] < 0.05 else "ns")
                    print(f"      Hour {h:02d}UTC: mean={row['mean']:+.6f} | WR={(ret_in[hours_in==h]>0).mean():.1%} | t={row['tstat']:+.2f} {sig}")

        print(f"\n    Out-of-sample results for those same hours:")

        if r["best_oos"]:
            oos = r["best_oos"]
            sig = "*** HOLDS" if oos["p_value"] < 0.01 else ("** HOLDS" if oos["p_value"] < 0.05 else "ns — FAILED")
            print(f"      Best hours OOS: mean={oos['mean_filtered']:+.6f} | baseline={oos['mean_baseline']:+.6f} | "
                  f"edge={oos['edge']:+.6f} | WR={oos['win_rate']:.1%} | t={oos['t_stat']:+.2f} | p={oos['p_value']:.4f}  [{sig}]")
            best_oos_all.append(oos)
        else:
            print(f"      Best hours OOS: insufficient data")

        if r["worst_oos"] and r["worst_hours_is"]:
            oos = r["worst_oos"]
            # Worst hours should have negative mean OOS to confirm
            confirmed = oos["mean_filtered"] < 0
            print(f"      Worst hours OOS: mean={oos['mean_filtered']:+.6f} | "
                  f"edge={oos['edge']:+.6f} | WR={oos['win_rate']:.1%} | t={oos['t_stat']:+.2f}  "
                  f"[{'CONFIRMED BAD' if confirmed else 'reversed — not reliable'}]")
            worst_oos_all.append(oos)

    # ── Summary across all folds ──────────────────────────────────────────────
    print(f"\n  {'-'*68}")
    print(f"  SUMMARY ACROSS ALL {len(results)} FOLDS")
    print(f"  {'-'*68}")

    if best_oos_all:
        means  = [x["mean_filtered"]  for x in best_oos_all]
        edges  = [x["edge"]           for x in best_oos_all]
        wrs    = [x["win_rate"]        for x in best_oos_all]
        sigs   = [x["significant"]     for x in best_oos_all]

        n_sig  = sum(sigs)
        consistency = n_sig / len(sigs)

        print(f"\n  Best-hours OOS performance across folds:")
        print(f"    Mean return per hour:  {np.mean(means):+.6f}  (avg across folds)")
        print(f"    Edge vs baseline:      {np.mean(edges):+.6f}  (avg advantage)")
        print(f"    Win rate:              {np.mean(wrs):.1%}  (avg across folds)")
        print(f"    Folds significant:     {n_sig}/{len(sigs)} = {consistency:.0%}")

        if consistency >= 0.75 and np.mean(means) > 0:
            print(f"\n  VERDICT: EDGE IS REAL")
            print(f"    Holds out-of-sample in {consistency:.0%} of folds.")
            print(f"    Safe to build strategy around these hours.")
        elif consistency >= 0.50:
            print(f"\n  VERDICT: WEAK EDGE — USE WITH CAUTION")
            print(f"    Holds in only {n_sig}/{len(sigs)} folds. Not fully robust.")
            print(f"    Combine with other filters before trading.")
        else:
            print(f"\n  VERDICT: NO REAL EDGE — OVERFITTING")
            print(f"    Only holds in {n_sig}/{len(sigs)} folds.")
            print(f"    The in-sample discovery was curve-fitting. Do not trade.")

    # ── Combined return distribution test ─────────────────────────────────────
    print(f"\n  Full-data confirmation (all 1 year, split 50/50):")
    split     = len(df) // 2
    df_first  = df.iloc[:split]
    df_second = df.iloc[split:]

    best_first, _  = pick_best_hours(df_first)
    res_second     = evaluate_hours(df_second, best_first) if best_first else None

    if res_second:
        sig = "SIGNIFICANT" if res_second["p_value"] < PVAL_THRESH else "not significant"
        print(f"    Hours discovered on first 6 months, tested on last 6 months:")
        print(f"    Best hours: {best_first}")
        print(f"    OOS mean={res_second['mean_filtered']:+.6f} | edge={res_second['edge']:+.6f} | "
              f"WR={res_second['win_rate']:.1%} | t={res_second['t_stat']:+.2f} | p={res_second['p_value']:.4f}  [{sig}]")


# ── Momentum Signal Validation ────────────────────────────────────────────────
def validate_momentum(symbol, df):
    """Walk-forward test for lagged return momentum (sign_lag1, sign_lag5)."""
    print(f"\n  {'-'*68}")
    print(f"  {symbol} -- MOMENTUM SIGNAL VALIDATION")
    print(f"  {'-'*68}")

    fold_size = len(df) // N_FOLDS
    rows = []

    for lag in [1, 3, 5, 10]:
        fold_results = []
        for fold in range(N_FOLDS):
            start    = fold * fold_size
            end      = start + fold_size if fold < N_FOLDS - 1 else len(df)
            df_fold  = df.iloc[start:end]
            split    = int(len(df_fold) * IN_SAMPLE)
            df_in    = df_fold.iloc[:split]
            df_out   = df_fold.iloc[split:]

            if len(df_in) < 200 or len(df_out) < 50:
                continue

            # In-sample IC for lag
            ret_in  = log_returns(df_in)
            lag_in  = np.sign(ret_in.shift(lag)).dropna()
            aligned = ret_in.reindex(lag_in.index).dropna()
            lag_in  = lag_in.reindex(aligned.index)
            if len(aligned) < MIN_OBS:
                continue
            ic_in, _ = stats.spearmanr(lag_in, aligned)

            # Out-of-sample IC for same lag
            ret_out  = log_returns(df_out)
            lag_out  = np.sign(ret_out.shift(lag)).dropna()
            aligned2 = ret_out.reindex(lag_out.index).dropna()
            lag_out  = lag_out.reindex(aligned2.index)
            if len(aligned2) < MIN_OBS:
                continue
            ic_out, pval_out = stats.spearmanr(lag_out, aligned2)

            fold_results.append({"ic_in": ic_in, "ic_out": ic_out, "pval_out": pval_out})

        if not fold_results:
            continue

        ic_in_mean  = np.mean([x["ic_in"]  for x in fold_results])
        ic_out_mean = np.mean([x["ic_out"] for x in fold_results])
        n_sig       = sum(1 for x in fold_results if x["pval_out"] < PVAL_THRESH and x["ic_out"] > 0)
        decay       = (ic_out_mean - ic_in_mean) / abs(ic_in_mean) if ic_in_mean != 0 else 0

        rows.append({
            "lag":          lag,
            "ic_insample":  ic_in_mean,
            "ic_outsample": ic_out_mean,
            "decay":        decay,
            "folds_sig":    n_sig,
        })

    if rows:
        df_r = pd.DataFrame(rows)
        print(f"\n  {'Lag':>5} {'IC in-sample':>14} {'IC out-sample':>14} {'Decay':>8} {'Folds sig':>10}  Verdict")
        print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*8} {'-'*10}  {'-'*20}")
        for _, row in df_r.iterrows():
            verdict = "REAL" if row["folds_sig"] >= N_FOLDS//2 and row["ic_outsample"] > 0.01 else "noise"
            print(f"  {int(row['lag']):>5} {row['ic_insample']:>+14.4f} {row['ic_outsample']:>+14.4f} "
                  f"{row['decay']:>+7.1%} {row['folds_sig']:>10}/{N_FOLDS}  {verdict}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION — TIME-OF-DAY & MOMENTUM SIGNALS")
    print(f"  {N_FOLDS} folds | {IN_SAMPLE:.0%} in-sample / {1-IN_SAMPLE:.0%} out-of-sample per fold")
    print(f"  Top {TOP_N_HOURS} hours selected per fold")
    print("=" * 70)

    if not os.path.exists(CACHE_PATH):
        print(f"\n[!] Cache not found. Run quant_research.py first.")
        return

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    all_verdicts = {}

    for symbol in SYMBOLS:
        df_h1 = data.get(symbol, {}).get("H1")
        if df_h1 is None or len(df_h1) < 500:
            print(f"\n[!] {symbol}: insufficient data, skipping")
            continue

        results = walk_forward(symbol, df_h1)
        print_walk_forward(symbol, df_h1, results)

        # Collect verdict
        if results:
            best_oos = [r["best_oos"] for r in results if r["best_oos"]]
            if best_oos:
                n_sig = sum(1 for x in best_oos if x["significant"])
                mean_edge = np.mean([x["edge"] for x in best_oos])
                all_verdicts[symbol] = {
                    "folds_sig":  n_sig,
                    "total_folds":len(best_oos),
                    "mean_edge":  mean_edge,
                    "real":       n_sig >= len(best_oos) * 0.75 and mean_edge > 0,
                }

        # Validate momentum signals too
        validate_momentum(symbol, df_h1)

    # ── Final cross-symbol verdict ────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  FINAL VERDICT — WHICH SIGNALS ARE REAL")
    print(f"{'='*70}\n")
    print(f"  {'Symbol':<10} {'Time Edge':<12} {'Folds OK':<10} {'Mean Edge':>12}  Decision")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12}  {'-'*30}")
    for sym, v in all_verdicts.items():
        verdict  = "BUILD STRATEGY" if v["real"] else ("USE WITH CARE" if v["folds_sig"] >= 1 else "DISCARD")
        folds_ok = f"{v['folds_sig']}/{v['total_folds']}"
        print(f"  {sym:<10} {'time-of-day':<12} {folds_ok:<10} {v['mean_edge']:>+12.6f}  {verdict}")

    print(f"\n  Signals marked BUILD STRATEGY have real out-of-sample edge.")
    print(f"  These are the pairs and signals we build bots on next.")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quantitative Research Pipeline — Deriv Synthetic Indices
=========================================================
Pulls historical candle data for all symbols + timeframes,
engineers 50+ statistical features, tests each for predictive
power using Information Coefficient (IC), and ranks signals
by real statistical edge.

Run: python quant_research.py
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from scipy import stats
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
APP_ID      = 1089
DERIV_WS    = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
SYMBOLS     = ["stpRNG", "R_10", "R_25", "R_50", "R_75", "R_100"]
TIMEFRAMES  = {"M5": 300, "M15": 900, "H1": 3600}   # M1 excluded — too many requests
TARGET_DAYS = 365        # pull 1 full year per symbol/timeframe
BATCH_SIZE  = 5000       # candles per API request
CACHE_DIR   = "quant_cache"
MIN_IC      = 0.02
MAX_PVAL    = 0.05
MIN_OBS     = 500        # raised — more data means we can demand more obs

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Data Collection ───────────────────────────────────────────────────────────
async def fetch_candles_paginated(ws, symbol: str, granularity: int, target_days: int = TARGET_DAYS) -> pd.DataFrame:
    target_bars = target_days * 24 * 3600 // granularity
    batches     = []   # newest first while collecting, reversed at end
    end_time    = "latest"
    batch_num   = 0

    while True:
        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": BATCH_SIZE,
            "end": end_time,
            "granularity": granularity,
            "style": "candles",
        }
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())

        if "error" in resp:
            print(f" [API error: {resp['error']['message']}]", end="")
            break

        candles = resp.get("candles", [])
        if not candles:
            break

        batches.append(candles)
        batch_num += 1
        total = sum(len(b) for b in batches)
        print(f"\r  {symbol} {granularity}s  — batch {batch_num} | {total:,} bars pulled...", end="", flush=True)

        if total >= target_bars or len(candles) < BATCH_SIZE:
            break

        end_time = candles[0]["epoch"] - 1   # step back before oldest candle
        await asyncio.sleep(0.4)

    # Reverse so data is chronological (oldest → newest)
    all_candles = []
    for batch in reversed(batches):
        all_candles.extend(batch)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["epoch"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("epoch").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]

    span_days = (df.index[-1] - df.index[0]).days
    print(f"\r  {symbol} {granularity}s  — {len(df):,} bars | {span_days} days ({span_days/30:.1f} months)    ")
    return df


async def collect_all_data() -> dict:
    cache_path = os.path.join(CACHE_DIR, "raw_data_1yr.pkl")

    if os.path.exists(cache_path):
        age_hours = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache_path)) / 3600
        if age_hours < 12:
            print(f"Loading cached 1-year data ({age_hours:.1f}h old)...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    print("Fetching 1 year of data from Deriv API (this will take ~5-10 minutes)...")
    print(f"Symbols: {', '.join(SYMBOLS)} | Timeframes: {', '.join(TIMEFRAMES.keys())}\n")
    data = {}

    async with websockets.connect(DERIV_WS, ping_interval=30, ping_timeout=10, open_timeout=20) as ws:
        for symbol in SYMBOLS:
            data[symbol] = {}
            for tf_name, granularity in TIMEFRAMES.items():
                df = await fetch_candles_paginated(ws, symbol, granularity, TARGET_DAYS)
                if not df.empty:
                    data[symbol][tf_name] = df
                await asyncio.sleep(0.5)

    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print("\nAll data saved to cache.")
    return data


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    # Log returns
    ret = np.log(df["close"] / df["close"].shift(1))

    # ── Lagged returns ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        f[f"ret_lag{lag}"]  = ret.shift(lag)
        f[f"sign_lag{lag}"] = np.sign(ret.shift(lag))

    # ── Momentum / Rate of Change ───────────────────────────────────────────
    for period in [3, 5, 10, 20, 50]:
        f[f"roc{period}"] = df["close"].pct_change(period).shift(1)

    # ── Realized Volatility ─────────────────────────────────────────────────
    for window in [5, 10, 20, 50]:
        f[f"rvol{window}"] = ret.rolling(window).std().shift(1)

    # Volatility regime ratios (spike detection)
    f["vol_ratio_5_20"]  = f["rvol5"]  / f["rvol20"].replace(0, np.nan)
    f["vol_ratio_10_50"] = f["rvol10"] / f["rvol50"].replace(0, np.nan)

    # ── Mean Reversion: Z-score ─────────────────────────────────────────────
    for window in [5, 10, 20, 50, 100]:
        roll_mean = df["close"].rolling(window).mean().shift(1)
        roll_std  = df["close"].rolling(window).std().shift(1)
        f[f"zscore{window}"] = (df["close"].shift(1) - roll_mean) / roll_std.replace(0, np.nan)

    # ── EMA Distance (how far price is from trend) ──────────────────────────
    for period in [5, 10, 20, 50, 100, 200]:
        ema = df["close"].ewm(span=period, adjust=False).mean()
        f[f"ema_dist{period}"] = ((df["close"] - ema) / ema).shift(1)

    # EMA crossover signal
    ema_fast = df["close"].ewm(span=5,  adjust=False).mean()
    ema_slow = df["close"].ewm(span=20, adjust=False).mean()
    f["ema_cross_5_20"]  = ((ema_fast - ema_slow) / df["close"]).shift(1)

    ema_fast = df["close"].ewm(span=10, adjust=False).mean()
    ema_slow = df["close"].ewm(span=50, adjust=False).mean()
    f["ema_cross_10_50"] = ((ema_fast - ema_slow) / df["close"]).shift(1)

    # ── RSI ─────────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - 100 / (1 + rs)
        f[f"rsi{period}"]      = rsi.shift(1)
        f[f"rsi{period}_dist"] = (rsi.shift(1) - 50) / 50  # normalized distance from 50

    # ── MACD ────────────────────────────────────────────────────────────────
    ema12  = df["close"].ewm(span=12, adjust=False).mean()
    ema26  = df["close"].ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - signal).shift(1)
    f["macd_sign"] = np.sign(f["macd_hist"])

    # ── Candle Structure ────────────────────────────────────────────────────
    body       = df["close"] - df["open"]
    crange     = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    f["candle_dir"]    = np.sign(body).shift(1)
    f["body_ratio"]    = (body / crange).shift(1)          # 1=full bull, -1=full bear, 0=doji
    f["upper_wick_r"]  = (upper_wick / crange).shift(1)
    f["lower_wick_r"]  = (lower_wick / crange).shift(1)
    f["wick_imbalance"]= ((lower_wick - upper_wick) / crange).shift(1)  # +ve = bullish wicks

    # ── ATR-normalized range ────────────────────────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    for window in [5, 14]:
        atr = tr.rolling(window).mean()
        f[f"atr_norm{window}"] = (atr / df["close"]).shift(1)

    # ── Price position within N-bar range ───────────────────────────────────
    for window in [10, 20, 50]:
        roll_hi  = df["high"].rolling(window).max()
        roll_lo  = df["low"].rolling(window).min()
        hl_range = (roll_hi - roll_lo).replace(0, np.nan)
        f[f"hl_pos{window}"] = ((df["close"] - roll_lo) / hl_range).shift(1)

    # ── Consecutive bars in same direction ──────────────────────────────────
    direction = np.sign(ret)
    consec    = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    f["consec_dir"] = (consec * direction).shift(1)  # +ve = N consecutive up bars

    # ── Time features ────────────────────────────────────────────────────────
    f["hour"]           = df.index.hour
    f["hour_sin"]       = np.sin(2 * np.pi * df.index.hour / 24)
    f["hour_cos"]       = np.cos(2 * np.pi * df.index.hour / 24)
    f["is_london"]      = df.index.hour.isin(range(7, 16)).astype(int)
    f["is_ny"]          = df.index.hour.isin(range(12, 21)).astype(int)
    f["is_overlap"]     = df.index.hour.isin(range(12, 16)).astype(int)

    # ── Prediction targets ───────────────────────────────────────────────────
    for n in [1, 3, 5]:
        future_ret       = ret.rolling(n).sum().shift(-n)
        f[f"target_{n}"] = future_ret

    return f.replace([np.inf, -np.inf], np.nan).dropna(subset=[c for c in f.columns if c.startswith("target_")])


# ── Statistical Testing ───────────────────────────────────────────────────────
def spearman_ic(x: pd.Series, y: pd.Series):
    mask    = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < MIN_OBS:
        return None
    ic, pval = stats.spearmanr(x_c, y_c)
    n        = len(x_c)
    t_stat   = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic**2, 1e-10))
    return {"ic": ic, "abs_ic": abs(ic), "t_stat": t_stat, "p_value": pval, "n_obs": n}


def directional_accuracy(x: pd.Series, y: pd.Series) -> float:
    mask    = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < MIN_OBS:
        return np.nan
    correct = (np.sign(x_c) == np.sign(y_c)).mean()
    return correct


def quintile_analysis(x: pd.Series, y: pd.Series, n_buckets: int = 5):
    mask    = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < MIN_OBS * 2:
        return None
    try:
        buckets  = pd.qcut(x_c, n_buckets, labels=False, duplicates="drop")
        q_returns = y_c.groupby(buckets).mean()
        spread   = q_returns.iloc[-1] - q_returns.iloc[0]  # top bucket - bottom bucket
        monotone = all(q_returns.diff().dropna() >= 0) or all(q_returns.diff().dropna() <= 0)
        return {"spread": spread, "monotone": monotone, "q_rets": q_returns.tolist()}
    except Exception:
        return None


def test_all_signals(features: pd.DataFrame, symbol: str, tf: str) -> list:
    target_cols  = [c for c in features.columns if c.startswith("target_")]
    feature_cols = [c for c in features.columns if not c.startswith("target_")]
    results      = []

    for target in target_cols:
        y = features[target]
        for feat in feature_cols:
            x   = features[feat]
            res = spearman_ic(x, y)
            if res is None:
                continue
            dir_acc = directional_accuracy(x, y)
            q_res   = quintile_analysis(x, y)

            results.append({
                "symbol":      symbol,
                "timeframe":   tf,
                "feature":     feat,
                "target":      target,
                "ic":          round(res["ic"],     4),
                "abs_ic":      round(res["abs_ic"], 4),
                "t_stat":      round(res["t_stat"], 2),
                "p_value":     round(res["p_value"], 4),
                "n_obs":       res["n_obs"],
                "dir_acc":     round(dir_acc, 4) if not np.isnan(dir_acc) else None,
                "q_spread":    round(q_res["spread"], 6) if q_res else None,
                "monotone":    q_res["monotone"] if q_res else False,
                "significant": res["p_value"] < MAX_PVAL and res["abs_ic"] >= MIN_IC,
            })

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_top_signals(all_results: pd.DataFrame):
    sig = all_results[all_results["significant"]].copy()

    if sig.empty:
        print("\n[!] No statistically significant signals found.")
        return

    print("\n" + "=" * 80)
    print("  TOP SIGNALS — RANKED BY ABSOLUTE IC")
    print("=" * 80)

    # Universal signals: work across multiple symbols
    cross_symbol = (
        sig.groupby(["feature", "target"])
        .agg(n_symbols=("symbol", "nunique"), mean_ic=("abs_ic", "mean"))
        .reset_index()
        .query("n_symbols >= 3")
        .sort_values("mean_ic", ascending=False)
    )

    SEP = "-" * 80
    SEP2 = "-" * 28

    if not cross_symbol.empty:
        print(f"\n{SEP}")
        print("  CROSS-SYMBOL SIGNALS (work on 3+ pairs - most robust)")
        print(SEP)
        print(f"  {'Feature':<30} {'Target':<12} {'Symbols':>7} {'Mean IC':>8}")
        print(f"  {'-'*30} {'-'*12} {'-'*7} {'-'*8}")
        for _, row in cross_symbol.head(20).iterrows():
            print(f"  {row['feature']:<30} {row['target']:<12} {row['n_symbols']:>7} {row['mean_ic']:>8.4f}")

    # Top per symbol
    print(f"\n{SEP}")
    print("  TOP 15 SIGNALS PER SYMBOL (target = 1-bar forward return)")
    print(SEP)
    target1 = sig[sig["target"] == "target_1"]
    for symbol in SYMBOLS:
        sym_sig = target1[target1["symbol"] == symbol].sort_values("abs_ic", ascending=False).head(15)
        if sym_sig.empty:
            continue
        print(f"\n  {symbol}:")
        print(f"    {'Feature':<28} {'TF':<5} {'IC':>7} {'Dir%':>6} {'t-stat':>7} {'Mono':>5}")
        print(f"    {'-'*28} {'-'*5} {'-'*7} {'-'*6} {'-'*7} {'-'*5}")
        for _, r in sym_sig.iterrows():
            mono = "YES" if r["monotone"] else "no"
            dacc = f"{r['dir_acc']*100:.1f}%" if r["dir_acc"] else "  N/A"
            print(f"    {r['feature']:<28} {r['timeframe']:<5} {r['ic']:>+7.4f} {dacc:>6} {r['t_stat']:>7.2f} {mono:>5}")

    # Feature type summary
    print(f"\n{SEP}")
    print("  SIGNAL CATEGORY SUMMARY")
    print(SEP)
    categories = {
        "Mean Reversion (zscore)": sig["feature"].str.startswith("zscore"),
        "Momentum (roc)":          sig["feature"].str.startswith("roc"),
        "EMA Distance":            sig["feature"].str.startswith("ema_dist"),
        "Lagged Returns":          sig["feature"].str.startswith("ret_lag"),
        "RSI":                     sig["feature"].str.startswith("rsi"),
        "Volatility":              sig["feature"].str.startswith("rvol") | sig["feature"].str.startswith("vol_"),
        "Candle Structure":        sig["feature"].isin(["body_ratio", "candle_dir", "wick_imbalance", "upper_wick_r", "lower_wick_r"]),
        "Time":                    sig["feature"].isin(["hour", "is_london", "is_ny", "is_overlap", "hour_sin", "hour_cos"]),
        "Consecutive Bars":        sig["feature"] == "consec_dir",
        "Price Position":          sig["feature"].str.startswith("hl_pos"),
    }
    print(f"  {'Category':<30} {'Sig. signals':>12} {'Avg IC':>8}")
    print(f"  {'-'*30} {'-'*12} {'-'*8}")
    for cat_name, mask in categories.items():
        cat_sig = sig[mask]
        if cat_sig.empty:
            continue
        print(f"  {cat_name:<30} {len(cat_sig):>12} {cat_sig['abs_ic'].mean():>8.4f}")

    # Best timeframe breakdown
    print(f"\n{SEP}")
    print("  SIGNALS BY TIMEFRAME")
    print(SEP)
    tf_summary = sig.groupby("timeframe").agg(
        n_signals=("feature", "count"),
        avg_ic=("abs_ic", "mean"),
        max_ic=("abs_ic", "max"),
    ).sort_values("avg_ic", ascending=False)
    print(f"  {'Timeframe':<12} {'Signals':>8} {'Avg IC':>8} {'Max IC':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for tf, row in tf_summary.iterrows():
        print(f"  {tf:<12} {row['n_signals']:>8} {row['avg_ic']:>8.4f} {row['max_ic']:>8.4f}")

    print("\n" + "=" * 80)
    total = len(sig)
    total_tested = len(all_results)
    print(f"  {total} significant signals out of {total_tested} tested ({total/total_tested*100:.1f}%)")
    print("=" * 80)


def save_results(all_results: pd.DataFrame):
    path = os.path.join(CACHE_DIR, "signal_results.csv")
    all_results.sort_values("abs_ic", ascending=False).to_csv(path, index=False)
    sig_path = os.path.join(CACHE_DIR, "significant_signals.csv")
    all_results[all_results["significant"]].sort_values("abs_ic", ascending=False).to_csv(sig_path, index=False)
    print(f"\nFull results saved to: {path}")
    print(f"Significant signals:   {sig_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 80)
    print("  DERIV SYNTHETIC INDICES — QUANTITATIVE SIGNAL RESEARCH")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Timeframes: {', '.join(TIMEFRAMES.keys())}")
    print(f"  IC threshold: {MIN_IC} | p-value: {MAX_PVAL}")
    print("=" * 80 + "\n")

    # Step 1: Collect data
    raw_data = await collect_all_data()

    # Step 2: Engineer features + test signals
    all_results = []
    for symbol in SYMBOLS:
        if symbol not in raw_data:
            continue
        for tf_name in TIMEFRAMES:
            if tf_name not in raw_data[symbol]:
                continue
            df = raw_data[symbol][tf_name]
            if len(df) < MIN_OBS:
                print(f"  Skipping {symbol} {tf_name} — insufficient data ({len(df)} bars)")
                continue

            print(f"Testing {symbol} {tf_name} ({len(df)} bars)...", end=" ")
            features = engineer_features(df)
            results  = test_all_signals(features, symbol, tf_name)
            n_sig    = sum(1 for r in results if r["significant"])
            print(f"{n_sig} significant signals")
            all_results.extend(results)

    if not all_results:
        print("\n[!] No results to report.")
        return

    results_df = pd.DataFrame(all_results)

    # Step 3: Print report
    print_top_signals(results_df)

    # Step 4: Save
    save_results(results_df)


if __name__ == "__main__":
    asyncio.run(main())

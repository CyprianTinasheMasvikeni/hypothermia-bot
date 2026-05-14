#!/usr/bin/env python3
"""
Symbol Characterizer + Strategy Matcher
========================================
For every Deriv synthetic symbol:
  1. Calculate Hurst Exponent (trending vs mean-reverting vs random)
  2. Calculate autocorrelation structure
  3. Calculate volatility clustering
  4. Detect spike character (for Boom/Crash)
  5. Assign the best-fit strategy
  6. Backtest that strategy on 1 year of data
  7. Report results ranked by return

This is the correct approach: let the data assign the strategy,
not the other way around.

Run: python symbol_characterizer.py
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
APP_ID      = 1089
DERIV_WS    = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
CACHE_DIR   = "quant_cache/universe"
TARGET_DAYS = 365
BATCH_SIZE  = 5000
GRAN        = 900   # M15

# Exclude symbols we already know work
SKIP_SYMBOLS = {"stpRNG", "R_25"}

# Strategy thresholds
HURST_TREND     = 0.54   # above this = trending
HURST_REVERT    = 0.46   # below this = mean-reverting
MIN_TRADES      = 20     # minimum trades to report result
SPIKE_THRESHOLD = 5.0    # spike if candle range > 5x ATR

os.makedirs(CACHE_DIR, exist_ok=True)


# ── Data fetching ─────────────────────────────────────────────────────────────
async def fetch_symbol(ws, symbol, gran=GRAN, days=TARGET_DAYS):
    cache = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    if os.path.exists(cache):
        age_h = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache)) / 3600
        if age_h < 48:
            with open(cache, "rb") as f:
                return pickle.load(f)

    target = days * 24 * 3600 // gran
    batches, end = [], "latest"

    while True:
        req = {"ticks_history": symbol, "adjust_start_time": 1,
               "count": BATCH_SIZE, "end": end, "granularity": gran, "style": "candles"}
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
        if "error" in resp:
            return pd.DataFrame()
        candles = resp.get("candles", [])
        if not candles:
            break
        batches.append(candles)
        if sum(len(b) for b in batches) >= target or len(candles) < BATCH_SIZE:
            break
        end = candles[0]["epoch"] - 1
        await asyncio.sleep(0.35)

    if not batches:
        return pd.DataFrame()
    all_c = [c for b in reversed(batches) for c in b]
    df = pd.DataFrame(all_c)
    df["epoch"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("epoch")[["open","high","low","close"]].astype(float)
    df = df[~df.index.duplicated()].sort_index()
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df


# ── Statistical Characterization ──────────────────────────────────────────────
def hurst_exponent(ts, max_lag=100):
    """Hurst Exponent via rescaled range (R/S) method."""
    ts = np.log(ts / ts.shift(1)).dropna().values
    lags = range(10, min(max_lag, len(ts) // 4))
    rs_vals = []
    for lag in lags:
        chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
        if not chunks:
            continue
        rs = []
        for chunk in chunks:
            mean = chunk.mean()
            dev  = np.cumsum(chunk - mean)
            r    = dev.max() - dev.min()
            s    = chunk.std()
            if s > 0:
                rs.append(r / s)
        if rs:
            rs_vals.append((np.log(lag), np.log(np.mean(rs))))
    if len(rs_vals) < 5:
        return 0.5
    lags_log, rs_log = zip(*rs_vals)
    h, _, _, _, _ = stats.linregress(lags_log, rs_log)
    return float(np.clip(h, 0.1, 0.9))


def autocorrelation_profile(returns, lags=[1,2,3,5,10,20]):
    """Returns autocorrelation at each lag."""
    result = {}
    for lag in lags:
        if len(returns) > lag + 10:
            ac = returns.autocorr(lag=lag)
            result[f"ac_{lag}"] = round(ac, 4) if not np.isnan(ac) else 0.0
        else:
            result[f"ac_{lag}"] = 0.0
    return result


def spike_profile(df):
    """Detect spike frequency and magnitude for Boom/Crash detection."""
    atr = (df["high"] - df["low"]).rolling(20).mean()
    range_ = df["high"] - df["low"]
    spike_ratio = (range_ / atr.replace(0, np.nan)).dropna()
    pct_spikes = (spike_ratio > SPIKE_THRESHOLD).mean()
    avg_spike_size = spike_ratio[spike_ratio > SPIKE_THRESHOLD].mean() if pct_spikes > 0 else 0
    return {"pct_spikes": pct_spikes, "avg_spike_size": avg_spike_size}


def characterize(df):
    """Full statistical characterization of a symbol."""
    if len(df) < 500:
        return None
    ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    h   = hurst_exponent(df["close"])
    ac  = autocorrelation_profile(ret)
    sp  = spike_profile(df)

    # Volatility clustering: autocorr of |returns|
    abs_ret_ac = ret.abs().autocorr(lag=1)

    if h > HURST_TREND:
        regime = "TRENDING"
    elif h < HURST_REVERT:
        regime = "MEAN_REVERTING"
    else:
        regime = "RANDOM"

    # Override: if >3% candles are spikes, it's a spike index
    if sp["pct_spikes"] > 0.03:
        regime = "SPIKE_INDEX"

    # Assign strategy
    strategy_map = {
        "TRENDING":      "trend_follow",
        "MEAN_REVERTING":"mean_revert",
        "RANDOM":        "skip",
        "SPIKE_INDEX":   "spike_fade",
    }

    return {
        "hurst":          round(h, 3),
        "regime":         regime,
        "strategy":       strategy_map[regime],
        "ac_1":           ac["ac_1"],
        "ac_5":           ac["ac_5"],
        "ac_10":          ac["ac_10"],
        "pct_spikes":     round(sp["pct_spikes"], 4),
        "vol_cluster":    round(abs_ret_ac, 3) if not np.isnan(abs_ret_ac) else 0,
        "n_bars":         len(df),
        "span_days":      (df.index[-1] - df.index[0]).days,
    }


# ── Strategies ────────────────────────────────────────────────────────────────
def run_trend_follow(df):
    """EMA crossover trend following — for trending symbols."""
    df = df.copy()
    df["ema20"]  = df["close"].ewm(span=20,  adjust=False).mean()
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["atr"]    = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"]    = _rsi(df["close"], 14)

    trades, in_trade, direction = [], False, None
    entry, sl, tp = 0, 0, 0

    for i in range(210, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]
        if in_trade:
            if direction == "long":
                if r["low"] <= sl:
                    trades.append({"pnl": (sl - entry)/entry, "result": "loss"})
                    in_trade = False
                elif r["high"] >= tp:
                    trades.append({"pnl": (tp - entry)/entry, "result": "win"})
                    in_trade = False
            else:
                if r["high"] >= sl:
                    trades.append({"pnl": (entry - sl)/entry, "result": "loss"})
                    in_trade = False
                elif r["low"] <= tp:
                    trades.append({"pnl": (entry - tp)/entry, "result": "win"})
                    in_trade = False
            continue

        atr = p["atr"]
        if atr <= 0 or np.isnan(atr):
            continue

        bull = (p["ema20"] > p["ema50"] > p["ema200"] and
                p["close"] > p["ema20"] and p["rsi"] > 50 and
                r["close"] > r["open"] and
                abs(r["close"] - r["ema20"]) < 1.5 * atr)

        bear = (p["ema20"] < p["ema50"] < p["ema200"] and
                p["close"] < p["ema20"] and p["rsi"] < 50 and
                r["close"] < r["open"] and
                abs(r["close"] - r["ema20"]) < 1.5 * atr)

        if bull:
            entry, direction, in_trade = r["close"], "long", True
            sl, tp = entry - 2*atr, entry + 3*atr
        elif bear:
            entry, direction, in_trade = r["close"], "short", True
            sl, tp = entry + 2*atr, entry - 3*atr

    return _trade_stats(trades)


def run_mean_revert(df):
    """Bollinger Band mean reversion — for mean-reverting symbols."""
    df = df.copy()
    df["mid"]  = df["close"].rolling(20).mean()
    df["std"]  = df["close"].rolling(20).std()
    df["upper"]= df["mid"] + 2 * df["std"]
    df["lower"]= df["mid"] - 2 * df["std"]
    df["atr"]  = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"]  = _rsi(df["close"], 14)
    df["zscore"]= (df["close"] - df["mid"]) / df["std"].replace(0, np.nan)

    trades, in_trade, direction = [], False, None
    entry, sl, tp = 0, 0, 0

    for i in range(25, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]
        if in_trade:
            if direction == "long":
                if r["low"] <= sl:
                    trades.append({"pnl": (sl - entry)/entry, "result": "loss"})
                    in_trade = False
                elif r["high"] >= tp:
                    trades.append({"pnl": (tp - entry)/entry, "result": "win"})
                    in_trade = False
            else:
                if r["high"] >= sl:
                    trades.append({"pnl": (entry - sl)/entry, "result": "loss"})
                    in_trade = False
                elif r["low"] <= tp:
                    trades.append({"pnl": (entry - tp)/entry, "result": "win"})
                    in_trade = False
            continue

        atr = p["atr"]
        if atr <= 0 or np.isnan(atr):
            continue

        # Buy when price is 2+ std below mean and RSI oversold
        if p["zscore"] < -2.0 and p["rsi"] < 35 and r["close"] > r["open"]:
            entry, direction, in_trade = r["close"], "long", True
            sl  = entry - 1.5 * atr
            tp  = p["mid"]   # target = mean

        # Sell when price is 2+ std above mean and RSI overbought
        elif p["zscore"] > 2.0 and p["rsi"] > 65 and r["close"] < r["open"]:
            entry, direction, in_trade = r["close"], "short", True
            sl  = entry + 1.5 * atr
            tp  = p["mid"]

    return _trade_stats(trades)


def run_spike_fade(df):
    """
    Fade the spike: after a large spike candle, trade in the
    opposite direction expecting a partial reversal.
    Works on Boom (fade up spikes) and Crash (fade down spikes).
    """
    df = df.copy()
    df["atr"]    = (df["high"] - df["low"]).rolling(20).mean()
    df["range"]  = df["high"] - df["low"]
    df["rsi"]    = _rsi(df["close"], 14)

    trades, in_trade, direction = [], False, None
    entry, sl, tp = 0, 0, 0
    COOLDOWN, cooldown = 3, 0

    for i in range(25, len(df)):
        r = df.iloc[i]
        p = df.iloc[i-1]

        if cooldown > 0:
            cooldown -= 1
            continue

        if in_trade:
            if direction == "long":
                if r["low"] <= sl:
                    trades.append({"pnl": (sl - entry)/entry, "result": "loss"})
                    in_trade = False; cooldown = COOLDOWN
                elif r["high"] >= tp:
                    trades.append({"pnl": (tp - entry)/entry, "result": "win"})
                    in_trade = False; cooldown = COOLDOWN
            else:
                if r["high"] >= sl:
                    trades.append({"pnl": (entry - sl)/entry, "result": "loss"})
                    in_trade = False; cooldown = COOLDOWN
                elif r["low"] <= tp:
                    trades.append({"pnl": (entry - tp)/entry, "result": "win"})
                    in_trade = False; cooldown = COOLDOWN
            continue

        atr = p["atr"]
        if atr <= 0 or np.isnan(atr):
            continue

        spike_ratio = p["range"] / atr if atr > 0 else 0

        # Boom spike (up spike) — fade it going short
        up_spike = (spike_ratio > SPIKE_THRESHOLD and
                    p["close"] > p["open"] and
                    p["rsi"] > 60)
        # Crash spike (down spike) — fade it going long
        down_spike = (spike_ratio > SPIKE_THRESHOLD and
                      p["close"] < p["open"] and
                      p["rsi"] < 40)

        if up_spike:
            entry, direction, in_trade = r["open"], "short", True
            sl = entry + 1.5 * atr
            tp = entry - 2.0 * atr
        elif down_spike:
            entry, direction, in_trade = r["open"], "long", True
            sl = entry - 1.5 * atr
            tp = entry + 2.0 * atr

    return _trade_stats(trades)


def _rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _trade_stats(trades):
    if len(trades) < MIN_TRADES:
        return None
    df_t  = pd.DataFrame(trades)
    wins  = df_t[df_t["result"] == "win"]
    losses= df_t[df_t["result"] == "loss"]
    wr    = len(wins) / len(df_t)
    avg_w = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_l = losses["pnl"].mean() if len(losses) > 0 else 0
    gw    = wins["pnl"].sum()    if len(wins)   > 0 else 0
    gl    = losses["pnl"].abs().sum() if len(losses) > 0 else 1e-9
    return {
        "n_trades":  len(df_t),
        "win_rate":  wr,
        "avg_win":   avg_w,
        "avg_loss":  avg_l,
        "pf":        gw / gl,
        "ev":        wr * avg_w + (1 - wr) * avg_l,
        "total_pnl": df_t["pnl"].sum(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 72)
    print("  SYMBOL CHARACTERIZER + STRATEGY MATCHER")
    print("  Each symbol gets the strategy that fits its natural behavior")
    print("=" * 72)

    # Get all symbols
    async with websockets.connect(DERIV_WS, ping_interval=30, ping_timeout=10, open_timeout=20) as ws:
        req = {"active_symbols": "brief", "product_type": "basic"}
        await ws.send(json.dumps(req))
        resp = json.loads(await ws.recv())
        all_syms = [s["symbol"] for s in resp.get("active_symbols", [])
                    if s.get("market") == "synthetic_index"
                    and not s.get("is_trading_suspended", False)
                    and s["symbol"] not in SKIP_SYMBOLS]

        print(f"\nTesting {len(all_syms)} symbols (skipping {', '.join(SKIP_SYMBOLS)})\n")

        results = []

        for i, sym in enumerate(all_syms):
            print(f"[{i+1:>2}/{len(all_syms)}] {sym:<18}", end=" ", flush=True)

            df = await fetch_symbol(ws, sym)
            if df.empty or len(df) < 500:
                print("insufficient data")
                await asyncio.sleep(0.3)
                continue

            char = characterize(df)
            if not char:
                print("characterization failed")
                continue

            # Run matched strategy
            strategy = char["strategy"]
            if strategy == "trend_follow":
                res = run_trend_follow(df)
            elif strategy == "mean_revert":
                res = run_mean_revert(df)
            elif strategy == "spike_fade":
                res = run_spike_fade(df)
            else:
                print(f"H={char['hurst']:.3f} [{char['regime']}] -- SKIP (random walk)")
                continue

            if res is None:
                print(f"H={char['hurst']:.3f} [{char['regime']}] -> {strategy} -- too few trades")
                continue

            flag = ""
            if res["win_rate"] >= 0.55 and res["pf"] >= 1.5 and res["ev"] > 0:
                flag = "  <<< STRONG"
            elif res["win_rate"] >= 0.50 and res["pf"] >= 1.2 and res["ev"] > 0:
                flag = "  < decent"

            print(f"H={char['hurst']:.3f} [{char['regime']:<14}] -> {strategy:<12} "
                  f"WR={res['win_rate']:.1%} PF={res['pf']:.2f} EV={res['ev']:+.4f}{flag}")

            results.append({
                "symbol":    sym,
                "hurst":     char["hurst"],
                "regime":    char["regime"],
                "strategy":  strategy,
                "n_trades":  res["n_trades"],
                "win_rate":  res["win_rate"],
                "pf":        res["pf"],
                "ev":        res["ev"],
                "total_pnl": res["total_pnl"],
                "span_days": char["span_days"],
                "ac_1":      char["ac_1"],
                "pct_spikes":char["pct_spikes"],
            })

            await asyncio.sleep(0.1)

    # ── Report ────────────────────────────────────────────────────────────────
    if not results:
        print("\nNo results.")
        return

    df_r = pd.DataFrame(results).sort_values("ev", ascending=False)

    print(f"\n\n{'='*72}")
    print("  FINAL RESULTS - RANKED BY EXPECTANCY (strategy matched to symbol)")
    print(f"{'='*72}")

    strong = df_r[(df_r["win_rate"] >= 0.55) & (df_r["pf"] >= 1.5) & (df_r["ev"] > 0)]
    decent = df_r[(df_r["win_rate"] >= 0.50) & (df_r["pf"] >= 1.2) & (df_r["ev"] > 0) &
                  (~df_r["symbol"].isin(strong["symbol"]))]
    rest   = df_r[~df_r["symbol"].isin(strong["symbol"]) & ~df_r["symbol"].isin(decent["symbol"])]

    def show(sub, title):
        if sub.empty:
            return
        print(f"\n  {title}")
        print(f"  {'Symbol':<14} {'Regime':<16} {'Strategy':<14} {'WR':>6} {'PF':>6} {'EV':>8} {'Trades':>7} {'Days':>5}")
        print(f"  {'-'*14} {'-'*16} {'-'*14} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*5}")
        for _, r in sub.iterrows():
            print(f"  {r['symbol']:<14} {r['regime']:<16} {r['strategy']:<14} "
                  f"{r['win_rate']:>6.1%} {r['pf']:>6.2f} {r['ev']:>+8.4f} "
                  f"{int(r['n_trades']):>7} {int(r['span_days']):>5}")

    show(strong, "STRONG EDGE - BUILD BOTS ON THESE")
    show(decent, "DECENT EDGE - WORTH MONITORING")
    show(rest,   "WEAK/NO EDGE - SKIP")

    # Save
    out = "quant_cache/characterizer_results.csv"
    df_r.to_csv(out, index=False)
    print(f"\n\nResults saved to: {out}")
    print(f"\nStrong edge: {len(strong)} | Decent: {len(decent)} | Skip: {len(rest)}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    asyncio.run(main())

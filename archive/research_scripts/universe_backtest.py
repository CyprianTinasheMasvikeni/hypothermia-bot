#!/usr/bin/env python3
"""
Universe Backtest — All Deriv Synthetic Indices
================================================
1. Discovers every available synthetic symbol from Deriv API
2. Pulls 2 years of M15 candle data for each (paginated)
3. Runs the proven EMA + pullback + confirmation strategy on each
4. Ranks all symbols by profitability, win rate, and expectancy
5. Outputs a definitive list of which symbols to trade

This is the Jim Simons approach: test one proven strategy across
every available market, run it only on the ones where it works.

Run: python universe_backtest.py
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
APP_ID       = 1089
DERIV_WS     = f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}"
TARGET_DAYS  = 730       # 2 years
BATCH_SIZE   = 5000
GRANULARITY  = 900       # M15
CACHE_DIR    = "quant_cache/universe"
RESULTS_PATH = "quant_cache/universe_results.csv"

# Strategy parameters (from proven backtest)
EMA_FAST     = 50
EMA_SLOW     = 200
ATR_PERIOD   = 14
SL_ATR       = 2.0       # stop loss = 2x ATR
TP_ATR       = 3.0       # take profit = 3x ATR
MIN_BODY_PCT = 0.55      # confirmation candle min body/range ratio
PULLBACK_ATR = 1.5       # max distance from EMA for entry

# Minimum requirements to consider a symbol
MIN_BARS     = 2000      # need at least this many M15 bars
MIN_TRADES   = 30        # need at least 30 trades to measure edge

os.makedirs(CACHE_DIR, exist_ok=True)

# ── Symbol Discovery ──────────────────────────────────────────────────────────
async def get_all_synthetic_symbols(ws) -> list:
    req = {"active_symbols": "brief", "product_type": "basic"}
    await ws.send(json.dumps(req))
    resp = json.loads(await ws.recv())

    symbols = []
    for s in resp.get("active_symbols", []):
        if s.get("market") == "synthetic_index" and not s.get("is_trading_suspended", False):
            symbols.append({
                "symbol":       s["symbol"],
                "display_name": s["display_name"],
                "submarket":    s.get("submarket", ""),
                "pip":          float(s.get("pip", 0.001)),
            })
    return symbols


# ── Data Collection ───────────────────────────────────────────────────────────
async def fetch_paginated(ws, symbol: str, granularity: int, target_days: int) -> pd.DataFrame:
    target_bars = target_days * 24 * 3600 // granularity
    batches     = []
    end_time    = "latest"

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
            return pd.DataFrame()

        candles = resp.get("candles", [])
        if not candles:
            break

        batches.append(candles)
        total = sum(len(b) for b in batches)

        if total >= target_bars or len(candles) < BATCH_SIZE:
            break

        end_time = candles[0]["epoch"] - 1
        await asyncio.sleep(0.3)

    if not batches:
        return pd.DataFrame()

    all_c = []
    for b in reversed(batches):
        all_c.extend(b)

    df = pd.DataFrame(all_c)
    df["epoch"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("epoch").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.duplicated(keep="first")]
    return df


# ── Strategy ──────────────────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    # ATR
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # Candle body
    df["body"]       = (df["close"] - df["open"]).abs()
    df["candle_rng"] = df["high"] - df["low"]
    df["body_pct"]   = df["body"] / df["candle_rng"].replace(0, np.nan)
    df["bull_candle"]= (df["close"] > df["open"]) & (df["body_pct"] >= MIN_BODY_PCT)
    df["bear_candle"]= (df["close"] < df["open"]) & (df["body_pct"] >= MIN_BODY_PCT)

    # Trend
    df["uptrend"]   = df["ema_fast"] > df["ema_slow"]
    df["downtrend"] = df["ema_fast"] < df["ema_slow"]

    # EMA mid
    df["ema_mid"] = (df["ema_fast"] + df["ema_slow"]) / 2

    return df.dropna()


def run_strategy(df: pd.DataFrame) -> dict:
    df = compute_indicators(df)
    if len(df) < MIN_BARS:
        return None

    trades     = []
    in_trade   = False
    trade_dir  = None
    entry_px   = 0.0
    sl_px      = 0.0
    tp_px      = 0.0

    for i in range(EMA_SLOW + 5, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_trade:
            # Check SL/TP using high/low of candle
            if trade_dir == "long":
                if row["low"] <= sl_px:
                    pnl = (sl_px - entry_px) / entry_px
                    trades.append({"result": "loss", "pnl": pnl, "date": df.index[i]})
                    in_trade = False
                elif row["high"] >= tp_px:
                    pnl = (tp_px - entry_px) / entry_px
                    trades.append({"result": "win", "pnl": pnl, "date": df.index[i]})
                    in_trade = False
            else:  # short
                if row["high"] >= sl_px:
                    pnl = (entry_px - sl_px) / entry_px
                    trades.append({"result": "loss", "pnl": -pnl, "date": df.index[i]})
                    in_trade = False
                elif row["low"] <= tp_px:
                    pnl = (entry_px - tp_px) / entry_px
                    trades.append({"result": "win", "pnl": pnl, "date": df.index[i]})
                    in_trade = False
            continue

        atr = prev["atr"]
        if atr <= 0:
            continue

        # Long signal: uptrend + price near EMA + bullish candle
        if (prev["uptrend"] and
            abs(prev["close"] - prev["ema_fast"]) < PULLBACK_ATR * atr and
            prev["close"] > prev["ema_slow"] and
            row["bull_candle"]):

            entry_px  = row["close"]
            sl_px     = entry_px - SL_ATR * atr
            tp_px     = entry_px + TP_ATR * atr
            in_trade  = True
            trade_dir = "long"

        # Short signal: downtrend + price near EMA + bearish candle
        elif (prev["downtrend"] and
              abs(prev["close"] - prev["ema_fast"]) < PULLBACK_ATR * atr and
              prev["close"] < prev["ema_slow"] and
              row["bear_candle"]):

            entry_px  = row["close"]
            sl_px     = entry_px + SL_ATR * atr
            tp_px     = entry_px - TP_ATR * atr
            in_trade  = True
            trade_dir = "short"

    if len(trades) < MIN_TRADES:
        return None

    df_t      = pd.DataFrame(trades)
    wins      = df_t[df_t["result"] == "win"]
    losses    = df_t[df_t["result"] == "loss"]
    win_rate  = len(wins) / len(df_t)
    avg_win   = wins["pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss  = losses["pnl"].mean() if len(losses) > 0 else 0

    # Profit factor
    gross_win  = wins["pnl"].sum()   if len(wins)   > 0 else 0
    gross_loss = losses["pnl"].abs().sum() if len(losses) > 0 else 1e-9
    pf         = gross_win / gross_loss

    # Expectancy per trade (in % of entry)
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # Monthly trade frequency
    if len(df_t) > 1:
        span_days    = (df_t["date"].iloc[-1] - df_t["date"].iloc[0]).days
        trades_month = len(df_t) / max(span_days / 30, 1)
    else:
        trades_month = 0

    # Sharpe-like: annualized return / std of trade PnLs
    pnl_std = df_t["pnl"].std()
    sharpe  = (expectancy / pnl_std * np.sqrt(trades_month * 12)) if pnl_std > 0 else 0

    return {
        "n_trades":      len(df_t),
        "win_rate":      win_rate,
        "avg_win_pct":   avg_win   * 100,
        "avg_loss_pct":  avg_loss  * 100,
        "profit_factor": pf,
        "expectancy_pct":expectancy * 100,
        "trades_month":  round(trades_month, 1),
        "sharpe":        round(sharpe, 2),
        "total_pnl_pct": df_t["pnl"].sum() * 100,
        "n_bars":        len(df),
        "span_days":     span_days if len(df_t) > 1 else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 72)
    print("  UNIVERSE BACKTEST — ALL DERIV SYNTHETIC INDICES")
    print(f"  Strategy: EMA{EMA_FAST}/{EMA_SLOW} + pullback + confirmation")
    print(f"  Data: {TARGET_DAYS} days ({TARGET_DAYS//365} years) of M15 candles")
    print(f"  SL: {SL_ATR}x ATR | TP: {TP_ATR}x ATR | Min body: {MIN_BODY_PCT:.0%}")
    print("=" * 72)

    async with websockets.connect(DERIV_WS, ping_interval=30, ping_timeout=10, open_timeout=20) as ws:

        # Step 1: Discover all synthetic symbols
        print("\nDiscovering all synthetic indices on Deriv...")
        symbols = await get_all_synthetic_symbols(ws)
        print(f"Found {len(symbols)} synthetic symbols.\n")

        # Group by submarket for display
        submarkets = {}
        for s in symbols:
            sm = s["submarket"]
            submarkets.setdefault(sm, []).append(s["symbol"])
        for sm, syms in sorted(submarkets.items()):
            print(f"  {sm}: {', '.join(syms)}")

        print(f"\n{'-'*72}")
        print("Pulling 2 years of M15 data + running backtest...")
        print(f"{'-'*72}\n")

        results = []

        for i, sym_info in enumerate(symbols):
            sym  = sym_info["symbol"]
            name = sym_info["display_name"]

            # Check cache
            cache_file = os.path.join(CACHE_DIR, f"{sym}.pkl")
            if os.path.exists(cache_file):
                age_h = (pd.Timestamp.now().timestamp() - os.path.getmtime(cache_file)) / 3600
                if age_h < 24:
                    with open(cache_file, "rb") as f:
                        df = pickle.load(f)
                    cached = True
                else:
                    df = await fetch_paginated(ws, sym, GRANULARITY, TARGET_DAYS)
                    cached = False
            else:
                df = await fetch_paginated(ws, sym, GRANULARITY, TARGET_DAYS)
                cached = False

            if df.empty or len(df) < MIN_BARS:
                print(f"  [{i+1:>2}/{len(symbols)}] {sym:<20} -- insufficient data ({len(df)} bars), skipping")
                await asyncio.sleep(0.3)
                continue

            if not cached:
                with open(cache_file, "wb") as f:
                    pickle.dump(df, f)

            span = (df.index[-1] - df.index[0]).days
            res  = run_strategy(df)

            if res is None:
                print(f"  [{i+1:>2}/{len(symbols)}] {sym:<20} -- too few trades, skipping  ({span}d data)")
                await asyncio.sleep(0.3)
                continue

            res["symbol"]       = sym
            res["display_name"] = name
            res["submarket"]    = sym_info["submarket"]
            results.append(res)

            flag = ""
            if res["win_rate"] >= 0.55 and res["profit_factor"] >= 1.5 and res["expectancy_pct"] > 0:
                flag = "  <<< STRONG EDGE"
            elif res["win_rate"] >= 0.50 and res["profit_factor"] >= 1.2:
                flag = "  < edge"

            print(f"  [{i+1:>2}/{len(symbols)}] {sym:<20} "
                  f"WR={res['win_rate']:.1%}  PF={res['profit_factor']:.2f}  "
                  f"EV={res['expectancy_pct']:+.3f}%  "
                  f"Trades/mo={res['trades_month']:.1f}  "
                  f"({span}d){flag}")

            if not cached:
                await asyncio.sleep(0.4)

    # ── Final Report ──────────────────────────────────────────────────────────
    if not results:
        print("\n[!] No results to report.")
        return

    df_r = pd.DataFrame(results).sort_values("expectancy_pct", ascending=False)

    print(f"\n\n{'='*72}")
    print("  FINAL RANKINGS - ALL SYMBOLS BY EDGE STRENGTH")
    print(f"{'='*72}\n")

    # Strong edge symbols
    strong = df_r[(df_r["win_rate"] >= 0.55) & (df_r["profit_factor"] >= 1.5) & (df_r["expectancy_pct"] > 0)]
    decent = df_r[(df_r["win_rate"] >= 0.50) & (df_r["profit_factor"] >= 1.2) & (df_r["expectancy_pct"] > 0) & (~df_r["symbol"].isin(strong["symbol"]))]
    rest   = df_r[~df_r["symbol"].isin(strong["symbol"]) & ~df_r["symbol"].isin(decent["symbol"])]

    def print_table(df_sub, title):
        if df_sub.empty:
            return
        print(f"\n  {title}")
        print(f"  {'Symbol':<20} {'SubMkt':<20} {'WR':>6} {'PF':>6} {'EV%':>8} {'Trades/mo':>10} {'Sharpe':>7} {'Days':>6}")
        print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*7} {'-'*6}")
        for _, row in df_sub.iterrows():
            print(f"  {row['symbol']:<20} {row['submarket']:<20} "
                  f"{row['win_rate']:>6.1%} {row['profit_factor']:>6.2f} "
                  f"{row['expectancy_pct']:>+8.3f} {row['trades_month']:>10.1f} "
                  f"{row['sharpe']:>7.2f} {row['span_days']:>6}")

    print_table(strong, "STRONG EDGE - RUN BOTS ON THESE")
    print_table(decent, "DECENT EDGE - WORTH MONITORING")
    print_table(rest,   "WEAK / NO EDGE - SKIP")

    # Save
    df_r.to_csv(RESULTS_PATH, index=False)
    print(f"\n\nFull results saved to: {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"  Total symbols tested:  {len(results)}")
    print(f"  Strong edge:           {len(strong)} symbols")
    print(f"  Decent edge:           {len(decent)} symbols")
    print(f"  Weak/no edge:          {len(rest)} symbols")
    if not strong.empty:
        total_trades_month = strong["trades_month"].sum()
        print(f"\n  Running bots on strong edge symbols gives:")
        print(f"  ~{total_trades_month:.0f} trades/month across all symbols combined")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    asyncio.run(main())

"""
fetch_commodity_data.py
=======================
Mines maximum available M1 + M5 + H1 candle history from Deriv WebSocket API
for commodities and forex pairs.

Usage:
  python fetch_commodity_data.py            # all symbols, all timeframes
  python fetch_commodity_data.py XAUUSD     # single symbol
  python fetch_commodity_data.py XAUUSD M1  # single symbol, single timeframe
  python fetch_commodity_data.py all M1     # all symbols, M1 only
"""

import asyncio
import json
import sys
from pathlib import Path
import pandas as pd
import websockets

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DERIV_WS = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
DATA_DIR  = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGETS = [
    ("frxXAUUSD", "XAUUSD"),   # Gold
    ("frxXAGUSD", "XAGUSD"),   # Silver
    ("frxXPDUSD", "XPDUSD"),   # Palladium
    ("frxXPTUSD", "XPTUSD"),   # Platinum
    ("frxEURUSD", "EURUSD"),   # Euro / US Dollar
    ("frxGBPUSD", "GBPUSD"),   # British Pound / US Dollar
    ("frxUSDJPY", "USDJPY"),   # US Dollar / Japanese Yen
    ("frxAUDUSD", "AUDUSD"),   # Australian Dollar / US Dollar
]

GRANS = {
    "M1": 60,
    "M5": 300,
    "H1": 3600,
}

MAX_BARS = {
    "M1": 300_000,   # ~10 months of active M1 bars
    "M5":  55_000,
    "H1":   8_000,
}

BATCH = 5000
DELAY = 0.6


async def fetch_batch(ws, symbol: str, gran: int, end, req_id: int) -> dict:
    req = {
        "ticks_history": symbol,
        "style":         "candles",
        "granularity":   gran,
        "count":         BATCH,
        "end":           end,
        "req_id":        req_id,
    }
    await ws.send(json.dumps(req))
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(raw)
        if msg.get("req_id") == req_id:
            return msg


async def fetch_symbol_gran(sym_deriv: str, name: str, gran_label: str, gran_sec: int, max_bars: int) -> pd.DataFrame:
    all_candles: list = []
    end = "latest"
    req_id = 1

    try:
        async with websockets.connect(DERIV_WS, ping_interval=30, ping_timeout=10, open_timeout=20) as ws:
            while len(all_candles) < max_bars:
                resp = await fetch_batch(ws, sym_deriv, gran_sec, end, req_id)
                req_id += 1

                if "error" in resp:
                    err = resp["error"]
                    print(f"    [{name} {gran_label}] Error: {err.get('message', err)}", flush=True)
                    if not all_candles:
                        return pd.DataFrame()
                    break

                candles = resp.get("candles", [])
                if not candles:
                    print(f"    [{name} {gran_label}] History exhausted.", flush=True)
                    break

                if all_candles:
                    known = {c["epoch"] for c in all_candles}
                    candles = [c for c in candles if c["epoch"] not in known]
                if not candles:
                    break

                all_candles.extend(candles)
                oldest = int(min(c["epoch"] for c in candles))
                end = oldest - 1

                total = len(all_candles)
                if total % 25000 < BATCH or total < BATCH + 1:
                    print(f"    [{name} {gran_label}] {total:>8,} bars  |  oldest epoch {oldest}", flush=True)

                if len(candles) < BATCH // 2:
                    print(f"    [{name} {gran_label}] Reached start of history ({total:,} bars).", flush=True)
                    break

                await asyncio.sleep(DELAY)

    except Exception as exc:
        print(f"    [{name} {gran_label}] Connection error: {exc}", flush=True)
        if not all_candles:
            return pd.DataFrame()
        print(f"    [{name} {gran_label}] Saving partial ({len(all_candles):,} bars).", flush=True)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles)
    df["time"] = pd.to_datetime(df["epoch"].astype(int), unit="s", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return (
        df[["time","epoch","open","high","low","close"]]
        .sort_values("time").reset_index(drop=True)
    )


async def main():
    args   = sys.argv[1:]
    sym_f  = args[0].upper() if args else None       # "XAUUSD" or "all" or None
    gran_f = args[1].upper() if len(args) > 1 else None  # "M1", "M5", "H1" or None

    targets = [(d, n) for d, n in TARGETS if (sym_f is None or sym_f == "ALL" or n == sym_f)]
    grans   = {k: v for k, v in GRANS.items() if gran_f is None or k == gran_f}

    print(flush=True)
    print("=" * 70, flush=True)
    print("  COMMODITY / FOREX DATA MINER — Deriv WebSocket API", flush=True)
    print(f"  Symbols  : {[t[1] for t in targets]}", flush=True)
    print(f"  Timeframes: {list(grans.keys())}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    results = []

    for sym_deriv, name in targets:
        print(f">>> {name}  ({sym_deriv})", flush=True)
        sym_r = {"symbol": name}

        for gran_label, gran_sec in grans.items():
            out_path = DATA_DIR / f"cache_{name}_{gran_label}.csv"

            # Skip if already have enough bars
            if out_path.exists():
                existing = pd.read_csv(out_path)
                if len(existing) >= MAX_BARS[gran_label] * 0.85:
                    print(f"  {gran_label}: already have {len(existing):,} bars — skipping.", flush=True)
                    sym_r[gran_label] = len(existing)
                    continue

            max_b = MAX_BARS[gran_label]
            df = await fetch_symbol_gran(sym_deriv, name, gran_label, gran_sec, max_b)

            if df.empty:
                print(f"  {gran_label}: NOT AVAILABLE", flush=True)
                sym_r[gran_label] = 0
                continue

            df.to_csv(out_path, index=False)
            sym_r[gran_label] = len(df)
            print(f"  {gran_label}: {len(df):,} bars  |  {df['time'].min().date()} → {df['time'].max().date()}  |  saved.", flush=True)

        results.append(sym_r)
        print(flush=True)

    print("=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Symbol':<10} {'M1':>10} {'M5':>10} {'H1':>10}  Status", flush=True)
    print("  " + "-" * 50, flush=True)
    for r in results:
        m1 = r.get("M1", "—")
        m5 = r.get("M5", "—")
        h1 = r.get("H1", "—")
        ok = "READY" if (isinstance(m1, int) and m1 >= 1000) or (isinstance(m5, int) and m5 >= 1000) else "NO DATA"
        print(f"  {r['symbol']:<10} {str(m1):>10} {str(m5):>10} {str(h1):>10}  {ok}", flush=True)
    print(flush=True)
    print("  Run: python commodity_backtest.py XAUUSD M1", flush=True)
    print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())

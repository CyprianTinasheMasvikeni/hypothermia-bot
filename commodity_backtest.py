"""
commodity_backtest.py  (vectorized — fast)
===========================================
Tests spike reversion, EMA pullback, and EMA crossover on commodity/forex data.
Signal detection is fully vectorized (numpy masks).
Supports M5 and M1 timeframes.

Usage:
  python commodity_backtest.py                      # all symbols, M5
  python commodity_backtest.py XAUUSD               # Gold, M5
  python commodity_backtest.py XAUUSD M1            # Gold, M1 (1-minute)
  python commodity_backtest.py XAUUSD M1 spike      # Gold M1, spike only
  python commodity_backtest.py XAUUSD M1 cross      # Gold M1, EMA cross only
  python commodity_backtest.py all M1               # all symbols, M1
"""

import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── CONSTANTS ────────────────────────────────────────────────────────────────
ATR_PERIOD   = 14
PARTIAL_R    = 2.0
PARTIAL_PCT  = 0.50
CHAND_TIERS  = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
ACTIVE_HOURS = frozenset(range(7, 21))   # 07:00–20:59 UTC

# Per-timeframe settings
TF_SETTINGS = {
    "M5":    {"cooldown": 8,  "max_day": 4,  "min_trades": 15},
    "M5_HF": {"cooldown": 2,  "max_day": 12, "min_trades": 30},  # high-frequency M5
    "M1":    {"cooldown": 8,  "max_day": 10, "min_trades": 20},
}

# ── PARAMETER GRIDS ─────────────────────────────────────────────────────────
# M5 grids (existing)
SPIKE_GRID = dict(
    h1_ema       = [14, 21, 50],
    spike_mult   = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    hold         = [12, 18, 24, 36, 48],
    direction    = ["BUY", "SELL", "BOTH"],
    session      = [True, False],
    regime       = ["counter", "any"],
)

PULLBACK_GRID = dict(
    h1_ema       = [21, 50, 100],
    m5_ema       = [21, 50],
    zone_atr     = [0.5, 0.75, 1.0, 1.5],
    hold         = [18, 24, 36, 48],
    session      = [True, False],
)

# M5 high-frequency grid — lower spike threshold, smaller cooldown baked into TF_SETTINGS
M5_HF_SPIKE_GRID = dict(
    h1_ema       = [14, 21, 50],
    spike_mult   = [1.5, 2.0, 2.5, 3.0],
    hold         = [6, 12, 18, 24],
    direction    = ["BUY", "SELL", "BOTH"],
    session      = [True, False],
    regime       = ["counter", "any"],
)

M5_HF_PULLBACK_GRID = dict(
    h1_ema       = [21, 50, 100],
    m5_ema       = [13, 21, 50],
    zone_atr     = [0.3, 0.5, 0.75, 1.0],
    hold         = [6, 12, 18, 24],
    session      = [True, False],
)

# M1 grids — shorter hold, always session-filtered, higher max_day
M1_SPIKE_GRID = dict(
    h1_ema       = [14, 21, 50],
    spike_mult   = [2.0, 2.5, 3.0, 3.5, 4.0],
    hold         = [12, 24, 36, 60, 120],   # M1 bars = 12 min to 2 hours
    direction    = ["BUY", "SELL", "BOTH"],
    session      = [True, False],
    regime       = ["counter", "any"],
)

M1_PULLBACK_GRID = dict(
    h1_ema       = [21, 50],
    m5_ema       = [8, 13, 21],
    zone_atr     = [0.5, 1.0, 1.5, 2.0],
    hold         = [12, 24, 36, 60],
    session      = [True, False],
)

# EMA crossover — M1 specific (fires much more often)
M1_CROSS_GRID = dict(
    h1_ema       = [21, 50],
    fast         = [5, 8, 13],
    slow         = [21, 34, 55],
    hold         = [12, 24, 36, 60],
    session      = [True, False],
    regime       = ["counter", "any"],
)


# ── DATA ─────────────────────────────────────────────────────────────────────

def available_symbols(tf: str = "M5") -> list:
    syms = []
    skip = ("BOOM","CRASH","R_","1HZ","STEP","RDBEAR","RDBULL",
            "WLDAUD","WLDEUR","WLDGBP","WLDXAU","JD")
    for p in sorted(DATA_DIR.glob(f"cache_*_{tf}.csv")):
        n = p.stem.replace("cache_","").replace(f"_{tf}","")
        if not any(n.startswith(s) for s in skip):
            syms.append(n)
    return syms


def load_raw(sym: str, gran: str) -> pd.DataFrame:
    path = DATA_DIR / f"cache_{sym}_{gran}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = df[c].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def build_merged(m5: pd.DataFrame, h1: pd.DataFrame, h1_ema_period: int) -> pd.DataFrame:
    h1 = h1.copy().sort_values("time").reset_index(drop=True)
    col = f"h1_ema{h1_ema_period}"
    h1[col] = h1["close"].ewm(span=h1_ema_period, adjust=False).mean()
    h1.rename(columns={"close":"h1_close"}, inplace=True)
    h1s = h1[["time","h1_close",col]].dropna().sort_values("time").reset_index(drop=True)
    m5 = m5.copy().sort_values("time").reset_index(drop=True)
    df = pd.merge_asof(m5, h1s, on="time", direction="backward")
    df = df.dropna(subset=["h1_close"]).reset_index(drop=True)

    # ATR
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-df["close"].shift(1)),
                    abs(df["low"] -df["close"].shift(1))))
    df["atr"] = tr.rolling(ATR_PERIOD).mean()

    # Weekend / after-session gap flag (skip first bar after gap > 6h)
    gap = (df["time"] - df["time"].shift(1)).dt.total_seconds() / 3600
    df["after_gap"] = gap > 6.0

    return df.dropna(subset=["atr", col]).reset_index(drop=True)


# ── CACHED MERGED DFS (one per h1_ema_period) ───────────────────────────────

def build_all_merged(m5: pd.DataFrame, h1: pd.DataFrame, periods: list) -> dict:
    return {p: build_merged(m5, h1, p) for p in periods}


# ── FORWARD SIMULATION (numpy, called per-trade) ────────────────────────────

def _sim_buy(entry, atr, h_arr, l_arr, c_arr, hold):
    sl = entry - atr
    size = 1.0; locked = 0.0; partial_done = False; peak = entry
    n = min(hold, len(h_arr))
    for i in range(n):
        lo, hi = l_arr[i], h_arr[i]
        if lo <= sl:
            return round(size * (-1.0) + locked, 3), "SL"
        if not partial_done and hi >= entry + atr * PARTIAL_R:
            locked += size * PARTIAL_PCT * PARTIAL_R
            size   *= (1 - PARTIAL_PCT)
            partial_done = True
        peak   = max(peak, hi)
        pr     = (peak - entry) / atr
        cm     = CHAND_TIERS[0][1]
        for mr, tm in CHAND_TIERS:
            if pr >= mr: cm = tm
        csl = peak - atr * cm
        if lo <= csl:
            return round(size * ((csl - entry) / atr) + locked, 3), "CHANDELIER"
    last = c_arr[n-1] if n > 0 else entry
    return round(size * ((last - entry) / atr) + locked, 3), "TIME"


def _sim_sell(entry, atr, h_arr, l_arr, c_arr, hold):
    sl = entry + atr
    size = 1.0; locked = 0.0; partial_done = False; trough = entry
    n = min(hold, len(h_arr))
    for i in range(n):
        lo, hi = l_arr[i], h_arr[i]
        if hi >= sl:
            return round(size * (-1.0) + locked, 3), "SL"
        if not partial_done and lo <= entry - atr * PARTIAL_R:
            locked += size * PARTIAL_PCT * PARTIAL_R
            size   *= (1 - PARTIAL_PCT)
            partial_done = True
        trough  = min(trough, lo)
        tr_r    = (entry - trough) / atr
        cm      = CHAND_TIERS[0][1]
        for mr, tm in CHAND_TIERS:
            if tr_r >= mr: cm = tm
        csl = trough + atr * cm
        if hi >= csl:
            return round(size * ((entry - csl) / atr) + locked, 3), "CHANDELIER"
    last = c_arr[n-1] if n > 0 else entry
    return round(size * ((entry - last) / atr) + locked, 3), "TIME"


# ── STRATEGY A: SPIKE REVERSION (vectorized) ─────────────────────────────────

def backtest_spike(df: pd.DataFrame, h1_ema_period: int,
                   spike_mult: float, hold: int,
                   direction: str, session_filter: bool,
                   regime_mode: str,
                   cooldown: int = 8, max_day: int = 4) -> pd.DataFrame:

    col = f"h1_ema{h1_ema_period}"
    if col not in df.columns:
        return pd.DataFrame()

    N = len(df)
    op  = df["open"].values
    hi  = df["high"].values
    lo  = df["low"].values
    cl  = df["close"].values
    atr = df["atr"].values
    h1c = df["h1_close"].values
    h1e = df[col].values
    hrs = df["time"].dt.hour.values
    agp = df["after_gap"].values.astype(bool)
    dts = df["time"].dt.date.astype(str).values
    mth = df["time"].dt.tz_localize(None).dt.to_period("M").astype(str).values

    # Vectorized spike detection
    body_dn = op - cl                # bearish body
    body_up = cl - op                # bullish body
    h1_bull = h1c > h1e
    h1_bear = h1c < h1e

    base = (~agp) & (atr > 0)
    if session_filter:
        base &= np.isin(hrs, list(ACTIVE_HOURS))

    buy_mask  = base & (body_dn > spike_mult * atr)
    sell_mask = base & (body_up > spike_mult * atr)

    if direction == "BUY":
        sell_mask[:] = False
    elif direction == "SELL":
        buy_mask[:] = False

    if regime_mode == "counter":
        buy_mask  &= ~h1_bear     # only fade down spike when H1 is NOT bearish
        sell_mask &= ~h1_bull     # only fade up spike when H1 is NOT bullish

    # Merge signals in time order
    buy_idxs  = np.where(buy_mask)[0]
    sell_idxs = np.where(sell_mask)[0]
    all_sigs  = sorted(
        [(i, "BUY")  for i in buy_idxs] +
        [(i, "SELL") for i in sell_idxs]
    )

    traded: set = set()
    tpd: dict   = {}
    lsi: int    = -999
    trades: list = []

    for idx, sig in all_sigs:
        if idx + hold + 1 >= N:
            break
        if idx in traded:
            continue
        if (idx - lsi) <= cooldown:
            lsi = idx
            continue
        lsi = idx

        date = dts[idx]
        if tpd.get(date, 0) >= max_day:
            continue

        entry    = float(op[idx + 1])   # fill next bar open
        atr_val  = float(atr[idx])
        fwd_end  = min(idx + 1 + hold, N)
        fwd_h    = hi[idx+1:fwd_end]
        fwd_l    = lo[idx+1:fwd_end]
        fwd_c    = cl[idx+1:fwd_end]

        if len(fwd_h) < 4:
            continue

        if sig == "BUY":
            r, reason = _sim_buy(entry, atr_val, fwd_h, fwd_l, fwd_c, hold)
        else:
            r, reason = _sim_sell(entry, atr_val, fwd_h, fwd_l, fwd_c, hold)

        tpd[date] = tpd.get(date, 0) + 1
        trades.append({
            "month":  mth[idx],
            "date":   date,
            "dir":    sig,
            "r":      r,
            "result": "W" if r > 0 else "L",
            "exit":   reason,
            "spike":  round(float(body_dn[idx] if sig=="BUY" else body_up[idx]) / atr_val, 2),
        })
        for k in range(idx, fwd_end):
            traded.add(k)

    return pd.DataFrame(trades)


# ── STRATEGY B: EMA PULLBACK (vectorized) ────────────────────────────────────

def backtest_pullback(df: pd.DataFrame, h1_ema_period: int,
                      m5_ema_period: int, zone_atr: float,
                      hold: int, session_filter: bool,
                      cooldown: int = 8, max_day: int = 4) -> pd.DataFrame:

    col = f"h1_ema{h1_ema_period}"
    if col not in df.columns:
        return pd.DataFrame()

    N = len(df)
    op  = df["open"].values
    hi  = df["high"].values
    lo  = df["low"].values
    cl  = df["close"].values
    atr = df["atr"].values
    h1c = df["h1_close"].values
    h1e = df[col].values
    hrs = df["time"].dt.hour.values
    agp = df["after_gap"].values.astype(bool)
    dts = df["time"].dt.date.astype(str).values
    mth = df["time"].dt.tz_localize(None).dt.to_period("M").astype(str).values

    # M5 EMA for pullback zone
    m5_ema = pd.Series(cl).ewm(span=m5_ema_period, adjust=False).mean().values

    h1_bull = h1c > h1e
    h1_bear = h1c < h1e
    bull_candle = cl > op
    bear_candle = cl < op
    near_ema    = np.abs(cl - m5_ema) <= zone_atr * atr

    base = (~agp) & (atr > 0) & (~np.isnan(m5_ema))
    if session_filter:
        base &= np.isin(hrs, list(ACTIVE_HOURS))

    buy_mask  = base & h1_bull & near_ema & bull_candle
    sell_mask = base & h1_bear & near_ema & bear_candle

    buy_idxs  = np.where(buy_mask)[0]
    sell_idxs = np.where(sell_mask)[0]
    all_sigs  = sorted(
        [(i, "BUY")  for i in buy_idxs] +
        [(i, "SELL") for i in sell_idxs]
    )

    traded: set = set()
    tpd: dict   = {}
    trades: list = []

    for idx, sig in all_sigs:
        if idx + hold + 1 >= N:
            break
        if idx in traded:
            continue

        date = dts[idx]
        if tpd.get(date, 0) >= max_day:
            continue

        entry   = float(op[idx + 1])
        atr_val = float(atr[idx])
        fwd_end = min(idx + 1 + hold, N)
        fwd_h   = hi[idx+1:fwd_end]
        fwd_l   = lo[idx+1:fwd_end]
        fwd_c   = cl[idx+1:fwd_end]

        if len(fwd_h) < 4:
            continue

        if sig == "BUY":
            r, reason = _sim_buy(entry, atr_val, fwd_h, fwd_l, fwd_c, hold)
        else:
            r, reason = _sim_sell(entry, atr_val, fwd_h, fwd_l, fwd_c, hold)

        tpd[date] = tpd.get(date, 0) + 1
        trades.append({
            "month":  mth[idx],
            "date":   date,
            "dir":    sig,
            "r":      r,
            "result": "W" if r > 0 else "L",
            "exit":   reason,
        })
        for k in range(idx, fwd_end):
            traded.add(k)

    return pd.DataFrame(trades)


# ── STATS ─────────────────────────────────────────────────────────────────────

def compute_stats(trades: pd.DataFrame, min_trades: int = 15) -> dict:
    if trades.empty or len(trades) < min_trades:
        return {}
    n   = len(trades)
    win = trades[trades["r"] > 0]
    los = trades[trades["r"] <= 0]
    wr  = len(win) / n
    gw  = win["r"].sum()
    gl  = abs(los["r"].sum())
    pf  = gw / gl if gl > 0 else float("inf")
    avg = trades["r"].mean()
    months = trades["month"].nunique()
    oos_pf = 0.0
    oos_start = int(n * 0.75)
    oos = trades.iloc[oos_start:]
    if len(oos) >= 5:
        ow = oos[oos["r"] > 0]["r"].sum()
        ol = abs(oos[oos["r"] <= 0]["r"].sum())
        oos_pf = ow / ol if ol > 0 else float("inf")
    return {"n": n, "wr": wr, "pf": pf, "avg_r": avg,
            "oos_pf": oos_pf, "months": months, "tpm": n / months}


# ── SWEEPS ────────────────────────────────────────────────────────────────────

def sweep_spike(dfs_by_ema: dict, cooldown: int = 8, max_day: int = 4,
                min_trades: int = 15, grid: dict = None) -> list:
    results = []
    g = grid if grid is not None else SPIKE_GRID
    combos = list(itertools.product(
        g["h1_ema"], g["spike_mult"], g["hold"],
        g["direction"], g["session"], g["regime"]
    ))
    total = len(combos)
    for i, (h1_ema, sm, hold, dirn, sess, regime) in enumerate(combos):
        if (i+1) % 200 == 0:
            print(f"    spike {i+1}/{total} ({(i+1)/total*100:.0f}%)", flush=True)
        df = dfs_by_ema.get(h1_ema)
        if df is None or df.empty:
            continue
        trades = backtest_spike(df, h1_ema, sm, hold, dirn, sess, regime, cooldown, max_day)
        st = compute_stats(trades, min_trades)
        if st:
            results.append({"strategy":"SPIKE","h1_ema":h1_ema,"spike_mult":sm,
                             "hold":hold,"dir":dirn,"session":sess,"regime":regime,**st})
    return results


def sweep_pullback(dfs_by_ema: dict, cooldown: int = 8, max_day: int = 4,
                   min_trades: int = 15, grid: dict = None) -> list:
    results = []
    g = grid if grid is not None else PULLBACK_GRID
    combos = list(itertools.product(
        g["h1_ema"], g["m5_ema"], g["zone_atr"], g["hold"], g["session"]
    ))
    total = len(combos)
    for i, (h1_ema, m5_ema, zone, hold, sess) in enumerate(combos):
        if (i+1) % 50 == 0:
            print(f"    pullback {i+1}/{total} ({(i+1)/total*100:.0f}%)", flush=True)
        df = dfs_by_ema.get(h1_ema)
        if df is None or df.empty:
            continue
        trades = backtest_pullback(df, h1_ema, m5_ema, zone, hold, sess, cooldown, max_day)
        st = compute_stats(trades, min_trades)
        if st:
            results.append({"strategy":"PULLBACK","h1_ema":h1_ema,"m5_ema":m5_ema,
                             "zone_atr":zone,"hold":hold,"session":sess,
                             "spike_mult":0,"dir":"—","regime":"—",**st})
    return results


# ── STRATEGY C: EMA CROSSOVER (M1 high-frequency) ────────────────────────────

def backtest_ema_cross(df: pd.DataFrame, h1_ema_period: int,
                       fast: int, slow: int,
                       hold: int, session_filter: bool,
                       regime_mode: str,
                       cooldown: int, max_day: int) -> pd.DataFrame:
    """
    EMA crossover: BUY when fast crosses above slow (in H1 bull regime).
    SELL when fast crosses below slow (in H1 bear regime).
    """
    col = f"h1_ema{h1_ema_period}"
    if col not in df.columns:
        return pd.DataFrame()

    N   = len(df)
    op  = df["open"].values
    hi  = df["high"].values
    lo  = df["low"].values
    cl  = df["close"].values
    atr = df["atr"].values
    h1c = df["h1_close"].values
    h1e = df[col].values
    hrs = df["time"].dt.hour.values
    agp = df["after_gap"].values.astype(bool)
    dts = df["time"].dt.date.astype(str).values
    mth = df["time"].dt.tz_localize(None).dt.to_period("M").astype(str).values

    ema_f = pd.Series(cl).ewm(span=fast,  adjust=False).mean().values
    ema_s = pd.Series(cl).ewm(span=slow,  adjust=False).mean().values

    # Crossover detection (vectorized)
    cross_up   = np.concatenate([[False], (ema_f[1:] > ema_s[1:]) & (ema_f[:-1] <= ema_s[:-1])])
    cross_down = np.concatenate([[False], (ema_f[1:] < ema_s[1:]) & (ema_f[:-1] >= ema_s[:-1])])

    h1_bull = h1c > h1e
    h1_bear = h1c < h1e

    base = (~agp) & (atr > 0) & ~np.isnan(ema_f) & ~np.isnan(ema_s)
    if session_filter:
        base &= np.isin(hrs, list(ACTIVE_HOURS))

    buy_mask  = base & cross_up
    sell_mask = base & cross_down

    if regime_mode == "counter":
        buy_mask  &= h1_bull   # cross up during H1 bull (trend continuation)
        sell_mask &= h1_bear
    # "any" = no regime filter

    buy_idxs  = np.where(buy_mask)[0]
    sell_idxs = np.where(sell_mask)[0]
    all_sigs  = sorted([(i,"BUY") for i in buy_idxs] + [(i,"SELL") for i in sell_idxs])

    traded: set = set()
    tpd: dict   = {}
    lsi: int    = -999
    trades: list = []

    for idx, sig in all_sigs:
        if idx + hold + 1 >= N:
            break
        if idx in traded:
            continue
        if (idx - lsi) <= cooldown:
            lsi = idx
            continue
        lsi = idx

        date = dts[idx]
        if tpd.get(date, 0) >= max_day:
            continue

        entry   = float(op[idx + 1])
        atr_val = float(atr[idx])
        if atr_val <= 0:
            continue

        fwd_end = min(idx + 1 + hold, N)
        fwd_h   = hi[idx+1:fwd_end]
        fwd_l   = lo[idx+1:fwd_end]
        fwd_c   = cl[idx+1:fwd_end]

        if len(fwd_h) < 4:
            continue

        r, reason = _sim_buy(entry, atr_val, fwd_h, fwd_l, fwd_c, hold) if sig == "BUY" \
                    else _sim_sell(entry, atr_val, fwd_h, fwd_l, fwd_c, hold)

        tpd[date] = tpd.get(date, 0) + 1
        trades.append({"month": mth[idx], "date": date, "dir": sig,
                        "r": r, "result": "W" if r > 0 else "L", "exit": reason})
        for k in range(idx, fwd_end):
            traded.add(k)

    return pd.DataFrame(trades)


def sweep_cross(dfs_by_ema: dict, cooldown: int, max_day: int, min_trades: int) -> list:
    results = []
    g = M1_CROSS_GRID
    combos = list(itertools.product(g["h1_ema"], g["fast"], g["slow"],
                                    g["hold"], g["session"], g["regime"]))
    total = len(combos)
    for i, (h1_ema, fast, slow, hold, sess, regime) in enumerate(combos):
        if fast >= slow:
            continue
        if (i+1) % 100 == 0:
            print(f"    cross {i+1}/{total} ({(i+1)/total*100:.0f}%)", flush=True)
        df = dfs_by_ema.get(h1_ema)
        if df is None or df.empty:
            continue
        trades = backtest_ema_cross(df, h1_ema, fast, slow, hold, sess, regime, cooldown, max_day)
        st = compute_stats(trades, min_trades)
        if st:
            results.append({"strategy":"CROSS","h1_ema":h1_ema,"fast":fast,"slow":slow,
                             "hold":hold,"session":sess,"regime":regime,
                             "spike_mult":0,"dir":"—","m5_ema":0,"zone_atr":0,**st})
    return results


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def print_rankings(results: list, top_n: int = 25):
    if not results:
        print("  No configs met the minimum trade threshold.")
        return
    df = pd.DataFrame(results).sort_values("pf", ascending=False)
    print(f"\n  Top {min(top_n, len(df))} by Profit Factor:")
    hdr = f"  {'Strat':<8} {'Params':<54} {'N':>5} {'WR%':>6} {'PF':>7} {'OOS_PF':>8} {'AvgR':>7} {'T/mo':>6}"
    print(hdr)
    print("  " + "-" * 102)
    for _, row in df.head(top_n).iterrows():
        strat = row["strategy"]
        if strat == "SPIKE":
            p = (f"spike={row['spike_mult']:.1f} dir={row['dir']:<4} "
                 f"hold={int(row['hold'])} h1ema={int(row['h1_ema'])} "
                 f"sess={'Y' if row['session'] else 'N'} reg={row['regime']}")
        elif strat == "CROSS":
            p = (f"ema{int(row['fast'])}/{int(row['slow'])} "
                 f"hold={int(row['hold'])} h1ema={int(row['h1_ema'])} "
                 f"sess={'Y' if row['session'] else 'N'} reg={row['regime']}")
        else:
            p = (f"m5ema={int(row['m5_ema'])} zone={row['zone_atr']:.2f} "
                 f"hold={int(row['hold'])} h1ema={int(row['h1_ema'])} "
                 f"sess={'Y' if row['session'] else 'N'}")
        pf_s  = f"{row['pf']:.3f}"     if row['pf']     != float("inf") else "   inf"
        oos_s = f"{row['oos_pf']:.3f}" if row['oos_pf'] != float("inf") else "   inf"
        print(f"  {strat:<8} {p:<54} {int(row['n']):>5} "
              f"{row['wr']*100:>5.1f}% {pf_s:>7} {oos_s:>8} "
              f"{row['avg_r']:>+7.3f} {row['tpm']:>6.1f}", flush=True)


def print_monthly(trades: pd.DataFrame, label: str):
    if trades.empty:
        return
    print(f"\n  Monthly breakdown — {label}")
    print(f"  {'Month':<10} {'N':>5} {'WR%':>6} {'PF':>7} {'AvgR':>7}")
    print("  " + "-" * 42)
    for month, g in trades.groupby("month"):
        n = len(g); w = (g["r"] > 0).sum()
        wr = w / n * 100
        gw = g[g["r"]>0]["r"].sum(); gl = abs(g[g["r"]<=0]["r"].sum())
        pf = gw/gl if gl>0 else float("inf")
        pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
        print(f"  {str(month):<10} {n:>5} {wr:>5.0f}% {pf_s:>7} {g['r'].mean():>+7.3f}")
    n = len(trades); w = (trades["r"]>0).sum()
    gw = trades[trades["r"]>0]["r"].sum(); gl = abs(trades[trades["r"]<=0]["r"].sum())
    pf = gw/gl if gl>0 else float("inf")
    pf_s = f"{pf:.2f}" if pf != float("inf") else "  inf"
    print("  " + "-" * 42)
    print(f"  {'TOTAL':<10} {n:>5} {w/n*100:>5.0f}% {pf_s:>7} {trades['r'].mean():>+7.3f}")
    if "exit" in trades.columns:
        print()
        print("  Exit reasons:")
        for ex, cnt in trades["exit"].value_counts().items():
            ar = trades[trades["exit"]==ex]["r"].mean()
            print(f"    {ex:<14} {cnt:>4}  avg R {ar:>+.3f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run_symbol(name: str, tf: str = "M5", family: str = "both"):
    cfg = TF_SETTINGS[tf]
    cooldown   = cfg["cooldown"]
    max_day    = cfg["max_day"]
    min_trades = cfg["min_trades"]

    # M5_HF uses M5 data files but with high-frequency grids
    data_tf = "M5" if tf == "M5_HF" else tf

    print(f"\n{'='*80}")
    print(f"  {name}  [{tf}]", flush=True)
    print(f"{'='*80}")

    lf = load_raw(name, data_tf)
    h1 = load_raw(name, "H1")
    if lf.empty or h1.empty:
        missing = data_tf if lf.empty else "H1"
        print(f"  No {missing} data. Run: python fetch_commodity_data.py {name} {missing}")
        return

    print(f"  {data_tf}: {len(lf):,} bars | {lf['time'].min().date()} to {lf['time'].max().date()}")
    print(f"  H1: {len(h1):,} bars | {h1['time'].min().date()} to {h1['time'].max().date()}", flush=True)

    # Select grids based on TF
    if tf == "M1":
        spike_grid    = M1_SPIKE_GRID
        pullback_grid = M1_PULLBACK_GRID
    elif tf == "M5_HF":
        spike_grid    = M5_HF_SPIKE_GRID
        pullback_grid = M5_HF_PULLBACK_GRID
    else:
        spike_grid    = SPIKE_GRID
        pullback_grid = PULLBACK_GRID

    # Collect all unique H1 EMA periods needed
    ema_periods = set()
    if family in ("spike", "both"):
        ema_periods.update(spike_grid["h1_ema"])
    if family in ("pullback", "both"):
        ema_periods.update(pullback_grid["h1_ema"])
    if family in ("cross", "both") and tf == "M1":
        ema_periods.update(M1_CROSS_GRID["h1_ema"])

    print(f"\n  Building indicator sets for H1 EMA periods: {sorted(ema_periods)}...", flush=True)
    dfs_by_ema = {p: build_merged(lf, h1, p) for p in ema_periods}
    for p, d in dfs_by_ema.items():
        print(f"    EMA{p}: {len(d):,} merged bars", flush=True)

    all_results = []

    if family in ("spike", "both"):
        total_spike = len(list(itertools.product(*[spike_grid[k] for k in spike_grid])))
        print(f"\n  Spike reversion sweep ({total_spike} configs)...", flush=True)
        sr = sweep_spike(dfs_by_ema, cooldown, max_day, min_trades, spike_grid)
        all_results.extend(sr)
        print(f"  Done — {len(sr)} configs passed {min_trades}-trade minimum.", flush=True)

    if family in ("pullback", "both"):
        total_pb = len(list(itertools.product(*[pullback_grid[k] for k in pullback_grid])))
        print(f"\n  EMA pullback sweep ({total_pb} configs)...", flush=True)
        pr = sweep_pullback(dfs_by_ema, cooldown, max_day, min_trades, pullback_grid)
        all_results.extend(pr)
        print(f"  Done — {len(pr)} configs passed {min_trades}-trade minimum.", flush=True)

    if family in ("cross", "both") and data_tf == "M1":
        total_cr = len(list(itertools.product(*[M1_CROSS_GRID[k] for k in M1_CROSS_GRID])))
        print(f"\n  EMA crossover sweep ({total_cr} configs)...", flush=True)
        cr = sweep_cross(dfs_by_ema, cooldown, max_day, min_trades)
        all_results.extend(cr)
        print(f"  Done — {len(cr)} configs passed {min_trades}-trade minimum.", flush=True)

    if not all_results:
        print(f"\n  No configs reached {min_trades} trades. Not enough data or no edge found.")
        return

    print_rankings(all_results, top_n=25)

    # Detailed monthly breakdown for best config
    best = max(all_results, key=lambda x: x["pf"])
    print(f"\n  Best: {best['strategy']} | PF={best['pf']:.3f} | WR={best['wr']*100:.1f}% | "
          f"OOS={best['oos_pf']:.3f} | {int(best['n'])} trades | {best['tpm']:.1f}/month")

    df_best = dfs_by_ema[best["h1_ema"]]
    strat = best["strategy"]
    if strat == "SPIKE":
        bt = backtest_spike(df_best, best["h1_ema"], best["spike_mult"],
                            int(best["hold"]), best["dir"], best["session"],
                            best["regime"], cooldown, max_day)
    elif strat == "CROSS":
        bt = backtest_ema_cross(df_best, best["h1_ema"], int(best["fast"]),
                                int(best["slow"]), int(best["hold"]),
                                best["session"], best["regime"], cooldown, max_day)
    else:
        bt = backtest_pullback(df_best, best["h1_ema"], int(best["m5_ema"]),
                               best["zone_atr"], int(best["hold"]),
                               best["session"], cooldown, max_day)

    print_monthly(bt, f"{name} — {best['strategy']}")

    # Save outputs
    tag = f"{name}_{tf}"
    bt.to_csv(DATA_DIR / f"commodity_best_{tag}.csv", index=False)
    pd.DataFrame(all_results).sort_values("pf", ascending=False).to_csv(
        DATA_DIR / f"commodity_rankings_{tag}.csv", index=False)
    print(f"\n  Saved: commodity_best_{tag}.csv  |  commodity_rankings_{tag}.csv", flush=True)


def main():
    args = sys.argv[1:]
    sym_filter = args[0].upper() if args else None
    tf         = args[1].upper() if len(args) > 1 else "M5"
    family     = args[2].lower() if len(args) > 2 else "both"

    # Handle "python commodity_backtest.py XAUUSD spike" (no TF given)
    if tf and tf not in TF_SETTINGS:
        family = tf.lower()
        tf = "M5"

    if tf not in TF_SETTINGS:
        print(f"  Unknown timeframe '{tf}'. Choose M5, M5_HF, or M1.")
        return

    min_trades = TF_SETTINGS[tf]["min_trades"]
    data_tf    = "M5" if tf == "M5_HF" else tf   # M5_HF reads M5 files

    print()
    print("=" * 80)
    print("  COMMODITY STRATEGY RESEARCH ENGINE  (vectorized)")
    print(f"  TF: {tf} | Min trades: {min_trades} | Chandelier exit + 50% partial @ 2R")
    print("=" * 80, flush=True)

    syms = available_symbols(data_tf)
    if not syms:
        print(f"\n  No {data_tf} data in data/. Run: python fetch_commodity_data.py all {data_tf}")
        return

    print(f"\n  Available [{data_tf}]: {syms}", flush=True)

    if sym_filter and sym_filter not in ("ALL", ""):
        if sym_filter not in syms:
            print(f"\n  '{sym_filter}' not found in {tf} data. Available: {syms}")
            return
        syms = [sym_filter]

    for sym in syms:
        run_symbol(sym, tf, family)

    print("\n  Research complete.")


if __name__ == "__main__":
    main()

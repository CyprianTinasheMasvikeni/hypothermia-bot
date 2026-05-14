"""
Forex Backtest -- All Deriv Forex Pairs
2 years of M15 data | Two strategies tested on every pair

Strategy 1: step_trend  -- Our proven EMA stack + pullback + confirmation
Strategy 2: london_breakout -- Asian range break at London open (8:00 UTC)

Session filter: Monday 00:00 - Friday 22:00 UTC (no weekend candles)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import websockets

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "forex_cache"
CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
import strategy_step_trend as strategy

import yfinance as yf

# yfinance symbol map -> Deriv symbol
FOREX_PAIRS = {
    "EURUSD=X": "frxEURUSD", "GBPUSD=X": "frxGBPUSD", "AUDUSD=X": "frxAUDUSD",
    "USDJPY=X": "frxUSDJPY", "USDCAD=X": "frxUSDCAD", "USDCHF=X": "frxUSDCHF",
    "EURGBP=X": "frxEURGBP", "EURJPY=X": "frxEURJPY", "EURCAD=X": "frxEURCAD",
    "EURCHF=X": "frxEURCHF", "GBPJPY=X": "frxGBPJPY", "GBPAUD=X": "frxGBPAUD",
    "EURAUD=X": "frxEURAUD", "AUDJPY=X": "frxAUDJPY", "AUDCAD=X": "frxAUDCAD",
    "AUDCHF=X": "frxAUDCHF", "AUDNZD=X": "frxAUDNZD", "EURNZD=X": "frxEURNZD",
    "GBPCAD=X": "frxGBPCAD", "GBPCHF=X": "frxGBPCHF", "GBPNZD=X": "frxGBPNZD",
    "NZDJPY=X": "frxNZDJPY", "NZDUSD=X": "frxNZDUSD",
}

# Risk / exit settings (same as live bots)
MULTIPLIER        = 100
SL_ATR_MULT       = 1.0
CHANDELIER_TIERS  = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R         = 2.0
PARTIAL_PCT       = 0.50
RISK_PCT          = 0.05
MAX_TRADES_DAY    = 6
DAILY_DD_LIMIT    = 0.03
ACCOUNT_DD_LIMIT  = 0.15
START_BALANCE     = 10_000.0

# Session: London + NY (8:00-20:00 UTC), skip weekends
SESSION_START = 8
SESSION_END   = 20

# London breakout params
ASIAN_START_H = 22   # previous day 22:00 UTC
ASIAN_END_H   = 8    # 8:00 UTC = London open
LBO_WINDOW_H  = 12   # only take LBO entries up to 12:00 UTC
LBO_MIN_RANGE_ATR = 0.5   # Asian range must be >= 0.5x ATR (avoid flat nights)
LBO_SL_MULT   = 1.0       # SL = 1x ATR below entry
LBO_TP_MULT   = 2.0       # TP = 2x ATR above entry (fixed TP for LBO)


# ── DATA (yfinance — 2 years H1) ──────────────────────────────────────────────
def load_h1(yf_symbol: str, refresh: bool = False) -> pd.DataFrame | None:
    label = yf_symbol.replace("=X", "")
    cache = CACHE_DIR / f"forex_{label}_H1.csv"
    if not refresh and cache.exists():
        df = pd.read_csv(cache, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        return df.sort_values("time").reset_index(drop=True)
    try:
        raw = yf.download(yf_symbol, period="730d", interval="1h",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() for col in raw.columns]
        else:
            raw.columns = [col.lower() for col in raw.columns]
        raw = raw.reset_index()
        time_col = [c for c in raw.columns if "datetime" in c.lower() or "date" in c.lower()][0]
        df = pd.DataFrame({
            "time":  pd.to_datetime(raw[time_col], utc=True),
            "open":  raw["open"].astype(float).values,
            "high":  raw["high"].astype(float).values,
            "low":   raw["low"].astype(float).values,
            "close": raw["close"].astype(float).values,
        })
        df = df.dropna().sort_values("time").reset_index(drop=True)
        df.to_csv(cache, index=False)
        return df
    except Exception as e:
        print(f"[err: {e}]", end=" ")
        return None


def drop_weekends(df: pd.DataFrame) -> pd.DataFrame:
    # weekday(): Mon=0 ... Sun=6
    mask = df["time"].dt.weekday < 5  # keep Mon-Fri
    # also drop Friday after 21:00 and Sunday before 22:00
    fri_late = (df["time"].dt.weekday == 4) & (df["time"].dt.hour >= 21)
    sun_early = (df["time"].dt.weekday == 6) & (df["time"].dt.hour < 22)
    return df[mask & ~fri_late & ~sun_early].reset_index(drop=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def chandelier_mult(peak_r: float) -> float:
    m = CHANDELIER_TIERS[0][1]
    for min_r, tier_m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            m = tier_m
    return m


def calc_stake(atr, entry, risk):
    sf = MULTIPLIER * SL_ATR_MULT * atr / entry
    return max(1.0, round(risk / sf, 2)) if sf > 0 else max(1.0, round(risk, 2))


def calc_pnl(direction, stake, entry, exit_p):
    d = 1 if direction == "BUY" else -1
    return stake * MULTIPLIER * d * (exit_p - entry) / entry


def run_sim(signals: list, start_balance: float) -> dict:
    """
    Generic simulator given a list of signal dicts:
    {idx, direction, entry, atr, date}
    Returns summary stats.
    """
    balance   = start_balance
    peak_bal  = start_balance
    wins = losses = 0
    trades = []
    peak_r_list = []
    r_list = []

    for sig in signals:
        if balance < peak_bal * (1 - ACCOUNT_DD_LIMIT):
            break
        entry = sig["entry"]
        atr   = sig["atr"]
        dirn  = sig["direction"]
        d     = 1 if dirn == "BUY" else -1
        risk  = balance * RISK_PCT
        stake = calc_stake(atr, entry, risk)
        sl    = entry - d * atr * SL_ATR_MULT

        # Simulate using OHLC from the candles after entry
        df_fwd   = sig["df_fwd"]   # forward candles
        partial_done = False
        locked_pnl   = 0.0
        peak_price   = entry
        peak_r_val   = 0.0
        current_stake = stake
        result        = None
        exit_price    = None

        for _, row in df_fwd.iterrows():
            hi = row["high"]
            lo = row["low"]

            # SL check
            sl_hit = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            if sl_hit:
                pnl = calc_pnl(dirn, current_stake * (1 - PARTIAL_PCT if partial_done else 1.0),
                               entry, sl)
                pnl += locked_pnl
                result = "LOSS"
                exit_price = sl
                break

            # Partial close at 2R
            if not partial_done:
                partial_price = entry + d * atr * PARTIAL_R
                if (d == 1 and hi >= partial_price) or (d == -1 and lo <= partial_price):
                    locked_pnl   = calc_pnl(dirn, stake * PARTIAL_PCT, entry, partial_price)
                    current_stake = stake * (1 - PARTIAL_PCT)
                    partial_done  = True

            # Update peak
            if d == 1:
                peak_price = max(peak_price, hi)
            else:
                peak_price = min(peak_price, lo)
            peak_r_val = abs(peak_price - entry) / atr if atr > 0 else 0

            # Chandelier
            cm       = chandelier_mult(peak_r_val)
            chand_sl = peak_price - d * atr * cm
            chand_hit = (d == 1 and lo <= chand_sl) or (d == -1 and hi >= chand_sl)
            if chand_hit:
                pnl = calc_pnl(dirn, current_stake, entry, chand_sl)
                pnl += locked_pnl
                result = "WIN"
                exit_price = chand_sl
                break
        else:
            # time exit
            last_close = df_fwd.iloc[-1]["close"]
            pnl = calc_pnl(dirn, current_stake, entry, last_close)
            pnl += locked_pnl
            result = "WIN" if pnl > 0 else "LOSS"
            exit_price = last_close

        balance += pnl
        peak_bal = max(peak_bal, balance)
        if result == "WIN":
            wins += 1
        else:
            losses += 1
        r_val = pnl / risk if risk > 0 else 0
        r_list.append(r_val)
        peak_r_list.append(peak_r_val)
        trades.append(pnl)

    total = wins + losses
    if total == 0:
        return None
    wr = wins / total
    avg_r = np.mean(r_list) if r_list else 0
    gross_w = sum(p for p in trades if p > 0)
    gross_l = abs(sum(p for p in trades if p < 0))
    pf = gross_w / gross_l if gross_l > 0 else 0
    ret = (balance / start_balance - 1) * 100
    peak = start_balance
    max_dd = 0.0
    bal = start_balance
    for t in trades:
        bal += t
        peak = max(peak, bal)
        max_dd = max(max_dd, (peak - bal) / peak)
    return {
        "trades": total, "wins": wins, "wr": wr, "pf": pf,
        "avg_r": avg_r, "ret": ret, "max_dd": max_dd * 100,
        "balance": balance,
    }


# ── STRATEGY 1: step_trend ────────────────────────────────────────────────────
def backtest_step_trend(df: pd.DataFrame) -> dict | None:
    ind = strategy.calculate_indicators(df.copy())
    ind = ind.dropna().reset_index(drop=True)
    if len(ind) < 250:
        return None

    WARMUP = 250
    signals = []
    last_sig_idx = -1

    for i in range(WARMUP, len(ind) - 1):
        row = ind.iloc[i]
        hour = row["time"].hour
        if not (SESSION_START <= hour < SESSION_END):
            continue
        if i == last_sig_idx:
            continue

        res = strategy.analyze_setup(ind.iloc[:i + 1])
        sig = res.get("signal", "WAIT")
        if sig not in ("BUY", "SELL"):
            continue

        atr = float(row["atr"])
        if atr <= 0:
            continue

        entry = float(ind.iloc[i + 1]["open"])
        # forward candles for simulation (max 96 candles = 1 day M15)
        fwd = ind.iloc[i + 1: i + 97].copy()
        if len(fwd) < 4:
            continue

        signals.append({
            "entry": entry, "atr": atr, "direction": sig,
            "date": row["time"], "df_fwd": fwd,
        })
        last_sig_idx = i + 1

    return run_sim(signals, START_BALANCE)


# ── STRATEGY 2: London Breakout ───────────────────────────────────────────────
def backtest_london_breakout(df: pd.DataFrame) -> dict | None:
    ind = strategy.calculate_indicators(df.copy())
    ind = ind.dropna().reset_index(drop=True)
    if len(ind) < 200:
        return None

    signals = []
    dates_traded = set()

    for i in range(200, len(ind) - 1):
        row = ind.iloc[i]
        hour = row["time"].hour
        date = row["time"].date()

        # Only enter during London open window
        if not (SESSION_START <= hour < LBO_WINDOW_H):
            continue
        if date in dates_traded:
            continue

        atr = float(row["atr"])
        if atr <= 0:
            continue

        # Define Asian range: candles between 22:00 yesterday and 8:00 today
        t = row["time"]
        asian_start = t.normalize() + pd.Timedelta(hours=ASIAN_START_H) - pd.Timedelta(days=1)
        asian_end   = t.normalize() + pd.Timedelta(hours=ASIAN_END_H)
        asian = ind[(ind["time"] >= asian_start) & (ind["time"] < asian_end)]
        if len(asian) < 4:
            continue

        asian_high = asian["high"].max()
        asian_low  = asian["low"].min()
        asian_range = asian_high - asian_low

        # Reject flat Asian sessions
        if asian_range < atr * LBO_MIN_RANGE_ATR:
            continue

        # Check if current candle breaks the Asian range
        close = float(row["close"])
        dirn = None
        if close > asian_high:
            dirn = "BUY"
        elif close < asian_low:
            dirn = "SELL"
        else:
            continue

        entry = float(ind.iloc[i + 1]["open"])
        fwd   = ind.iloc[i + 1: i + 65].copy()  # max 16 hours
        if len(fwd) < 4:
            continue

        signals.append({
            "entry": entry, "atr": atr, "direction": dirn,
            "date": date, "df_fwd": fwd,
        })
        dates_traded.add(date)

    return run_sim(signals, START_BALANCE)


# ── STRATEGY 3: EMA Trend + Session ──────────────────────────────────────────
def backtest_ema_trend(df: pd.DataFrame) -> dict | None:
    """
    Simpler trend: EMA20>EMA50, price pulls back to EMA20, bullish close.
    London + NY session only.
    """
    d = df.copy()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss))
    d["tr"] = np.maximum(d["high"] - d["low"],
               np.maximum(abs(d["high"] - d["close"].shift(1)),
                          abs(d["low"] - d["close"].shift(1))))
    d["atr"] = d["tr"].rolling(14).mean()
    d = d.dropna().reset_index(drop=True)
    if len(d) < 250:
        return None

    signals = []
    last_sig_idx = -1

    for i in range(220, len(d) - 1):
        row = d.iloc[i]
        hour = row["time"].hour
        if not (SESSION_START <= hour < SESSION_END):
            continue
        if i == last_sig_idx:
            continue

        atr = float(row["atr"])
        if atr <= 0:
            continue

        bullish_stack = row["ema20"] > row["ema50"] > row["ema200"]
        bearish_stack = row["ema20"] < row["ema50"] < row["ema200"]
        pullback_buy  = abs(row["close"] - row["ema20"]) <= atr * 1.5
        pullback_sell = abs(row["close"] - row["ema20"]) <= atr * 1.5
        bull_candle   = row["close"] > row["open"]
        bear_candle   = row["close"] < row["open"]
        body          = abs(row["close"] - row["open"]) / (row["high"] - row["low"]) \
                        if row["high"] > row["low"] else 0
        rsi_bull = row["rsi"] > 50
        rsi_bear = row["rsi"] < 50

        dirn = None
        if bullish_stack and pullback_buy and bull_candle and body >= 0.45 and rsi_bull:
            dirn = "BUY"
        elif bearish_stack and pullback_sell and bear_candle and body >= 0.45 and rsi_bear:
            dirn = "SELL"
        if dirn is None:
            continue

        entry = float(d.iloc[i + 1]["open"])
        fwd   = d.iloc[i + 1: i + 97].copy()
        if len(fwd) < 4:
            continue

        signals.append({
            "entry": entry, "atr": atr, "direction": dirn,
            "date": row["time"], "df_fwd": fwd,
        })
        last_sig_idx = i + 1

    return run_sim(signals, START_BALANCE)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 76)
    print("  FOREX BACKTEST -- All Deriv Forex Pairs")
    print("  2 years H1 (yfinance) | 3 strategies | $10,000 start")
    print("  Session: Mon-Fri 08:00-20:00 UTC")
    print("=" * 76)
    print()

    results = []
    pairs = list(FOREX_PAIRS.keys())

    for i, yf_sym in enumerate(pairs):
        label = yf_sym.replace("=X", "")
        print(f"  [{i+1:02d}/{len(pairs)}] {label:<10}", end=" ", flush=True)

        df = load_h1(yf_sym)
        if df is None or len(df) < 500:
            print("-- no data")
            continue

        df = drop_weekends(df)
        days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
        print(f"({len(df):,} candles, {days}d) ", end="", flush=True)

        r1 = backtest_step_trend(df)
        r2 = backtest_london_breakout(df)
        r3 = backtest_ema_trend(df)

        def fmt(r):
            if r is None or r["trades"] == 0:
                return "  --   "
            return f"PF={r['pf']:.2f}"

        print(f"step_trend:{fmt(r1)}  lbo:{fmt(r2)}  ema:{fmt(r3)}")

        for strat, r in [("step_trend", r1), ("london_breakout", r2), ("ema_trend", r3)]:
            if r and r["trades"] > 0:
                results.append({
                    "symbol": label, "strategy": strat,
                    "trades": r["trades"], "wr": r["wr"],
                    "pf": r["pf"], "avg_r": r["avg_r"],
                    "ret": r["ret"], "max_dd": r["max_dd"],
                    "days": days,
                })

    if not results:
        print("No results.")
        return

    df_r = pd.DataFrame(results)
    df_r = df_r.sort_values("pf", ascending=False).reset_index(drop=True)

    print()
    print("=" * 76)
    print("  FINAL RESULTS -- RANKED BY PROFIT FACTOR")
    print()
    print(f"  {'Symbol':<12} {'Strategy':<18} {'Trades':>7} {'WR':>7} {'PF':>6} {'AvgR':>7} {'Ret%':>7} {'MaxDD':>7}")
    print("  " + "-" * 72)

    for _, row in df_r.iterrows():
        edge = ""
        if row["pf"] >= 1.40 and row["wr"] >= 0.48 and row["trades"] >= 20:
            edge = " *** STRONG"
        elif row["pf"] >= 1.20 and row["trades"] >= 15:
            edge = " *   DECENT"
        print(
            f"  {row['symbol']:<12} {row['strategy']:<18} "
            f"{row['trades']:>7} {row['wr']:>6.1%} {row['pf']:>6.2f} "
            f"{row['avg_r']:>+6.2f}R {row['ret']:>+6.1f}% {row['max_dd']:>6.1f}%"
            f"{edge}"
        )

    strong = df_r[(df_r["pf"] >= 1.40) & (df_r["wr"] >= 0.48) & (df_r["trades"] >= 20)]
    decent = df_r[(df_r["pf"] >= 1.20) & (df_r["trades"] >= 15) & ~df_r.index.isin(strong.index)]

    print()
    print("=" * 76)
    print(f"  Strong edge (PF>=1.4, WR>=48%, 20+ trades): {len(strong)}")
    print(f"  Decent edge (PF>=1.2, 15+ trades):          {len(decent)}")
    print("=" * 76)

    df_r.to_csv(BASE_DIR / "quant_cache" / "forex_results.csv", index=False)
    print(f"  Full results saved to quant_cache/forex_results.csv")


if __name__ == "__main__":
    main()

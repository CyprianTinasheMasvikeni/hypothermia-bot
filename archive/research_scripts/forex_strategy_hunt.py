"""
Forex Strategy Hunter
Tests 5 strategies purpose-built for forex behavior on H1 data.
No synthetic index comparisons — forex only.

Strategy A: Daily Trend + H1 Pullback
  Daily EMA50 bias -> H1 EMA crossover + RSI + pullback entry

Strategy B: London Breakout (refined)
  Asian range (22:00-07:00) break at London open
  Range size filter + ATR confirmation

Strategy C: Previous Day High/Low Break
  Break of yesterday's high or low during London/NY session

Strategy D: Bollinger Band Mean Reversion
  Price outside 2std band + RSI extreme -> fade back to mean
  Works on range-bound pairs (EURGBP, EURCHF etc.)

Strategy E: ADX Momentum Breakout
  ADX > 25 (trending) + price breaks 20-bar high/low
  Rides strong directional moves
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR  = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "forex_cache"
sys.path.insert(0, str(BASE_DIR))

PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "USDJPY", "USDCAD", "USDCHF",
    "EURGBP", "EURJPY", "EURCAD", "EURCHF", "GBPJPY", "GBPAUD",
    "EURAUD", "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "EURNZD",
    "GBPCAD", "GBPCHF", "GBPNZD", "NZDJPY", "NZDUSD",
]

# Risk settings — same framework across all strategies
MULTIPLIER       = 100
SL_ATR_MULT      = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
RISK_PCT         = 0.05
ACCOUNT_DD       = 0.15
START_BAL        = 10_000.0

# Sessions (UTC hours)
LONDON_OPEN  = 7
LONDON_CLOSE = 16
NY_OPEN      = 13
NY_CLOSE     = 20


# ── DATA ──────────────────────────────────────────────────────────────────────
def load(pair: str) -> pd.DataFrame | None:
    cache = CACHE_DIR / f"forex_{pair}_H1.csv"
    if not cache.exists():
        return None
    df = pd.read_csv(cache, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df[df["time"].dt.weekday < 5].sort_values("time").reset_index(drop=True)
    return df


# ── INDICATORS ────────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # EMAs
    for p in [8, 20, 50, 200]:
        d[f"ema{p}"] = d["close"].ewm(span=p, adjust=False).mean()

    # ATR
    d["tr"] = np.maximum(d["high"] - d["low"],
               np.maximum(abs(d["high"] - d["close"].shift(1)),
                          abs(d["low"]  - d["close"].shift(1))))
    d["atr"] = d["tr"].rolling(14).mean()

    # RSI
    delta = d["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Bollinger Bands (20, 2std)
    d["bb_mid"]  = d["close"].rolling(20).mean()
    d["bb_std"]  = d["close"].rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_mid"] - 2 * d["bb_std"]

    # ADX
    plus_dm  = (d["high"] - d["high"].shift(1)).clip(lower=0)
    minus_dm = (d["low"].shift(1) - d["low"]).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14    = d["tr"].rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean()  / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx       = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = dx.rolling(14).mean()
    d["+di"]  = plus_di
    d["-di"]  = minus_di

    # EMA slope as % (percentage per candle — works across all price levels)
    d["ema20_slope_pct"] = (d["ema20"] - d["ema20"].shift(5)) / d["ema20"].shift(5) * 100
    d["ema50_slope_pct"] = (d["ema50"] - d["ema50"].shift(5)) / d["ema50"].shift(5) * 100

    # 20-bar high/low for breakout
    d["high20"] = d["high"].rolling(20).max()
    d["low20"]  = d["low"].rolling(20).min()

    # Previous day OHLC
    d["date"] = d["time"].dt.date
    daily = d.groupby("date").agg(
        d_high=("high", "max"), d_low=("low", "min"),
        d_open=("open", "first"), d_close=("close", "last")
    ).reset_index()
    daily["prev_high"]  = daily["d_high"].shift(1)
    daily["prev_low"]   = daily["d_low"].shift(1)
    daily["prev_close"] = daily["d_close"].shift(1)
    d = d.merge(daily[["date", "prev_high", "prev_low", "prev_close"]], on="date", how="left")

    return d.dropna(subset=["atr", "rsi", "adx"]).reset_index(drop=True)


# ── SIMULATOR ─────────────────────────────────────────────────────────────────
def simulate(signals: list, start_bal: float) -> dict | None:
    if not signals:
        return None
    bal    = start_bal
    peak   = start_bal
    wins   = losses = 0
    rs     = []
    equity = []

    for s in signals:
        if bal < peak * (1 - ACCOUNT_DD):
            break
        entry = s["entry"]
        atr   = s["atr"]
        dirn  = s["dir"]
        d     = 1 if dirn == "BUY" else -1
        risk  = bal * RISK_PCT
        sf    = MULTIPLIER * SL_ATR_MULT * atr / entry
        stake = max(1.0, round(risk / sf, 2)) if sf > 0 else max(1.0, round(risk, 2))
        sl    = entry - d * atr * SL_ATR_MULT

        partial_done  = False
        locked_pnl    = 0.0
        cur_stake     = stake
        peak_price    = entry
        peak_r        = 0.0
        pnl           = None

        for _, row in s["fwd"].iterrows():
            hi = row["high"]
            lo = row["low"]

            # SL
            if (d == 1 and lo <= sl) or (d == -1 and hi >= sl):
                pnl = cur_stake * MULTIPLIER * d * (sl - entry) / entry + locked_pnl
                break

            # Partial at 2R
            if not partial_done:
                pp = entry + d * atr * PARTIAL_R
                if (d == 1 and hi >= pp) or (d == -1 and lo <= pp):
                    locked_pnl   = cur_stake * PARTIAL_PCT * MULTIPLIER * d * (pp - entry) / entry
                    cur_stake   *= (1 - PARTIAL_PCT)
                    partial_done = True

            # Peak tracking
            peak_price = max(peak_price, hi) if d == 1 else min(peak_price, lo)
            peak_r     = abs(peak_price - entry) / atr if atr > 0 else 0

            # Chandelier
            cm    = CHANDELIER_TIERS[0][1]
            for mr, tm in CHANDELIER_TIERS:
                if peak_r >= mr:
                    cm = tm
            csl = peak_price - d * atr * cm
            if (d == 1 and lo <= csl) or (d == -1 and hi >= csl):
                pnl = cur_stake * MULTIPLIER * d * (csl - entry) / entry + locked_pnl
                break

        if pnl is None:
            last = s["fwd"].iloc[-1]["close"]
            pnl  = cur_stake * MULTIPLIER * d * (last - entry) / entry + locked_pnl

        bal  += pnl
        peak  = max(peak, bal)
        r_val = pnl / risk if risk > 0 else 0
        rs.append(r_val)
        equity.append(bal)
        if pnl > 0:
            wins += 1
        else:
            losses += 1

    total = wins + losses
    if total < 5:
        return None
    gross_w = sum(p for p in rs if p > 0) * (start_bal * RISK_PCT)
    gross_l = abs(sum(p for p in rs if p < 0)) * (start_bal * RISK_PCT)
    pf      = gross_w / gross_l if gross_l > 0 else 0
    pk      = start_bal
    mdd     = 0.0
    b       = start_bal
    for r in rs:
        b  += r * start_bal * RISK_PCT
        pk  = max(pk, b)
        mdd = max(mdd, (pk - b) / pk)
    return {
        "trades": total, "wins": wins, "wr": wins / total,
        "pf": pf, "avg_r": np.mean(rs), "ret": (bal / start_bal - 1) * 100,
        "max_dd": mdd * 100,
    }


# ── STRATEGY A: Daily Bias + H1 Trend Pullback ────────────────────────────────
def strat_a(df: pd.DataFrame) -> dict | None:
    """
    Daily EMA50 direction sets bias.
    H1: EMA20 > EMA50 (bull) + price pulls back toward EMA20 + RSI 45-65 range.
    Entry on bullish/bearish close. Forex-calibrated slopes (% based).
    """
    sigs = []
    last = -1
    for i in range(220, len(df) - 1):
        if i == last:
            continue
        r    = df.iloc[i]
        hour = r["time"].hour
        if not (LONDON_OPEN <= hour < NY_CLOSE):
            continue

        atr = r["atr"]
        if atr <= 0 or pd.isna(atr):
            continue

        # % slope filter (forex calibrated: 0.01% per 5 bars is meaningful)
        slope20 = r["ema20_slope_pct"]
        slope50 = r["ema50_slope_pct"]

        bull = (r["ema20"] > r["ema50"] > r["ema200"]
                and slope20 > 0.005 and slope50 > 0.002
                and abs(r["close"] - r["ema20"]) < atr * 1.5
                and 40 < r["rsi"] < 65
                and r["close"] > r["open"])

        bear = (r["ema20"] < r["ema50"] < r["ema200"]
                and slope20 < -0.005 and slope50 < -0.002
                and abs(r["close"] - r["ema20"]) < atr * 1.5
                and 35 < r["rsi"] < 60
                and r["close"] < r["open"])

        dirn = "BUY" if bull else ("SELL" if bear else None)
        if dirn is None:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 49].copy()
        if len(fwd) < 4:
            continue
        sigs.append({"entry": entry, "atr": atr, "dir": dirn, "fwd": fwd})
        last = i + 1

    return simulate(sigs, START_BAL)


# ── STRATEGY B: London Breakout (refined) ────────────────────────────────────
def strat_b(df: pd.DataFrame) -> dict | None:
    """
    Asian range 22:00-07:00 UTC.
    At London open (07:00-10:00): enter on break of Asian H/L.
    Range filter: must be 0.3-3x ATR (not too tight, not news spike).
    Only 1 trade per day.
    """
    sigs = []
    traded = set()
    for i in range(100, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < LONDON_OPEN or hour >= 10:
            continue
        if date in traded:
            continue

        atr = r["atr"]
        if atr <= 0:
            continue

        # Asian range from yesterday 22:00 to today 07:00
        t           = r["time"]
        asian_start = t.normalize() - pd.Timedelta(hours=2)   # yesterday 22:00
        asian_end   = t.normalize() + pd.Timedelta(hours=LONDON_OPEN)
        asian       = df[(df["time"] >= asian_start) & (df["time"] < asian_end)]
        if len(asian) < 5:
            continue

        a_high = asian["high"].max()
        a_low  = asian["low"].min()
        rng    = a_high - a_low

        if rng < atr * 0.3 or rng > atr * 3.0:
            continue

        close = float(r["close"])
        dirn  = None
        if close > a_high:
            dirn = "BUY"
        elif close < a_low:
            dirn = "SELL"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 33].copy()   # max 8 hrs
        if len(fwd) < 4:
            continue
        sigs.append({"entry": entry, "atr": atr, "dir": dirn, "fwd": fwd})
        traded.add(date)

    return simulate(sigs, START_BAL)


# ── STRATEGY C: Previous Day High/Low Break ───────────────────────────────────
def strat_c(df: pd.DataFrame) -> dict | None:
    """
    Break and close above yesterday's high = BUY.
    Break and close below yesterday's low = SELL.
    Session: London open only (07:00-12:00 UTC). 1 trade per day.
    """
    sigs = []
    traded = set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < LONDON_OPEN or hour >= 12:
            continue
        if date in traded:
            continue

        atr      = r["atr"]
        prev_h   = r["prev_high"]
        prev_l   = r["prev_low"]
        close    = r["close"]
        if pd.isna(prev_h) or pd.isna(prev_l) or atr <= 0:
            continue

        dirn = None
        if close > prev_h and r["close"] > r["open"]:
            dirn = "BUY"
        elif close < prev_l and r["close"] < r["open"]:
            dirn = "SELL"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 25].copy()
        if len(fwd) < 4:
            continue
        sigs.append({"entry": entry, "atr": atr, "dir": dirn, "fwd": fwd})
        traded.add(date)

    return simulate(sigs, START_BAL)


# ── STRATEGY D: Bollinger Band Mean Reversion ────────────────────────────────
def strat_d(df: pd.DataFrame) -> dict | None:
    """
    Price closes outside 2std band AND RSI extreme -> fade back to mid.
    RSI > 75 at upper band = SELL, RSI < 25 at lower band = BUY.
    Works on range-bound pairs. TP = BB midline.
    """
    sigs = []
    last = -1
    for i in range(30, len(df) - 1):
        if i == last:
            continue
        r    = df.iloc[i]
        hour = r["time"].hour
        if not (LONDON_OPEN <= hour < NY_CLOSE):
            continue

        atr = r["atr"]
        if atr <= 0:
            continue

        close    = r["close"]
        bb_upper = r["bb_upper"]
        bb_lower = r["bb_lower"]
        bb_mid   = r["bb_mid"]
        rsi      = r["rsi"]

        if pd.isna(bb_upper) or pd.isna(rsi):
            continue

        dirn = None
        if close > bb_upper and rsi > 72:
            dirn = "SELL"
        elif close < bb_lower and rsi < 28:
            dirn = "BUY"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        # TP = BB midline, SL = 1x ATR from entry
        d_val   = 1 if dirn == "BUY" else -1
        tp      = bb_mid
        # forward: hold until TP hit or SL hit (max 24 candles)
        fwd     = df.iloc[i + 1: i + 25].copy()
        if len(fwd) < 4:
            continue

        # Override SL ATR for this strategy (tighter — 1.5x ATR)
        custom_atr = atr * 1.0  # keep same, chandelier handles exit

        sigs.append({"entry": entry, "atr": custom_atr, "dir": dirn, "fwd": fwd})
        last = i + 1

    return simulate(sigs, START_BAL)


# ── STRATEGY E: ADX Momentum Breakout ────────────────────────────────────────
def strat_e(df: pd.DataFrame) -> dict | None:
    """
    ADX > 25 (market is trending strongly).
    +DI > -DI: buy breakout of 20-bar high.
    -DI > +DI: sell breakdown of 20-bar low.
    Session: London + NY overlap (13:00-18:00 UTC) for max momentum.
    """
    sigs = []
    last = -1
    for i in range(50, len(df) - 1):
        if i == last:
            continue
        r    = df.iloc[i]
        hour = r["time"].hour
        if not (LONDON_OPEN <= hour < NY_CLOSE):
            continue

        atr = r["atr"]
        if atr <= 0:
            continue

        adx    = r["adx"]
        pdi    = r["+di"]
        mdi    = r["-di"]
        high20 = r["high20"]
        low20  = r["low20"]

        if pd.isna(adx) or adx < 22:
            continue

        close = r["close"]
        prev_close = float(df.iloc[i - 1]["close"])
        prev_high20 = float(df.iloc[i - 1]["high20"]) if i > 0 else high20
        prev_low20  = float(df.iloc[i - 1]["low20"])  if i > 0 else low20

        dirn = None
        if pdi > mdi and close > high20 and prev_close <= prev_high20:
            dirn = "BUY"
        elif mdi > pdi and close < low20 and prev_close >= prev_low20:
            dirn = "SELL"
        else:
            continue

        entry = float(df.iloc[i + 1]["open"])
        fwd   = df.iloc[i + 1: i + 49].copy()
        if len(fwd) < 4:
            continue
        sigs.append({"entry": entry, "atr": atr, "dir": dirn, "fwd": fwd})
        last = i + 1

    return simulate(sigs, START_BAL)


# ── MAIN ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    "A_trend_pullback":  strat_a,
    "B_london_break":    strat_b,
    "C_prevday_break":   strat_c,
    "D_bb_reversion":    strat_d,
    "E_adx_momentum":    strat_e,
}

def main():
    print("=" * 78)
    print("  FOREX STRATEGY HUNT -- 5 Forex-Native Strategies")
    print("  2 years H1 data | 23 pairs | $10,000 start")
    print("=" * 78)
    print()

    results = []

    for pair in PAIRS:
        df_raw = load(pair)
        if df_raw is None:
            print(f"  {pair:<10} -- no cache")
            continue

        df = add_indicators(df_raw)
        days = (df["time"].iloc[-1] - df["time"].iloc[0]).days
        print(f"  {pair:<10} ({days}d)  ", end="", flush=True)

        row_results = {}
        for name, fn in STRATEGIES.items():
            r = fn(df)
            if r:
                row_results[name] = r
                tag = f"PF={r['pf']:.2f}"
            else:
                tag = "  --  "
            print(f"{name[:1]}:{tag}  ", end="", flush=True)
        print()

        for name, r in row_results.items():
            results.append({
                "pair": pair, "strategy": name,
                "trades": r["trades"], "wr": r["wr"],
                "pf": r["pf"], "avg_r": r["avg_r"],
                "ret": r["ret"], "max_dd": r["max_dd"], "days": days,
            })

    if not results:
        print("No results.")
        return

    df_r = pd.DataFrame(results).sort_values("pf", ascending=False).reset_index(drop=True)

    print()
    print("=" * 78)
    print("  TOP RESULTS -- Ranked by Profit Factor")
    print(f"  Threshold for STRONG: PF >= 1.40, WR >= 45%, Trades >= 20")
    print()
    print(f"  {'Pair':<10} {'Strategy':<22} {'Tr':>4} {'WR':>6} {'PF':>6} {'AvgR':>6} {'Ret%':>7} {'DD%':>6}")
    print("  " + "-" * 70)

    for _, row in df_r.head(25).iterrows():
        flag = ""
        if row["pf"] >= 1.40 and row["wr"] >= 0.45 and row["trades"] >= 20:
            flag = "  *** STRONG"
        elif row["pf"] >= 1.20 and row["trades"] >= 15:
            flag = "  *   DECENT"
        elif row["pf"] >= 1.10 and row["trades"] >= 10:
            flag = "      WATCH"
        print(
            f"  {row['pair']:<10} {row['strategy']:<22} "
            f"{row['trades']:>4} {row['wr']:>5.1%} {row['pf']:>6.2f} "
            f"{row['avg_r']:>+5.2f}R {row['ret']:>+6.1f}% {row['max_dd']:>5.1f}%"
            f"{flag}"
        )

    strong = df_r[(df_r["pf"] >= 1.40) & (df_r["wr"] >= 0.45) & (df_r["trades"] >= 20)]
    decent = df_r[(df_r["pf"] >= 1.20) & (df_r["trades"] >= 15)]

    print()
    print("=" * 78)
    print(f"  Strong edge: {len(strong)} | Decent edge: {len(decent)}")

    if not strong.empty:
        print()
        print("  STRONG EDGE CANDIDATES:")
        for _, row in strong.iterrows():
            print(f"    {row['pair']} + {row['strategy']}  "
                  f"PF={row['pf']:.2f}  WR={row['wr']:.1%}  "
                  f"{row['trades']} trades  +{row['ret']:.1f}%")

    print("=" * 78)
    df_r.to_csv(BASE_DIR / "quant_cache" / "forex_strategy_results.csv", index=False)


if __name__ == "__main__":
    main()

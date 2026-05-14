"""
Crypto Strategy Hunt
- Downloads BTC, ETH, SOL, BNB, XRP via yfinance (H1, 2 years)
- Characterizes each: ATR-CV (is volatility predictable?), Hurst
- Runs 5 strategies adapted for 24/7 crypto markets
- Verdict: which pair/strategy combo is worth a deep dive?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
CACHE_DIR  = BASE_DIR / "forex_cache"

# ── Strategy params (same engine as stpRNG/R_25 bots) ────────────────────────
MULTIPLIER       = 1          # crypto priced in USD directly (not pips)
SL_ATR_MULT      = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R        = 2.0
PARTIAL_PCT      = 0.50
RISK_PCT         = 0.05
START_BAL        = 10_000.0
ACCOUNT_DD       = 0.15       # kill switch at 15% drawdown

CRYPTO_PAIRS = [
    ("BTC-USD",  "BTCUSD"),
    ("ETH-USD",  "ETHUSD"),
    ("SOL-USD",  "SOLUSD"),
    ("BNB-USD",  "BNBUSD"),
    ("XRP-USD",  "XRPUSD"),
]


# ── Data ──────────────────────────────────────────────────────────────────────
def download_crypto(yf_ticker, symbol):
    cache = CACHE_DIR / f"crypto_{symbol}_H1.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["time"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        print(f"  {symbol}: loaded from cache ({len(df)} rows)")
        return df

    print(f"  {symbol}: downloading from yfinance...", end="", flush=True)
    raw = yf.download(yf_ticker, period="730d", interval="1h",
                      auto_adjust=True, progress=False)
    if raw.empty:
        print(" FAILED")
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]

    raw = raw.reset_index()
    raw = raw.rename(columns={"datetime": "time", "date": "time",
                               "index": "time", "Datetime": "time"})
    # find the time column
    for col in raw.columns:
        if "time" in col.lower() or "date" in col.lower():
            raw = raw.rename(columns={col: "time"})
            break

    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    df = raw[["time", "open", "high", "low", "close"]].dropna().copy()
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f" done ({len(df)} rows, {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    return df


def add_indicators(df):
    d = df.copy()
    d["tr"] = np.maximum(d["high"] - d["low"],
               np.maximum(abs(d["high"] - d["close"].shift(1)),
                          abs(d["low"]  - d["close"].shift(1))))
    d["atr"]   = d["tr"].rolling(14).mean()
    d["ema20"]  = d["close"].ewm(span=20).mean()
    d["ema50"]  = d["close"].ewm(span=50).mean()
    d["ema200"] = d["close"].ewm(span=200).mean()
    d["bb_mid"] = d["close"].rolling(20).mean()
    d["bb_std"] = d["close"].rolling(20).std()
    d["bb_up"]  = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lo"]  = d["bb_mid"] - 2 * d["bb_std"]
    # RSI
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)
    # ADX
    tr   = d["tr"]
    dmp  = (d["high"] - d["high"].shift(1)).clip(lower=0)
    dmn  = (d["low"].shift(1) - d["low"]).clip(lower=0)
    atr14 = tr.rolling(14).mean()
    d["adx"] = (abs(dmp.rolling(14).mean() / atr14 - dmn.rolling(14).mean() / atr14) /
                ((dmp.rolling(14).mean() / atr14 + dmn.rolling(14).mean() / atr14).replace(0, np.nan)) * 100
                ).rolling(14).mean()
    # Previous day levels
    d["date"] = d["time"].dt.date
    daily = d.groupby("date").agg(
        d_high=("high","max"), d_low=("low","min"),
        d_open=("open","first"), d_close=("close","last")
    ).reset_index()
    daily["prev_high"]  = daily["d_high"].shift(1)
    daily["prev_low"]   = daily["d_low"].shift(1)
    daily["prev_close"] = daily["d_close"].shift(1)
    d = d.merge(daily[["date","prev_high","prev_low","prev_close"]], on="date", how="left")
    return d.dropna(subset=["atr"]).reset_index(drop=True)


# ── Trade engine (same chandelier system) ─────────────────────────────────────
def run_trade(entry, atr, dirn, fwd, bal):
    d     = 1 if dirn == "BUY" else -1
    risk  = bal * RISK_PCT
    # For crypto: stake is in USD directly
    # PnL = stake * (exit - entry) * direction
    sl    = entry - d * atr * SL_ATR_MULT
    stake = risk / (atr * SL_ATR_MULT) if atr > 0 else 0  # units of asset

    partial_done = False
    locked_pnl   = 0.0
    cur_stake    = stake
    peak_price   = entry
    peak_r       = 0.0

    for _, row in fwd.iterrows():
        hi, lo = row["high"], row["low"]

        if (d == 1 and lo <= sl) or (d == -1 and hi >= sl):
            pnl = cur_stake * d * (sl - entry) + locked_pnl
            return round(pnl, 2), peak_r, "SL"

        if not partial_done:
            pp = entry + d * atr * PARTIAL_R
            if (d == 1 and hi >= pp) or (d == -1 and lo <= pp):
                locked_pnl   = cur_stake * PARTIAL_PCT * d * (pp - entry)
                cur_stake   *= (1 - PARTIAL_PCT)
                partial_done = True

        peak_price = max(peak_price, hi) if d == 1 else min(peak_price, lo)
        peak_r     = abs(peak_price - entry) / atr if atr > 0 else 0

        cm = CHANDELIER_TIERS[0][1]
        for mr, tm in CHANDELIER_TIERS:
            if peak_r >= mr:
                cm = tm
        csl = peak_price - d * atr * cm
        if (d == 1 and lo <= csl) or (d == -1 and hi >= csl):
            pnl = cur_stake * d * (csl - entry) + locked_pnl
            return round(pnl, 2), peak_r, "CHANDELIER"

    last = fwd.iloc[-1]["close"]
    pnl  = cur_stake * d * (last - entry) + locked_pnl
    return round(pnl, 2), peak_r, "TIME"


# ── Characterizer ──────────────────────────────────────────────────────────────
def characterize(symbol, df):
    atr_cv = (df["atr"].std() / df["atr"].mean()) if df["atr"].mean() > 0 else 0

    # Hurst exponent (simplified R/S)
    prices = df["close"].values[-2000:]
    lags   = [8, 16, 32, 64, 128]
    rs_vals = []
    for lag in lags:
        chunks = [prices[i:i+lag] for i in range(0, len(prices)-lag, lag)]
        rs_c   = []
        for c in chunks:
            m  = np.mean(c)
            cs = np.cumsum(c - m)
            r  = cs.max() - cs.min()
            s  = np.std(c, ddof=1)
            if s > 0:
                rs_c.append(r / s)
        if rs_c:
            rs_vals.append((lag, np.mean(rs_c)))
    if len(rs_vals) >= 3:
        x = np.log([r[0] for r in rs_vals])
        y = np.log([r[1] for r in rs_vals])
        hurst = np.polyfit(x, y, 1)[0]
    else:
        hurst = 0.5

    predictability = "Very predictable" if atr_cv < 0.12 else \
                     "Predictable"       if atr_cv < 0.18 else \
                     "Moderate"          if atr_cv < 0.25 else \
                     "Unpredictable"

    return round(atr_cv, 3), round(hurst, 3), predictability


# ── 5 Strategies ──────────────────────────────────────────────────────────────

def strat_ema_trend(df):
    """Strategy A: % EMA trend pullback. Slope based on % move, not raw price."""
    trades, traded = [], set()
    for i in range(210, len(df) - 1):
        r    = df.iloc[i]
        date = r["time"].date()
        if date in traded:
            continue
        atr = r["atr"]
        if atr <= 0 or pd.isna(atr):
            continue
        e20, e50, e200 = r["ema20"], r["ema50"], r["ema200"]
        slope20 = (e20 - df.iloc[i-5]["ema20"]) / df.iloc[i-5]["ema20"] * 100
        slope50 = (e50 - df.iloc[i-5]["ema50"]) / df.iloc[i-5]["ema50"] * 100

        dirn = None
        if e20 > e50 > e200 and slope20 > 0.05 and slope50 > 0.02:
            if r["low"] <= e20 <= r["high"]:
                dirn = "BUY"
        elif e20 < e50 < e200 and slope20 < -0.05 and slope50 < -0.02:
            if r["low"] <= e20 <= r["high"]:
                dirn = "SELL"
        if dirn is None:
            continue

        entry = float(df.iloc[i+1]["open"])
        fwd   = df.iloc[i+1: i+49].copy()
        if len(fwd) < 4:
            continue
        trades.append(("EMA_TREND", date, dirn, entry, atr, fwd))
        traded.add(date)
    return trades


def strat_asia_breakout(df):
    """Strategy B: Asia session (00:00-08:00 UTC) range, break during 08:00-16:00."""
    trades, traded = [], set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < 8 or hour >= 16:
            continue
        if date in traded:
            continue
        atr = r["atr"]
        if atr <= 0 or pd.isna(atr):
            continue

        t          = r["time"]
        asia_start = t.normalize()                        # 00:00 UTC
        asia_end   = t.normalize() + pd.Timedelta(hours=8)
        asian      = df[(df["time"] >= asia_start) & (df["time"] < asia_end)]
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
        if dirn is None:
            continue

        entry = float(df.iloc[i+1]["open"])
        fwd   = df.iloc[i+1: i+33].copy()
        if len(fwd) < 4:
            continue
        trades.append(("ASIA_BREAK", date, dirn, entry, atr, fwd))
        traded.add(date)
    return trades


def strat_prevday_break(df):
    """Strategy C: Previous day high/low break, 08:00-20:00 UTC window."""
    trades, traded = [], set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        hour = r["time"].hour
        date = r["time"].date()
        if hour < 8 or hour >= 20:
            continue
        if date in traded:
            continue
        atr    = r["atr"]
        prev_h = r["prev_high"]
        prev_l = r["prev_low"]
        if pd.isna(prev_h) or pd.isna(prev_l) or atr <= 0:
            continue

        close = r["close"]
        dirn  = None
        if close > prev_h and r["close"] > r["open"]:
            dirn = "BUY"
        elif close < prev_l and r["close"] < r["open"]:
            dirn = "SELL"
        if dirn is None:
            continue

        entry = float(df.iloc[i+1]["open"])
        fwd   = df.iloc[i+1: i+25].copy()
        if len(fwd) < 4:
            continue
        trades.append(("PREVDAY", date, dirn, entry, atr, fwd))
        traded.add(date)
    return trades


def strat_bb_reversion(df):
    """Strategy D: Bollinger Band extremes + RSI confirmation."""
    trades, traded = [], set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        date = r["time"].date()
        if date in traded:
            continue
        atr = r["atr"]
        if atr <= 0 or pd.isna(atr) or pd.isna(r["rsi"]):
            continue

        dirn = None
        if r["close"] >= r["bb_up"] and r["rsi"] >= 72:
            dirn = "SELL"
        elif r["close"] <= r["bb_lo"] and r["rsi"] <= 28:
            dirn = "BUY"
        if dirn is None:
            continue

        entry = float(df.iloc[i+1]["open"])
        fwd   = df.iloc[i+1: i+25].copy()
        if len(fwd) < 4:
            continue
        trades.append(("BB_REVERT", date, dirn, entry, atr, fwd))
        traded.add(date)
    return trades


def strat_adx_momentum(df):
    """Strategy E: ADX > 22 momentum breakout of 20-bar high/low."""
    trades, traded = [], set()
    for i in range(50, len(df) - 1):
        r    = df.iloc[i]
        date = r["time"].date()
        if date in traded:
            continue
        atr = r["atr"]
        if atr <= 0 or pd.isna(atr) or pd.isna(r["adx"]):
            continue
        if r["adx"] < 22:
            continue

        window  = df.iloc[i-20:i]
        w20_hi  = window["high"].max()
        w20_lo  = window["low"].min()
        close   = r["close"]
        dirn    = None
        if close > w20_hi:
            dirn = "BUY"
        elif close < w20_lo:
            dirn = "SELL"
        if dirn is None:
            continue

        entry = float(df.iloc[i+1]["open"])
        fwd   = df.iloc[i+1: i+49].copy()
        if len(fwd) < 4:
            continue
        trades.append(("ADX_MOM", date, dirn, entry, atr, fwd))
        traded.add(date)
    return trades


# ── Run + score ───────────────────────────────────────────────────────────────
def score_trades(raw_trades):
    bal  = START_BAL
    peak = START_BAL
    mdd  = 0.0
    results = []
    for (strat, date, dirn, entry, atr, fwd) in raw_trades:
        pnl, peak_r, reason = run_trade(entry, atr, dirn, fwd, bal)
        r_val = pnl / (bal * RISK_PCT) if bal > 0 else 0
        bal  += pnl
        peak  = max(peak, bal)
        mdd   = max(mdd, (peak - bal) / peak)
        results.append({
            "strategy": strat, "date": str(date), "dir": dirn,
            "pnl": pnl, "r": round(r_val, 2), "peak_r": round(peak_r, 2),
            "result": "WIN" if pnl > 0 else "LOSS", "reason": reason,
        })
        if mdd >= ACCOUNT_DD:
            break  # kill switch
    return results, round(bal, 2), round(mdd, 4)


def summarize(symbol, strat_name, results, final_bal, mdd):
    if not results:
        return None
    df_t   = pd.DataFrame(results)
    total  = len(df_t)
    wins   = (df_t["result"] == "WIN").sum()
    wr     = wins / total
    gw     = df_t[df_t["pnl"] > 0]["pnl"].sum()
    gl     = abs(df_t[df_t["pnl"] < 0]["pnl"].sum())
    pf     = gw / gl if gl > 0 else 0
    avg_r  = df_t["r"].mean()
    ret    = (final_bal / START_BAL - 1) * 100
    return {
        "symbol": symbol, "strategy": strat_name,
        "trades": total, "wr": round(wr, 3), "pf": round(pf, 2),
        "avg_r": round(avg_r, 2), "ret_pct": round(ret, 1),
        "mdd_pct": round(mdd * 100, 1),
    }


STRATEGIES = [
    ("EMA_TREND",  strat_ema_trend),
    ("ASIA_BREAK", strat_asia_breakout),
    ("PREVDAY",    strat_prevday_break),
    ("BB_REVERT",  strat_bb_reversion),
    ("ADX_MOM",    strat_adx_momentum),
]


def main():
    print()
    print("=" * 70)
    print("  CRYPTO STRATEGY HUNT: BTC ETH SOL BNB XRP")
    print("  Chandelier exit system — same engine as stpRNG/R_25 bots")
    print("=" * 70)
    print()

    # ── Step 1: Download / cache ──────────────────────────────────────────────
    print("DOWNLOADING DATA:")
    datasets = {}
    for yf_ticker, symbol in CRYPTO_PAIRS:
        df = download_crypto(yf_ticker, symbol)
        if df is not None and len(df) > 200:
            df = add_indicators(df)
            datasets[symbol] = df

    print()

    # ── Step 2: Characterize ──────────────────────────────────────────────────
    print("CHARACTERIZATION (ATR-CV < 0.12 = ideal for chandelier):")
    print(f"  {'Symbol':<10} {'ATR-CV':>8} {'Hurst':>7}  Verdict")
    print(f"  {'-'*55}")
    for symbol, df in datasets.items():
        atrcv, hurst, verdict = characterize(symbol, df)
        chandelier_ok = "GOOD" if atrcv < 0.18 else ("MARGINAL" if atrcv < 0.25 else "BAD")
        print(f"  {symbol:<10} {atrcv:>8.3f} {hurst:>7.3f}  {verdict} [{chandelier_ok}]")
    print()
    print("  Reference: stpRNG=0.083 (Very predictable), R_25=0.129 (Predictable)")
    print()

    # ── Step 3: Strategy hunt ─────────────────────────────────────────────────
    print("STRATEGY RESULTS:")
    print(f"  {'Symbol':<10} {'Strategy':<12} {'Tr':>5} {'WR':>7} {'PF':>6} "
          f"{'AvgR':>6} {'Ret%':>7} {'MDD%':>7}  Flag")
    print(f"  {'-'*78}")

    all_results = []
    for symbol, df in datasets.items():
        for strat_name, strat_fn in STRATEGIES:
            raw = strat_fn(df)
            if not raw:
                print(f"  {symbol:<10} {strat_name:<12} {'0':>5}  (no signals)")
                continue
            results, final_bal, mdd = score_trades(raw)
            row = summarize(symbol, strat_name, results, final_bal, mdd)
            if row is None:
                continue
            flag = ""
            if row["pf"] >= 1.5 and row["trades"] >= 20:
                flag = " <<< STRONG"
            elif row["pf"] >= 1.25 and row["trades"] >= 15:
                flag = " << DECENT"
            elif row["pf"] < 1.0:
                flag = " (losing)"
            print(f"  {symbol:<10} {strat_name:<12} {row['trades']:>5} "
                  f"{row['wr']:>6.1%} {row['pf']:>6.2f} "
                  f"{row['avg_r']:>+5.2f}R {row['ret_pct']:>+6.1f}% "
                  f"{row['mdd_pct']:>6.1f}%{flag}")
            all_results.append(row)

    # ── Step 4: Top candidates ────────────────────────────────────────────────
    if all_results:
        print()
        print("TOP CANDIDATES (PF >= 1.25, >= 15 trades):")
        top = [r for r in all_results if r["pf"] >= 1.25 and r["trades"] >= 15]
        top.sort(key=lambda x: x["pf"], reverse=True)
        if top:
            for r in top[:5]:
                print(f"  {r['symbol']} + {r['strategy']}: "
                      f"PF={r['pf']:.2f}, WR={r['wr']:.1%}, "
                      f"{r['trades']} trades, ret={r['ret_pct']:+.1f}%")
        else:
            print("  None passed the threshold.")
    print()


if __name__ == "__main__":
    main()

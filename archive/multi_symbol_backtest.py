"""
Multi-symbol backtest: Step Index vs Step Index 200 vs Step Index 300
Uses B:NEW settings (progressive chandelier + 25% partial)
5 windows x 15000 M5 bars per symbol
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import strategy_step_trend as strategy

BARS = 15000
TREND_TF = mt5.TIMEFRAME_M15
ENTRY_TF  = mt5.TIMEFRAME_M5

RISK_BASE = 0.05; RISK_HOT = 0.08; RISK_COLD = 0.03; STREAK_THRESHOLD = 2
SL_ATR_MULT = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R = 2.0; PARTIAL_PCT = 0.25
MAX_HOLD_CANDLES = 96; MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT = 0.03; ACCOUNT_DD_LIMIT = 0.15
SESSION_START = 9; SESSION_END = 19; SKIP_HOURS = {11, 14}
START_BALANCE = 100.0

def chandelier_mult(peak_r):
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            mult = m
    return mult

def current_risk(cw, cl):
    if cw >= STREAK_THRESHOLD: return RISK_HOT
    if cl >= STREAK_THRESHOLD: return RISK_COLD
    return RISK_BASE

def fetch(symbol, tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time","open","high","low","close","tick_volume"]]

def run_window(symbol, start_pos):
    trend_raw = fetch(symbol, TREND_TF, BARS, start_pos // 3)
    entry_raw = fetch(symbol, ENTRY_TF,  BARS, start_pos)
    if trend_raw is None or entry_raw is None: return []
    trend_df = strategy.calculate_indicators(trend_raw)
    entry_df = strategy.calculate_indicators(entry_raw)

    trades = []; balance = START_BALANCE; peak = START_BALANCE
    open_trade = None; tbd = {}; dsb = {}; cw = 0; cl = 0

    def get_bias(t):
        slc = trend_df[trend_df["time"] <= t]
        if len(slc) < 220: return "NEUTRAL"
        return strategy.analyze_setup(slc).get("checks", {}).get("trend_bias", "NEUTRAL")

    for i in range(220, len(entry_df) - 1):
        c = entry_df.iloc[i]; nc = entry_df.iloc[i+1]
        day = c["time"].date(); hour = c["time"].hour
        tbd.setdefault(day, 0); dsb.setdefault(day, balance)
        if balance < peak * (1 - ACCOUNT_DD_LIMIT): break

        if open_trade is not None:
            open_trade["age"] += 1
            t = open_trade["type"]; atr = open_trade["atr"]; d = 1 if t == "BUY" else -1
            hi, lo = c["high"], c["low"]

            if not open_trade["partial_done"]:
                pt = open_trade["entry"] + d * atr * PARTIAL_R
                if (t == "BUY" and hi >= pt) or (t == "SELL" and lo <= pt):
                    open_trade["locked_pnl"] = (pt - open_trade["entry"]) * d * (open_trade["size"] * PARTIAL_PCT)
                    open_trade["size"] *= (1 - PARTIAL_PCT)
                    open_trade["partial_done"] = True

            if t == "BUY": open_trade["peak"] = max(open_trade["peak"], hi)
            else:          open_trade["peak"] = min(open_trade["peak"], lo)

            peak_r = abs(open_trade["peak"] - open_trade["entry"]) / atr
            cm = chandelier_mult(peak_r)
            csl = open_trade["peak"] - d * atr * cm
            if t == "BUY": open_trade["sl"] = max(open_trade["sl"], csl)
            else:          open_trade["sl"] = min(open_trade["sl"], csl)

            sl_hit = (t == "BUY" and lo <= open_trade["sl"]) or (t == "SELL" and hi >= open_trade["sl"])
            timed  = open_trade["age"] >= MAX_HOLD_CANDLES
            if sl_hit or timed:
                ep  = open_trade["sl"] if sl_hit else c["close"]
                raw = (ep - open_trade["entry"]) * d * open_trade["size"]
                pnl = raw + open_trade["locked_pnl"]
                if pnl > 0: cw += 1; cl = 0
                else:       cl += 1; cw = 0
                balance += pnl; peak = max(peak, balance)
                trades.append({
                    "pnl": round(pnl, 4),
                    "r_multiple": round(pnl / open_trade["risk_amount"], 3) if open_trade["risk_amount"] else 0,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "balance": round(balance, 4),
                    "risk_pct": open_trade["risk_pct"],
                })
                tbd[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None: continue
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS: continue
        if balance < dsb[day] * (1 - DAILY_DD_LIMIT): continue
        if tbd[day] >= MAX_TRADES_PER_DAY: continue

        bias = get_bias(c["time"])
        if bias not in {"STRONG_BUY", "STRONG_SELL"}: continue
        es = entry_df.iloc[:i+1].copy()
        er = strategy.analyze_setup(es)
        sig = er.get("signal", "WAIT")
        if sig == "WAIT": continue
        req = "BUY" if bias == "STRONG_BUY" else "SELL"
        if sig != req: continue

        atr = float(entry_df.iloc[i]["atr"])
        if pd.isna(atr) or atr <= 0: continue
        rp = current_risk(cw, cl); ra = balance * rp; sz = ra / atr
        ep2 = float(nc["open"]); d2 = 1 if sig == "BUY" else -1
        sl2 = ep2 - d2 * atr * SL_ATR_MULT
        open_trade = {
            "type": sig, "entry": ep2, "sl": sl2, "atr": atr,
            "size": sz, "risk_pct": rp, "risk_amount": ra,
            "entry_time": nc["time"], "age": 0,
            "peak": ep2, "partial_done": False, "locked_pnl": 0.0,
        }
    return trades

def backtest_symbol(symbol):
    all_trades = []
    window_results = []
    for w in range(5):
        trades = run_window(symbol, w * BARS)
        all_trades.extend(trades)
        if trades:
            df_w = pd.DataFrame(trades)
            bal_curve = [START_BALANCE] + list(df_w["balance"])
            peak = START_BALANCE
            max_dd = 0
            for b in bal_curve:
                peak = max(peak, b)
                dd = (peak - b) / peak
                max_dd = max(max_dd, dd)
            window_results.append({
                "trades": len(trades),
                "final_bal": bal_curve[-1],
                "max_dd_pct": round(max_dd * 100, 2),
            })
    return all_trades, window_results

def print_symbol_report(symbol, all_trades, window_results):
    if not all_trades:
        print(f"\n  {symbol}: NO TRADES")
        return

    df = pd.DataFrame(all_trades)
    wins   = df[df["result"] == "WIN"]
    losses = df[df["result"] == "LOSS"]
    total  = len(df)
    wr     = len(wins) / total * 100 if total else 0
    net    = df["pnl"].sum()
    avg_w  = wins["r_multiple"].mean() if len(wins) else 0
    avg_l  = losses["r_multiple"].mean() if len(losses) else 0
    rr     = abs(avg_w / avg_l) if avg_l else 0
    expectancy = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)
    monsters = len(wins[wins["r_multiple"] >= 3.0])

    # max DD across all windows combined
    all_dd = [w["max_dd_pct"] for w in window_results]
    worst_dd = max(all_dd) if all_dd else 0
    avg_dd   = sum(all_dd) / len(all_dd) if all_dd else 0

    # trades per month approx: 5 windows x ~1.72 months each
    trades_per_month = total / (5 * 1.72)

    print(f"\n  Symbol        : {symbol}")
    print(f"  Total trades  : {total}  (~{trades_per_month:.1f}/month)")
    print(f"  Win rate      : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net profit    : ${net:+.2f}")
    print(f"  Avg winner    : +{avg_w:.3f}R   Avg loser: {avg_l:.3f}R")
    print(f"  Reward:Risk   : {rr:.2f}x")
    print(f"  Expectancy    : {expectancy:+.3f}R per trade")
    print(f"  Monsters (3R+): {monsters} trades")
    print(f"  Worst win-DD  : {worst_dd:.1f}%   Avg win-DD: {avg_dd:.1f}%")
    print(f"  Per window    :", end="")
    for i, w in enumerate(window_results):
        print(f"  W{i+1}: {w['trades']}tr ${w['final_bal']:.0f} DD{w['max_dd_pct']:.0f}%", end="")
    print()

def verdict(results):
    print("\n" + "="*65)
    print("  VERDICT: ADD TO LIVE BOT?")
    print("="*65)
    symbols = list(results.keys())
    for sym in symbols:
        df = pd.DataFrame(results[sym]["trades"]) if results[sym]["trades"] else pd.DataFrame()
        if df.empty:
            print(f"  {sym:<25} : SKIP - no data")
            continue
        wins = df[df["result"]=="WIN"]
        total = len(df)
        wr = len(wins)/total*100 if total else 0
        avg_w = wins["r_multiple"].mean() if len(wins) else 0
        avg_l = df[df["result"]=="LOSS"]["r_multiple"].mean() if len(df[df["result"]=="LOSS"]) else 0
        expectancy = (wr/100 * avg_w) + ((1-wr/100) * avg_l)
        net = df["pnl"].sum()
        ok = wr >= 50 and expectancy >= 0.5 and net > 0
        label = "YES - add it" if ok else "NO - skip it"
        print(f"  {sym:<25} : {label}  (WR={wr:.0f}%, E={expectancy:+.2f}R, net=${net:+.2f})")

def main():
    mt5.initialize()
    symbols = ["Step Index", "Step Index 200", "Step Index 300"]
    results = {}

    print("Running backtest on 3 symbols... (may take 1-2 minutes)")
    print("="*65)

    for sym in symbols:
        print(f"  Testing {sym}...")
        trades, window_results = backtest_symbol(sym)
        results[sym] = {"trades": trades, "window_results": window_results}

    print("\n" + "="*65)
    print("  RESULTS")
    print("="*65)
    for sym in symbols:
        print_symbol_report(sym, results[sym]["trades"], results[sym]["window_results"])

    verdict(results)
    mt5.shutdown()

if __name__ == "__main__":
    main()

"""
Volume test: 3-way comparison
  A: CURRENT  -- STRONG bias only, M5 entry (baseline)
  B: WEAK     -- STRONG + WEAK bias, M5 entry
  C: M1 ENTRY -- STRONG bias only, M1 entry

Goal: find which gets more trades without destroying win rate / expectancy.
B:NEW exit settings throughout (25% partial @ 2R, progressive chandelier).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import strategy_step_trend as strategy

SYMBOL = "Step Index"
BARS   = 15000

TREND_TF = mt5.TIMEFRAME_M15
M5_TF    = mt5.TIMEFRAME_M5
M1_TF    = mt5.TIMEFRAME_M1

RISK_BASE = 0.05; RISK_HOT = 0.08; RISK_COLD = 0.03; STREAK_THRESHOLD = 2
SL_ATR_MULT = 1.0
CHANDELIER_TIERS = [(0.0, 3.0), (2.0, 2.5), (4.0, 2.0)]
PARTIAL_R = 2.0; PARTIAL_PCT = 0.25
MAX_HOLD_CANDLES_M5 = 96   # 96x5min = 8h
MAX_HOLD_CANDLES_M1 = 480  # 480x1min = 8h
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT = 0.03; ACCOUNT_DD_LIMIT = 0.15
SESSION_START = 9; SESSION_END = 19; SKIP_HOURS = {11, 14}
START_BALANCE = 100.0

STRONG_BIASES = {"STRONG_BUY", "STRONG_SELL"}
WEAK_BIASES   = {"STRONG_BUY", "STRONG_SELL", "WEAK_BUY", "WEAK_SELL"}

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

def fetch(tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, start_pos, bars)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time","open","high","low","close","tick_volume"]]

def signal_to_dir(bias):
    """Map bias string to required trade direction."""
    if bias in {"STRONG_BUY", "WEAK_BUY"}:   return "BUY"
    if bias in {"STRONG_SELL", "WEAK_SELL"}: return "SELL"
    return None

def run_window(start_pos, entry_tf, allowed_biases, max_hold):
    m5_ratio  = 3 if entry_tf == M5_TF else 15   # M5=3x M15, M1=15x M15
    trend_raw = fetch(TREND_TF, BARS, start_pos // m5_ratio)
    entry_raw = fetch(entry_tf,  BARS, start_pos)
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
            cm     = chandelier_mult(peak_r)
            csl    = open_trade["peak"] - d * atr * cm
            if t == "BUY": open_trade["sl"] = max(open_trade["sl"], csl)
            else:          open_trade["sl"] = min(open_trade["sl"], csl)

            sl_hit = (t == "BUY" and lo <= open_trade["sl"]) or (t == "SELL" and hi >= open_trade["sl"])
            timed  = open_trade["age"] >= max_hold
            if sl_hit or timed:
                ep  = open_trade["sl"] if sl_hit else c["close"]
                raw = (ep - open_trade["entry"]) * d * open_trade["size"]
                pnl = raw + open_trade["locked_pnl"]
                if pnl > 0: cw += 1; cl = 0
                else:       cl += 1; cw = 0
                balance += pnl; peak = max(peak, balance)
                trades.append({
                    "pnl":        round(pnl, 4),
                    "r_multiple": round(pnl / open_trade["risk_amount"], 3) if open_trade["risk_amount"] else 0,
                    "result":     "WIN" if pnl > 0 else "LOSS",
                    "balance":    round(balance, 4),
                    "risk_pct":   open_trade["risk_pct"],
                    "bias":       open_trade["bias"],
                })
                tbd[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None: continue
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS: continue
        if balance < dsb[day] * (1 - DAILY_DD_LIMIT): continue
        if tbd[day] >= MAX_TRADES_PER_DAY: continue

        bias = get_bias(c["time"])
        if bias not in allowed_biases: continue
        req = signal_to_dir(bias)
        if req is None: continue

        es = entry_df.iloc[:i+1].copy()
        er = strategy.analyze_setup(es)
        sig = er.get("signal", "WAIT")
        if sig == "WAIT" or sig != req: continue

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
            "bias": bias,
        }
    return trades

def run_scenario(label, entry_tf, allowed_biases, max_hold):
    all_trades = []
    per_window = []
    for w in range(5):
        trades = run_window(w * BARS, entry_tf, allowed_biases, max_hold)
        all_trades.extend(trades)
        if trades:
            df_w = pd.DataFrame(trades)
            bal = [START_BALANCE] + list(df_w["balance"])
            pk = START_BALANCE; mdd = 0
            for b in bal:
                pk = max(pk, b)
                mdd = max(mdd, (pk - b) / pk)
            per_window.append({"trades": len(trades), "final": bal[-1], "dd": round(mdd*100,1)})
        else:
            per_window.append({"trades": 0, "final": START_BALANCE, "dd": 0})
    return {"label": label, "trades": all_trades, "windows": per_window}

def print_report(res):
    label = res["label"]
    df    = pd.DataFrame(res["trades"]) if res["trades"] else pd.DataFrame()
    wins  = df[df["result"]=="WIN"]  if not df.empty else pd.DataFrame()
    losses= df[df["result"]=="LOSS"] if not df.empty else pd.DataFrame()
    total = len(df)
    wr    = len(wins)/total*100 if total else 0
    net   = df["pnl"].sum() if not df.empty else 0
    avg_w = wins["r_multiple"].mean()   if len(wins)   else 0
    avg_l = losses["r_multiple"].mean() if len(losses) else 0
    rr    = abs(avg_w / avg_l) if avg_l else 0
    exp   = (wr/100 * avg_w) + ((1-wr/100) * avg_l)
    mon   = len(wins[wins["r_multiple"] >= 3.0]) if len(wins) else 0
    tpm   = total / (5 * 1.72)

    all_dd  = [w["dd"] for w in res["windows"]]
    worst_dd = max(all_dd) if all_dd else 0

    print(f"\n  {'='*60}")
    print(f"  {label}")
    print(f"  {'='*60}")
    print(f"  Total trades   : {total}  (~{tpm:.1f}/month)")
    print(f"  Win rate       : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net profit     : ${net:+.2f}")
    print(f"  Avg winner     : +{avg_w:.3f}R   Avg loser: {avg_l:.3f}R")
    print(f"  Reward:Risk    : {rr:.2f}x")
    print(f"  Expectancy     : {exp:+.3f}R per trade")
    print(f"  Monsters (3R+) : {mon} trades")
    print(f"  Worst window DD: {worst_dd:.1f}%")
    print(f"  Per window     :", end="")
    for i, w in enumerate(res["windows"]):
        print(f"  W{i+1}: {w['trades']}tr ${w['final']:.0f} DD{w['dd']:.0f}%", end="")
    print()

    # If WEAK variant, show breakdown by bias quality
    if not df.empty and "bias" in df.columns:
        strong_t = df[df["bias"].isin({"STRONG_BUY","STRONG_SELL"})]
        weak_t   = df[df["bias"].isin({"WEAK_BUY","WEAK_SELL"})]
        if len(weak_t) > 0:
            sw = strong_t[strong_t["result"]=="WIN"]
            ww = weak_t[weak_t["result"]=="WIN"]
            s_wr = len(sw)/len(strong_t)*100 if len(strong_t) else 0
            w_wr = len(ww)/len(weak_t)*100   if len(weak_t)   else 0
            s_exp = (s_wr/100 * sw["r_multiple"].mean() if len(sw) else 0) + \
                    ((1-s_wr/100) * strong_t[strong_t["result"]=="LOSS"]["r_multiple"].mean() if len(strong_t[strong_t["result"]=="LOSS"]) else 0)
            w_exp = (w_wr/100 * ww["r_multiple"].mean() if len(ww) else 0) + \
                    ((1-w_wr/100) * weak_t[weak_t["result"]=="LOSS"]["r_multiple"].mean() if len(weak_t[weak_t["result"]=="LOSS"]) else 0)
            print(f"\n  Bias breakdown:")
            print(f"    STRONG only : {len(strong_t)} trades  WR={s_wr:.0f}%  E={s_exp:+.2f}R")
            print(f"    WEAK only   : {len(weak_t)} trades   WR={w_wr:.0f}%  E={w_exp:+.2f}R")

def main():
    mt5.initialize()

    print("Running 3-way volume test...")
    print("A = current  |  B = allow WEAK bias  |  C = M1 entry")

    res_a = run_scenario("A: CURRENT  (STRONG bias, M5 entry)", M5_TF, STRONG_BIASES, MAX_HOLD_CANDLES_M5)
    print("  A done.")
    res_b = run_scenario("B: WEAK BIAS (STRONG+WEAK, M5 entry)", M5_TF, WEAK_BIASES, MAX_HOLD_CANDLES_M5)
    print("  B done.")
    res_c = run_scenario("C: M1 ENTRY  (STRONG bias, M1 entry)", M1_TF, STRONG_BIASES, MAX_HOLD_CANDLES_M1)
    print("  C done.")

    print_report(res_a)
    print_report(res_b)
    print_report(res_c)

    # Summary table
    print(f"\n  {'='*60}")
    print(f"  QUICK COMPARISON")
    print(f"  {'='*60}")
    print(f"  {'Scenario':<28} {'Trades':>6} {'T/mo':>5} {'WR':>6} {'Net':>8} {'Exp':>7} {'MaxDD':>7}")
    print(f"  {'-'*60}")
    for res in [res_a, res_b, res_c]:
        df    = pd.DataFrame(res["trades"]) if res["trades"] else pd.DataFrame()
        wins  = df[df["result"]=="WIN"]   if not df.empty else pd.DataFrame()
        losses= df[df["result"]=="LOSS"]  if not df.empty else pd.DataFrame()
        total = len(df)
        wr    = len(wins)/total*100 if total else 0
        net   = df["pnl"].sum() if not df.empty else 0
        avg_w = wins["r_multiple"].mean()   if len(wins)   else 0
        avg_l = losses["r_multiple"].mean() if len(losses) else 0
        exp   = (wr/100 * avg_w) + ((1-wr/100) * avg_l)
        tpm   = total / (5 * 1.72)
        all_dd = [w["dd"] for w in res["windows"]]
        worst_dd = max(all_dd) if all_dd else 0
        print(f"  {res['label'][:28]:<28} {total:>6} {tpm:>5.1f} {wr:>5.0f}% ${net:>+7.2f} {exp:>+6.3f}R {worst_dd:>6.1f}%")

    mt5.shutdown()

if __name__ == "__main__":
    main()

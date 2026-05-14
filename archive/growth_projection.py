"""
Monte Carlo growth projection: $100 -> $1000
Uses actual backtest trade R-multiples, simulates 10,000 paths.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import strategy_step_trend as strategy
import random

# ── pull actual trade R-multiples from backtest ───────────────────────────────
BARS = 15000
TREND_TF = mt5.TIMEFRAME_M15
ENTRY_TF  = mt5.TIMEFRAME_M5
SYMBOL = "Step Index"

RISK_BASE=0.05; RISK_HOT=0.08; RISK_COLD=0.03; STREAK_THRESHOLD=2
SL_ATR_MULT=1.0
CHANDELIER_TIERS=[(0.0,3.0),(2.0,2.5),(4.0,2.0)]
PARTIAL_R=2.0; PARTIAL_PCT=0.25
MAX_HOLD_CANDLES=96; MAX_TRADES_PER_DAY=6
DAILY_DD_LIMIT=0.03; ACCOUNT_DD_LIMIT=0.15
SESSION_START=9; SESSION_END=19; SKIP_HOURS={11,14}
START_BALANCE=100.0

def chandelier_mult(peak_r):
    mult = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r: mult = m
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

def collect_trades():
    all_r = []
    for w in range(5):
        trend_raw = fetch(TREND_TF, BARS, (w*BARS)//3)
        entry_raw = fetch(ENTRY_TF,  BARS, w*BARS)
        if trend_raw is None or entry_raw is None: continue
        trend_df = strategy.calculate_indicators(trend_raw)
        entry_df = strategy.calculate_indicators(entry_raw)
        balance=START_BALANCE; peak=START_BALANCE
        open_trade=None; tbd={}; dsb={}; cw=0; cl=0

        def get_bias(t):
            slc = trend_df[trend_df["time"]<=t]
            if len(slc)<220: return "NEUTRAL"
            return strategy.analyze_setup(slc).get("checks",{}).get("trend_bias","NEUTRAL")

        for i in range(220, len(entry_df)-1):
            c=entry_df.iloc[i]; nc=entry_df.iloc[i+1]
            day=c["time"].date(); hour=c["time"].hour
            tbd.setdefault(day,0); dsb.setdefault(day,balance)
            if balance<peak*(1-ACCOUNT_DD_LIMIT): break

            if open_trade is not None:
                open_trade["age"]+=1
                t=open_trade["type"]; atr=open_trade["atr"]; d=1 if t=="BUY" else -1
                hi,lo=c["high"],c["low"]
                if not open_trade["partial_done"]:
                    pt=open_trade["entry"]+d*atr*PARTIAL_R
                    if (t=="BUY" and hi>=pt) or (t=="SELL" and lo<=pt):
                        open_trade["locked_pnl"]=(pt-open_trade["entry"])*d*(open_trade["size"]*PARTIAL_PCT)
                        open_trade["size"]*=(1-PARTIAL_PCT)
                        open_trade["partial_done"]=True
                if t=="BUY": open_trade["peak"]=max(open_trade["peak"],hi)
                else:        open_trade["peak"]=min(open_trade["peak"],lo)
                peak_r=abs(open_trade["peak"]-open_trade["entry"])/atr
                cm=chandelier_mult(peak_r)
                csl=open_trade["peak"]-d*atr*cm
                if t=="BUY": open_trade["sl"]=max(open_trade["sl"],csl)
                else:        open_trade["sl"]=min(open_trade["sl"],csl)
                sl_hit=(t=="BUY" and lo<=open_trade["sl"]) or (t=="SELL" and hi>=open_trade["sl"])
                if sl_hit or open_trade["age"]>=MAX_HOLD_CANDLES:
                    ep=open_trade["sl"] if sl_hit else c["close"]
                    raw=(ep-open_trade["entry"])*d*open_trade["size"]
                    pnl=raw+open_trade["locked_pnl"]
                    r=pnl/open_trade["risk_amount"] if open_trade["risk_amount"] else 0
                    all_r.append(round(r,3))
                    if pnl>0: cw+=1; cl=0
                    else: cl+=1; cw=0
                    balance+=pnl; peak=max(peak,balance)
                    tbd[open_trade["entry_time"].date()]+=1
                    open_trade=None

            if open_trade is not None: continue
            if not (SESSION_START<=hour<SESSION_END) or hour in SKIP_HOURS: continue
            if balance<dsb[day]*(1-DAILY_DD_LIMIT): continue
            if tbd[day]>=MAX_TRADES_PER_DAY: continue
            bias=get_bias(c["time"])
            if bias not in {"STRONG_BUY","STRONG_SELL"}: continue
            es=entry_df.iloc[:i+1].copy()
            er=strategy.analyze_setup(es)
            sig=er.get("signal","WAIT")
            if sig=="WAIT": continue
            req="BUY" if bias=="STRONG_BUY" else "SELL"
            if sig!=req: continue
            atr=float(entry_df.iloc[i]["atr"])
            if pd.isna(atr) or atr<=0: continue
            rp=current_risk(cw,cl); ra=balance*rp; sz=ra/atr
            ep2=float(nc["open"]); d2=1 if sig=="BUY" else -1
            sl2=ep2-d2*atr*SL_ATR_MULT
            open_trade={"type":sig,"entry":ep2,"sl":sl2,"atr":atr,"size":sz,
                        "risk_pct":rp,"risk_amount":ra,"entry_time":nc["time"],
                        "age":0,"peak":ep2,"partial_done":False,"locked_pnl":0.0}
    return all_r

def monte_carlo(r_pool, n_sims=10000, trades_per_month=3.1,
                target=1000, start=100, max_months=60):
    """
    Simulate n_sims paths. Each month draw trades_per_month R-values
    from r_pool, apply to balance with dynamic risk sizing.
    Returns months_to_target for each sim that hit target.
    """
    results = []
    for _ in range(n_sims):
        balance = start
        peak    = start
        cw = cl = 0
        months = 0
        hit = False
        for m in range(max_months):
            n_trades = max(1, int(round(random.gauss(trades_per_month, 0.8))))
            for _ in range(n_trades):
                r = random.choice(r_pool)
                # dynamic risk
                if cw >= STREAK_THRESHOLD:   rp = RISK_HOT
                elif cl >= STREAK_THRESHOLD: rp = RISK_COLD
                else:                        rp = RISK_BASE
                pnl = balance * rp * r
                balance += pnl
                peak = max(peak, balance)
                if r > 0: cw += 1; cl = 0
                else:     cl += 1; cw = 0
                # hard floor - account DD kill
                if balance < peak * (1 - ACCOUNT_DD_LIMIT):
                    balance = 0; break
            if balance <= 0: break
            months += 1
            if balance >= target:
                hit = True
                results.append(months)
                break
        if not hit and balance > 0:
            results.append(None)  # didn't reach target in max_months

    return results

def main():
    mt5.initialize()
    print("Collecting actual trade R-multiples from backtest...")
    r_pool = collect_trades()
    mt5.shutdown()

    if not r_pool:
        print("No trades found.")
        return

    wins   = [r for r in r_pool if r > 0]
    losses = [r for r in r_pool if r <= 0]
    wr     = len(wins)/len(r_pool)*100
    avg_w  = sum(wins)/len(wins)
    avg_l  = sum(losses)/len(losses) if losses else 0
    exp    = (wr/100)*avg_w + (1-wr/100)*avg_l

    print(f"Trade sample: {len(r_pool)} trades | WR={wr:.0f}% | Avg W={avg_w:+.2f}R | Avg L={avg_l:+.2f}R | E={exp:+.3f}R")
    print(f"Running 10,000 Monte Carlo simulations ($100 -> $1,000)...")

    results = monte_carlo(r_pool, n_sims=10000, trades_per_month=3.1,
                          target=1000, start=100, max_months=60)

    hit      = [r for r in results if r is not None]
    missed   = [r for r in results if r is None]
    hit_pct  = len(hit)/len(results)*100

    hit_sorted = sorted(hit)
    p10 = hit_sorted[int(len(hit_sorted)*0.10)] if hit else None
    p25 = hit_sorted[int(len(hit_sorted)*0.25)] if hit else None
    p50 = hit_sorted[int(len(hit_sorted)*0.50)] if hit else None
    p75 = hit_sorted[int(len(hit_sorted)*0.75)] if hit else None
    p90 = hit_sorted[int(len(hit_sorted)*0.90)] if hit else None

    print()
    print("="*60)
    print("  GROWTH PROJECTION: $100 -> $1,000  (10x)")
    print("="*60)
    print(f"  Simulations that hit $1,000 : {hit_pct:.0f}% of paths")
    print(f"  Simulations that didn't hit : {100-hit_pct:.0f}% (bust or too slow)")
    print()
    print("  Time to reach $1,000 (months):")
    print(f"  Best 10% of runs    : {p10} months")
    print(f"  Better half (25th)  : {p25} months")
    print(f"  MEDIAN (50th pct)   : {p50} months  <-- most likely")
    print(f"  Slower half (75th)  : {p75} months")
    print(f"  Worst 10% of runs   : {p90} months")
    print()

    # milestone breakdown
    print("="*60)
    print("  MILESTONES (median path)")
    print("="*60)
    milestones = [200, 300, 500, 750, 1000]
    for target_bal in milestones:
        res = monte_carlo(r_pool, n_sims=5000, trades_per_month=3.1,
                          target=target_bal, start=100, max_months=60)
        h = sorted([r for r in res if r is not None])
        med = h[len(h)//2] if h else None
        pct = len(h)/len(res)*100
        label = f"${target_bal:>5}"
        if med:
            print(f"  {label} : median {med:>2} months  ({pct:.0f}% of paths hit this)")
        else:
            print(f"  {label} : unlikely to reach  ({pct:.0f}% hit it)")

    print()
    print("="*60)
    print("  WHAT DEPOSIT SIZE GETS YOU TO $1,000 FASTER?")
    print("="*60)
    for deposit in [100, 200, 500]:
        res = monte_carlo(r_pool, n_sims=5000, trades_per_month=3.1,
                          target=1000, start=deposit, max_months=60)
        h = sorted([r for r in res if r is not None])
        med = h[len(h)//2] if h else None
        pct = len(h)/len(res)*100
        if med:
            print(f"  Start ${deposit:<4} -> $1,000  median {med:>2} months  ({pct:.0f}% hit it)")
        else:
            print(f"  Start ${deposit:<4} -> $1,000  unlikely  ({pct:.0f}% hit it)")

if __name__ == "__main__":
    main()

"""
Test: what happens when we remove / loosen the RSI momentum filter?
  A: CURRENT   -- momentum_up required (RSI>50 rising 2 bars)
  B: NO MOMENTUM -- skip RSI check entirely
  C: SOFT MOMENTUM -- RSI>45 and just 1 bar rising (relaxed)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import strategy_step_trend as strategy

SYMBOL = "Step Index"
BARS   = 15000
TREND_TF = mt5.TIMEFRAME_M15
ENTRY_TF  = mt5.TIMEFRAME_M5

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

# ── custom signal check ───────────────────────────────────────────────────────
def check_signal(es, bias, mode):
    """
    mode: 'current' | 'no_momentum' | 'soft_momentum'
    Returns 'BUY', 'SELL', or 'WAIT'
    """
    chk  = strategy.analyze_setup(es).get("checks", {})
    is_buy  = bias == "STRONG_BUY"
    is_sell = bias == "STRONG_SELL"

    not_choppy = not chk.get("market_choppy", True)
    trend_ok   = (is_buy  and chk.get("trend_up")) or (is_sell and chk.get("trend_down"))
    pullback   = (is_buy  and chk.get("valid_pullback_buy")) or (is_sell and chk.get("valid_pullback_sell"))
    right_dir  = (is_buy  and chk.get("bullish_candle")) or (is_sell and chk.get("bearish_candle"))
    confirm    = (is_buy  and chk.get("bullish_confirmation")) or (is_sell and chk.get("bearish_confirmation"))

    if mode == "current":
        momentum = (is_buy  and chk.get("momentum_up")) or (is_sell and chk.get("momentum_down"))
    elif mode == "no_momentum":
        momentum = True  # skip check entirely
    elif mode == "soft_momentum":
        # RSI > 45 and just 1 bar of slope
        rsi = chk.get("rsi", 0)
        rsi_slope = chk.get("rsi_slope", 0)
        if is_buy:  momentum = rsi > 45 and rsi_slope > 0
        else:       momentum = rsi < 55 and rsi_slope < 0

    all_ok = not_choppy and trend_ok and pullback and right_dir and confirm and momentum
    if not all_ok: return "WAIT"
    return "BUY" if is_buy else "SELL"

def run_window(start_pos, mode):
    trend_raw = fetch(TREND_TF, BARS, start_pos//3)
    entry_raw = fetch(ENTRY_TF, BARS, start_pos)
    if trend_raw is None or entry_raw is None: return []
    trend_df = strategy.calculate_indicators(trend_raw)
    entry_df = strategy.calculate_indicators(entry_raw)

    trades=[]; balance=START_BALANCE; peak=START_BALANCE
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
                if pnl>0: cw+=1; cl=0
                else: cl+=1; cw=0
                balance+=pnl; peak=max(peak,balance)
                trades.append({
                    "pnl": round(pnl,4),
                    "r_multiple": round(r,3),
                    "result": "WIN" if pnl>0 else "LOSS",
                    "balance": round(balance,4),
                })
                tbd[open_trade["entry_time"].date()]+=1
                open_trade=None

        if open_trade is not None: continue
        if not (SESSION_START<=hour<SESSION_END) or hour in SKIP_HOURS: continue
        if balance<dsb[day]*(1-DAILY_DD_LIMIT): continue
        if tbd[day]>=MAX_TRADES_PER_DAY: continue

        bias=get_bias(c["time"])
        if bias not in {"STRONG_BUY","STRONG_SELL"}: continue

        es=entry_df.iloc[:i+1].copy()
        sig=check_signal(es, bias, mode)
        if sig=="WAIT": continue

        atr=float(entry_df.iloc[i]["atr"])
        if pd.isna(atr) or atr<=0: continue
        rp=current_risk(cw,cl); ra=balance*rp; sz=ra/atr
        ep2=float(nc["open"]); d2=1 if sig=="BUY" else -1
        sl2=ep2-d2*atr*SL_ATR_MULT
        open_trade={"type":sig,"entry":ep2,"sl":sl2,"atr":atr,"size":sz,
                    "risk_pct":rp,"risk_amount":ra,"entry_time":nc["time"],
                    "age":0,"peak":ep2,"partial_done":False,"locked_pnl":0.0}
    return trades

def run_scenario(label, mode):
    all_trades=[]; windows=[]
    for w in range(5):
        trades=run_window(w*BARS, mode)
        all_trades.extend(trades)
        if trades:
            df_w=pd.DataFrame(trades)
            bal=[START_BALANCE]+list(df_w["balance"])
            pk=START_BALANCE; mdd=0
            for b in bal:
                pk=max(pk,b)
                mdd=max(mdd,(pk-b)/pk)
            windows.append({"t":len(trades),"f":bal[-1],"dd":round(mdd*100,1)})
        else:
            windows.append({"t":0,"f":START_BALANCE,"dd":0})
    return {"label":label,"trades":all_trades,"windows":windows}

def print_result(res):
    df=pd.DataFrame(res["trades"]) if res["trades"] else pd.DataFrame()
    wins=df[df["result"]=="WIN"] if not df.empty else pd.DataFrame()
    losses=df[df["result"]=="LOSS"] if not df.empty else pd.DataFrame()
    total=len(df)
    wr=len(wins)/total*100 if total else 0
    net=df["pnl"].sum() if not df.empty else 0
    avg_w=wins["r_multiple"].mean() if len(wins) else 0
    avg_l=losses["r_multiple"].mean() if len(losses) else 0
    exp=(wr/100*avg_w)+((1-wr/100)*avg_l)
    mon=len(wins[wins["r_multiple"]>=3.0]) if len(wins) else 0
    tpm=total/(5*1.72)
    all_dd=[w["dd"] for w in res["windows"]]
    worst_dd=max(all_dd) if all_dd else 0

    print(f"\n  {res['label']}")
    print(f"  {'-'*55}")
    print(f"  Trades/month  : ~{tpm:.1f}   (total {total})")
    print(f"  Win rate      : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Net profit    : ${net:+.2f}")
    print(f"  Avg W / Avg L : +{avg_w:.2f}R / {avg_l:.2f}R")
    print(f"  Expectancy    : {exp:+.3f}R per trade")
    print(f"  Monsters(3R+) : {mon}")
    print(f"  Worst DD      : {worst_dd:.1f}%")
    print(f"  Per window    :", end="")
    for i,w in enumerate(res["windows"]):
        print(f"  W{i+1}:{w['t']}t ${w['f']:.0f} DD{w['dd']:.0f}%", end="")
    print()

def main():
    mt5.initialize()

    print("Running 3-way momentum filter test...")
    res_a=run_scenario("A: CURRENT      (RSI>50 rising 2 bars)", "current")
    print("  A done.")
    res_b=run_scenario("B: NO MOMENTUM  (remove RSI check)", "no_momentum")
    print("  B done.")
    res_c=run_scenario("C: SOFT MOMENTUM (RSI>45, 1 bar slope)", "soft_momentum")
    print("  C done.")

    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print_result(res_a)
    print_result(res_b)
    print_result(res_c)

    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"  {'Scenario':<28} {'T/mo':>5} {'WR':>6} {'Net':>9} {'Exp':>8} {'DD':>7}")
    print(f"  {'-'*58}")
    for res in [res_a, res_b, res_c]:
        df=pd.DataFrame(res["trades"]) if res["trades"] else pd.DataFrame()
        wins=df[df["result"]=="WIN"] if not df.empty else pd.DataFrame()
        losses=df[df["result"]=="LOSS"] if not df.empty else pd.DataFrame()
        total=len(df)
        wr=len(wins)/total*100 if total else 0
        net=df["pnl"].sum() if not df.empty else 0
        avg_w=wins["r_multiple"].mean() if len(wins) else 0
        avg_l=losses["r_multiple"].mean() if len(losses) else 0
        exp=(wr/100*avg_w)+((1-wr/100)*avg_l)
        tpm=total/(5*1.72)
        worst_dd=max([w["dd"] for w in res["windows"]]) if res["windows"] else 0
        print(f"  {res['label'][:28]:<28} {tpm:>5.1f} {wr:>5.0f}% ${net:>+8.2f} {exp:>+7.3f}R {worst_dd:>6.1f}%")

    mt5.shutdown()

if __name__ == "__main__":
    main()

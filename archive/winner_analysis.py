import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import strategy_step_trend as strategy

SYMBOL="Step Index"; BARS=15000
TREND_TF=mt5.TIMEFRAME_M15; ENTRY_TF=mt5.TIMEFRAME_M5
RISK_BASE=0.05; RISK_HOT=0.08; RISK_COLD=0.03; STREAK_THRESHOLD=2
SL_ATR_MULT=1.0; CHANDELIER_MULT=3.0; PARTIAL_R=2.0; PARTIAL_PCT=0.5
MAX_HOLD_CANDLES=96; MAX_TRADES_PER_DAY=6; DAILY_DD_LIMIT=0.03
ACCOUNT_DD_LIMIT=0.15; SESSION_START=9; SESSION_END=19; SKIP_HOURS={11,14}
START_BALANCE=100.0

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

def run_window(start_pos):
    trend_raw = fetch(TREND_TF, BARS, start_pos//3)
    entry_raw = fetch(ENTRY_TF, BARS, start_pos)
    if trend_raw is None or entry_raw is None: return []
    trend_df = strategy.calculate_indicators(trend_raw)
    entry_df = strategy.calculate_indicators(entry_raw)
    trades=[]; balance=START_BALANCE; peak=START_BALANCE
    open_trade=None; tbd={}; dsb={}; cw=0; cl=0

    def get_bias(t):
        slc = trend_df[trend_df["time"] <= t]
        if len(slc) < 220: return "NEUTRAL"
        return strategy.analyze_setup(slc).get("checks", {}).get("trend_bias", "NEUTRAL")

    for i in range(220, len(entry_df)-1):
        c = entry_df.iloc[i]; nc = entry_df.iloc[i+1]
        day = c["time"].date(); hour = c["time"].hour
        tbd.setdefault(day, 0); dsb.setdefault(day, balance)
        if balance < peak * (1 - ACCOUNT_DD_LIMIT): break

        if open_trade is not None:
            open_trade["age"] += 1
            t = open_trade["type"]; atr = open_trade["atr"]; d = 1 if t=="BUY" else -1
            hi, lo = c["high"], c["low"]
            if not open_trade["partial_done"]:
                pt = open_trade["entry"] + d*atr*PARTIAL_R
                if (t=="BUY" and hi>=pt) or (t=="SELL" and lo<=pt):
                    open_trade["locked_pnl"] = (pt - open_trade["entry"]) * d * (open_trade["size"] * PARTIAL_PCT)
                    open_trade["size"] *= (1 - PARTIAL_PCT)
                    open_trade["partial_done"] = True
            if t=="BUY": open_trade["peak"] = max(open_trade["peak"], hi)
            else: open_trade["peak"] = min(open_trade["peak"], lo)
            csl = open_trade["peak"] - d*atr*CHANDELIER_MULT
            if t=="BUY": open_trade["sl"] = max(open_trade["sl"], csl)
            else: open_trade["sl"] = min(open_trade["sl"], csl)
            sl_hit = (t=="BUY" and lo<=open_trade["sl"]) or (t=="SELL" and hi>=open_trade["sl"])
            timed = open_trade["age"] >= MAX_HOLD_CANDLES
            if sl_hit or timed:
                ep = open_trade["sl"] if sl_hit else c["close"]
                raw = (ep - open_trade["entry"]) * d * open_trade["size"]
                pnl = raw + open_trade["locked_pnl"]
                if pnl > 0: cw += 1; cl = 0
                else: cl += 1; cw = 0
                balance += pnl; peak = max(peak, balance)
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": c["time"],
                    "type": t,
                    "entry": open_trade["entry"],
                    "exit": ep,
                    "risk_pct": open_trade["risk_pct"],
                    "risk_amount": open_trade["risk_amount"],
                    "pnl": round(pnl, 4),
                    "r_multiple": round(pnl / open_trade["risk_amount"], 3) if open_trade["risk_amount"] else 0,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "partial": open_trade["partial_done"],
                    "hold_hours": round(open_trade["age"]*5/60, 1),
                    "balance": round(balance, 4),
                })
                tbd[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None: continue
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS: continue
        if balance < dsb[day] * (1 - DAILY_DD_LIMIT): continue
        if tbd[day] >= MAX_TRADES_PER_DAY: continue
        bias = get_bias(c["time"])
        if bias not in {"STRONG_BUY","STRONG_SELL"}: continue
        es = entry_df.iloc[:i+1].copy()
        er = strategy.analyze_setup(es)
        sig = er.get("signal","WAIT")
        if sig == "WAIT": continue
        req = "BUY" if bias=="STRONG_BUY" else "SELL"
        if sig != req: continue
        atr = float(entry_df.iloc[i]["atr"])
        if pd.isna(atr) or atr <= 0: continue
        rp = current_risk(cw, cl); ra = balance*rp; sz = ra/atr
        ep2 = float(nc["open"]); d2 = 1 if sig=="BUY" else -1
        sl2 = ep2 - d2*atr*SL_ATR_MULT
        open_trade = {
            "type": sig, "entry": ep2, "sl": sl2, "atr": atr,
            "size": sz, "risk_pct": rp, "risk_amount": ra,
            "entry_time": nc["time"], "age": 0,
            "peak": ep2, "partial_done": False, "locked_pnl": 0.0,
        }
    return trades


def main():
    mt5.initialize()

    all_trades = []
    for w in range(5):
        all_trades.extend(run_window(w * BARS))

    df = pd.DataFrame(all_trades)
    wins   = df[df["result"]=="WIN"].sort_values("r_multiple", ascending=False)
    losses = df[df["result"]=="LOSS"].sort_values("r_multiple")

    print()
    print("="*65)
    print("  TOP 5 BIGGEST WINNERS")
    print("="*65)
    print(f"  {'Date':<12} {'Dir':<5} {'Risk':<5} {'R':>7} {'PnL':>8} {'Hold':>7}  Partial")
    print("  " + "-"*58)
    for _, r in wins.head(5).iterrows():
        partial_str = "YES - partial close hit" if r["partial"] else "no partial"
        print(f"  {str(r['entry_time'])[:10]:<12} {r['type']:<5} {r['risk_pct']*100:.0f}%   {r['r_multiple']:>+6.3f}R  ${r['pnl']:>+7.2f}  {r['hold_hours']:>5.1f}h  {partial_str}")

    print()
    print("="*65)
    print("  TOP 5 BIGGEST LOSERS")
    print("="*65)
    print(f"  {'Date':<12} {'Dir':<5} {'Risk':<5} {'R':>7} {'PnL':>8} {'Hold':>7}")
    print("  " + "-"*58)
    for _, r in losses.head(5).iterrows():
        print(f"  {str(r['entry_time'])[:10]:<12} {r['type']:<5} {r['risk_pct']*100:.0f}%   {r['r_multiple']:>+6.3f}R  ${r['pnl']:>+7.2f}  {r['hold_hours']:>5.1f}h")

    print()
    print("="*65)
    print("  R DISTRIBUTION -- ALL WINNERS")
    print("="*65)
    bins   = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 999]
    labels = ["0-0.5R","0.5-1R","1-1.5R","1.5-2R","2-3R","3-5R","5R+"]
    for i, (lo_b, hi_b) in enumerate(zip(bins, bins[1:])):
        grp = wins[(wins["r_multiple"] >= lo_b) & (wins["r_multiple"] < hi_b)]
        bar = "#" * len(grp)
        print(f"  {labels[i]:<10}: {len(grp):>3} trades  {bar}")

    print()
    print("="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Total trades      : {len(df)}")
    print(f"  Winners           : {len(wins)}")
    print(f"  Losers            : {len(losses)}")
    print(f"  Best single trade : +{wins.iloc[0]['r_multiple']:.3f}R = ${wins.iloc[0]['pnl']:+.2f}  on {str(wins.iloc[0]['entry_time'])[:10]}")
    print(f"  Worst single trade: {losses.iloc[0]['r_multiple']:.3f}R = ${losses.iloc[0]['pnl']:+.2f}  on {str(losses.iloc[0]['entry_time'])[:10]}")
    print(f"  Avg winner        : +{wins['r_multiple'].mean():.3f}R = ${wins['pnl'].mean():+.2f}")
    print(f"  Avg loser         : {losses['r_multiple'].mean():.3f}R = ${losses['pnl'].mean():+.2f}")
    print(f"  Reward:Risk ratio : {abs(wins['r_multiple'].mean() / losses['r_multiple'].mean()):.2f}x")
    print(f"  Avg hold (winners): {wins['hold_hours'].mean():.1f}h")
    print(f"  Avg hold (losers) : {losses['hold_hours'].mean():.1f}h")
    print(f"  Partial closes    : {df['partial'].sum()} trades hit the 2R partial close target")

    mt5.shutdown()

if __name__ == "__main__":
    main()

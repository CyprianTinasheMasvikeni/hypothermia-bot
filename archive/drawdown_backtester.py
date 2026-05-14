"""
Drawdown-Focused Backtester -- Step Index
Strategy : step_trend  (M15 trend bias + M5 entry)
Risk     : Dynamic 3% / 5% / 8% -- matches live_bot.py streak logic
Session  : 09:00 - 19:00 GMT  |  skip hours 11, 14

What this script reports (per window AND across all windows):
  • Max drawdown %  (worst peak-to-trough)
  • Max drawdown duration  (candles from peak to recovery)
  • Longest consecutive loss streak
  • Win rate, net profit, avg R per trade
  • Monthly breakdown -- trades / WR / net / max intramonth DD
  • Recovery factor  (net / max DD)
  • Calmar-style ratio  (annualised return / max DD)

Data coverage:
  5 windows x 15,000 M5 bars = 375,000 minutes ~ 8.7 calendar months
  (Step Index is 24/7 -- no market-closed gaps)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import strategy_step_trend as strategy

# -- CONFIG (mirrors live_bot.py exactly) -------------------------------------
SYMBOL             = "Step Index"
TREND_TF           = mt5.TIMEFRAME_M15
ENTRY_TF           = mt5.TIMEFRAME_M5
BARS               = 15000
WINDOWS            = 5

RISK_BASE          = 0.05
RISK_HOT           = 0.08
RISK_COLD          = 0.03
STREAK_THRESHOLD   = 2

SL_ATR_MULT        = 1.0
CHANDELIER_MULT    = 3.0
PARTIAL_R          = 2.0
PARTIAL_PCT        = 0.5
MAX_HOLD_CANDLES   = 96   # 96 x M5 = 8 hours
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT     = 0.03
ACCOUNT_DD_LIMIT   = 0.15
SESSION_START      = 9
SESSION_END        = 19
SKIP_HOURS         = {11, 14}
START_BALANCE      = 100.0


# -- MT5 ----------------------------------------------------------------------

def connect():
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        return False
    info = mt5.account_info()
    if info:
        print(f"Connected | account={info.login} | server={info.server}")
    return True


def fetch(tf, bars, start_pos=0):
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, start_pos, bars)
    if rates is None or len(rates) < 300:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


# -- DYNAMIC RISK --------------------------------------------------------------

def current_risk(consec_wins, consec_losses, risk_hot=RISK_HOT):
    if consec_wins >= STREAK_THRESHOLD:
        return risk_hot
    if consec_losses >= STREAK_THRESHOLD:
        return RISK_COLD
    return RISK_BASE


# -- SIMULATE -----------------------------------------------------------------

def simulate(trend_df, entry_df, risk_hot=RISK_HOT):
    trend_df = strategy.calculate_indicators(trend_df)
    entry_df = strategy.calculate_indicators(entry_df)

    trades       = []
    balance      = START_BALANCE
    peak_balance = START_BALANCE
    open_trade   = None
    trades_by_day         = {}
    daily_start_balance   = {}
    consec_wins           = 0
    consec_losses         = 0
    account_shutdown      = False
    balance_curve         = [(entry_df.iloc[220]["time"], START_BALANCE)]

    def get_trend_bias(entry_time):
        slc = trend_df[trend_df["time"] <= entry_time]
        if len(slc) < 220:
            return "NEUTRAL"
        res = strategy.analyze_setup(slc)
        return res.get("checks", {}).get("trend_bias", "NEUTRAL")

    for i in range(220, len(entry_df) - 1):
        candle      = entry_df.iloc[i]
        next_candle = entry_df.iloc[i + 1]
        day         = candle["time"].date()
        hour        = candle["time"].hour
        trades_by_day.setdefault(day, 0)
        daily_start_balance.setdefault(day, balance)

        if account_shutdown:
            break
        if balance < peak_balance * (1 - ACCOUNT_DD_LIMIT):
            account_shutdown = True
            break

        # -- Manage open trade -------------------------------------------------
        if open_trade is not None:
            open_trade["age"] += 1
            t   = open_trade["type"]
            atr = open_trade["atr"]
            d   = 1 if t == "BUY" else -1
            hi, lo = candle["high"], candle["low"]

            # Partial close at PARTIAL_R
            if not open_trade["partial_done"]:
                partial_target = open_trade["entry"] + d * atr * PARTIAL_R
                hit_partial = (t == "BUY" and hi >= partial_target) or \
                              (t == "SELL" and lo <= partial_target)
                if hit_partial:
                    locked = (partial_target - open_trade["entry"]) * d * \
                             (open_trade["size"] * PARTIAL_PCT)
                    open_trade["locked_pnl"]   = locked
                    open_trade["size"]        *= (1.0 - PARTIAL_PCT)
                    open_trade["partial_done"] = True

            # Update peak for Chandelier
            if t == "BUY":
                open_trade["peak"] = max(open_trade["peak"], hi)
            else:
                open_trade["peak"] = min(open_trade["peak"], lo)

            # Chandelier SL only moves in favour
            chand_sl = open_trade["peak"] - d * atr * CHANDELIER_MULT
            if t == "BUY":
                open_trade["sl"] = max(open_trade["sl"], chand_sl)
            else:
                open_trade["sl"] = min(open_trade["sl"], chand_sl)

            sl_hit = (t == "BUY" and lo <= open_trade["sl"]) or \
                     (t == "SELL" and hi >= open_trade["sl"])
            timed  = open_trade["age"] >= MAX_HOLD_CANDLES

            if sl_hit or timed:
                exit_price = open_trade["sl"] if sl_hit else candle["close"]
                raw_pnl    = (exit_price - open_trade["entry"]) * d * open_trade["size"]
                total_pnl  = raw_pnl + open_trade["locked_pnl"]

                if (open_trade["sl"] > open_trade["entry"] and t == "BUY") or \
                   (open_trade["sl"] < open_trade["entry"] and t == "SELL"):
                    result = "WIN"
                elif timed:
                    result = "TIME_EXIT"
                else:
                    result = "LOSS"
                if total_pnl > 0 and result == "LOSS":
                    result = "WIN"
                elif total_pnl <= 0 and result == "WIN":
                    result = "LOSS"

                balance += total_pnl
                peak_balance = max(peak_balance, balance)
                balance_curve.append((candle["time"], balance))

                r_mult = total_pnl / open_trade["risk_amount"] if open_trade["risk_amount"] else 0
                if total_pnl > 0:
                    consec_wins   += 1
                    consec_losses  = 0
                else:
                    consec_losses += 1
                    consec_wins    = 0

                trades.append({
                    "entry_time":    open_trade["entry_time"],
                    "exit_time":     candle["time"],
                    "type":          t,
                    "entry":         open_trade["entry"],
                    "exit":          exit_price,
                    "sl_initial":    open_trade["sl_initial"],
                    "atr":           atr,
                    "risk_pct":      open_trade["risk_pct"],
                    "risk_amount":   open_trade["risk_amount"],
                    "pnl":           round(total_pnl, 4),
                    "r_multiple":    round(r_mult, 3),
                    "result":        result,
                    "reason":        "SL" if sl_hit else "TIME",
                    "partial_done":  open_trade["partial_done"],
                    "balance_after": round(balance, 4),
                    "consec_wins_after":   consec_wins,
                    "consec_losses_after": consec_losses,
                })
                trades_by_day[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None:
            continue

        # -- Entry guards ------------------------------------------------------
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS:
            continue
        if balance < daily_start_balance[day] * (1 - DAILY_DD_LIMIT):
            continue
        if trades_by_day[day] >= MAX_TRADES_PER_DAY:
            continue

        # -- Signal ------------------------------------------------------------
        trend_bias = get_trend_bias(candle["time"])
        if trend_bias not in {"STRONG_BUY", "STRONG_SELL"}:
            continue

        entry_slice = entry_df.iloc[:i + 1].copy()
        entry_res   = strategy.analyze_setup(entry_slice)
        entry_sig   = entry_res.get("signal", "WAIT")
        if entry_sig == "WAIT":
            continue

        required = "BUY" if trend_bias == "STRONG_BUY" else "SELL"
        if entry_sig != required:
            continue

        atr = float(entry_df.iloc[i]["atr"])
        if pd.isna(atr) or atr <= 0:
            continue

        risk_pct    = current_risk(consec_wins, consec_losses, risk_hot=risk_hot)
        risk_amount = balance * risk_pct
        size        = risk_amount / atr
        entry_price = float(next_candle["open"])
        d           = 1 if entry_sig == "BUY" else -1
        sl          = entry_price - d * atr * SL_ATR_MULT

        open_trade = {
            "type":         entry_sig,
            "entry":        entry_price,
            "sl":           sl,
            "sl_initial":   sl,
            "atr":          atr,
            "size":         size,
            "risk_pct":     risk_pct,
            "risk_amount":  risk_amount,
            "entry_time":   next_candle["time"],
            "age":          0,
            "peak":         entry_price,
            "partial_done": False,
            "locked_pnl":   0.0,
        }

    # Force-close any open trade
    if open_trade is not None:
        f   = entry_df.iloc[-1]
        raw = (float(f["close"]) - open_trade["entry"]) * (1 if open_trade["type"] == "BUY" else -1) * open_trade["size"]
        total_pnl = raw + open_trade["locked_pnl"]
        balance  += total_pnl
        balance_curve.append((f["time"], balance))
        trades.append({
            "entry_time":    open_trade["entry_time"],
            "exit_time":     f["time"],
            "type":          open_trade["type"],
            "entry":         open_trade["entry"],
            "exit":          float(f["close"]),
            "sl_initial":    open_trade["sl_initial"],
            "atr":           open_trade["atr"],
            "risk_pct":      open_trade["risk_pct"],
            "risk_amount":   open_trade["risk_amount"],
            "pnl":           round(total_pnl, 4),
            "r_multiple":    round(total_pnl / open_trade["risk_amount"], 3) if open_trade["risk_amount"] else 0,
            "result":        "WIN" if total_pnl > 0 else "LOSS",
            "reason":        "FORCED",
            "partial_done":  open_trade["partial_done"],
            "balance_after": round(balance, 4),
            "consec_wins_after":   consec_wins,
            "consec_losses_after": consec_losses,
        })

    return trades, balance_curve


# -- DRAWDOWN STATS ------------------------------------------------------------

def compute_drawdown(balance_curve):
    """Returns max_dd_pct, max_dd_abs, max_dd_duration_points"""
    if len(balance_curve) < 2:
        return 0.0, 0.0, 0

    balances = [b for _, b in balance_curve]
    peak     = balances[0]
    peak_idx = 0
    max_dd   = 0.0
    max_dd_abs = 0.0
    max_dd_dur = 0

    trough_idx  = 0
    in_dd       = False
    dd_start    = 0

    for i, b in enumerate(balances):
        if b >= peak:
            peak     = b
            peak_idx = i
            if in_dd:
                dur = i - dd_start
                if dur > max_dd_dur:
                    max_dd_dur = dur
            in_dd    = False
        else:
            dd = (peak - b) / peak
            dd_abs = peak - b
            if dd > max_dd:
                max_dd     = dd
                max_dd_abs = dd_abs
            if not in_dd:
                in_dd    = True
                dd_start = peak_idx

    if in_dd:
        dur = len(balances) - 1 - dd_start
        if dur > max_dd_dur:
            max_dd_dur = dur

    return round(max_dd * 100, 2), round(max_dd_abs, 4), max_dd_dur


def longest_loss_streak(trades):
    best = streak = 0
    for t in trades:
        if t["result"] == "LOSS":
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def monthly_stats(trades):
    if not trades:
        return []
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    rows = []
    for month, grp in df.groupby("month"):
        wins   = (grp["result"] == "WIN").sum()
        losses = (grp["result"] == "LOSS").sum()
        total  = len(grp)
        net    = grp["pnl"].sum()
        wr     = wins / total * 100 if total else 0

        # intramonth drawdown from balance curve points in this month
        bals = grp["balance_after"].tolist()
        if bals:
            pk = bals[0]
            mdd = 0.0
            for b in bals:
                pk = max(pk, b)
                mdd = max(mdd, (pk - b) / pk * 100 if pk else 0)
        else:
            mdd = 0.0

        rows.append({
            "month":  str(month),
            "trades": total,
            "wins":   wins,
            "losses": losses,
            "wr_pct": round(wr, 1),
            "net":    round(net, 4),
            "max_dd_pct": round(mdd, 2),
            "tag":    "PROFIT" if net > 0 else "LOSS",
        })
    return rows


# -- PRINT HELPERS -------------------------------------------------------------

def print_window_stats(w_idx, trades, balance_curve, start_time, end_time):
    total = len(trades)
    if total == 0:
        print(f"  Window {w_idx+1}: NO TRADES")
        return

    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = sum(1 for t in trades if t["result"] == "LOSS")
    net     = sum(t["pnl"] for t in trades)
    wr      = wins / total * 100
    avg_r   = sum(t["r_multiple"] for t in trades) / total
    max_dd, max_dd_abs, dd_dur = compute_drawdown(balance_curve)
    streak  = longest_loss_streak(trades)
    final_b = trades[-1]["balance_after"]
    profit_factor = abs(sum(t["pnl"] for t in trades if t["pnl"] > 0)) / \
                    max(abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)), 0.001)
    recovery = net / max(max_dd_abs, 0.001)

    risk_used = {}
    for t in trades:
        r = t.get("risk_pct", 0.05)
        risk_used[r] = risk_used.get(r, 0) + 1

    print(f"\n  Window {w_idx+1}  [{start_time.date()} -> {end_time.date()}]")
    print(f"  {'-'*52}")
    print(f"  Trades: {total}  |  Wins: {wins}  |  Losses: {losses}  |  WR: {wr:.1f}%")
    print(f"  Net P&L:        ${net:>+8.2f}  (on $100)")
    print(f"  Final balance:  ${final_b:>8.2f}")
    print(f"  Avg R/trade:    {avg_r:>+6.3f}R")
    print(f"  Max Drawdown:   {max_dd:>6.2f}%  (${max_dd_abs:.2f})  -- lasted {dd_dur} balance points")
    print(f"  Longest streak: {streak} consecutive losses")
    print(f"  Profit factor:  {profit_factor:.2f}")
    print(f"  Recovery factor:{recovery:.2f}  (net / max_dd_abs)")
    risk_str = "  ".join(f"{int(k*100)}%={v}tr" for k, v in sorted(risk_used.items()))
    print(f"  Risk breakdown: {risk_str}")


def print_monthly_breakdown(all_trades):
    rows = monthly_stats(all_trades)
    if not rows:
        print("  No monthly data.")
        return
    print(f"\n  {'Month':<12} | {'Tr':>4} | {'WR%':>6} | {'Net$100':>8} | {'Net$1000':>9} | {'MaxDD%':>7} | Tag")
    print(f"  {'-'*72}")
    for r in rows:
        print(f"  {r['month']:<12} | {r['trades']:>4} | {r['wr_pct']:>5.1f}% | "
              f"${r['net']:>+7.2f} | ${r['net']*10:>+8.2f} | {r['max_dd_pct']:>6.2f}% | [{r['tag']}]")


def print_combined_summary(all_trades, all_curves):
    if not all_trades:
        print("No trades across all windows.")
        return

    total  = len(all_trades)
    wins   = sum(1 for t in all_trades if t["result"] == "WIN")
    losses = sum(1 for t in all_trades if t["result"] == "LOSS")
    net    = sum(t["pnl"] for t in all_trades)
    wr     = wins / total * 100
    avg_r  = sum(t["r_multiple"] for t in all_trades) / total

    # Rebuild a single balance curve treating each window independently
    combined_curve = []
    for curve in all_curves:
        combined_curve.extend(curve)
    combined_curve.sort(key=lambda x: x[0])

    max_dd, max_dd_abs, dd_dur = compute_drawdown(combined_curve)
    streak = longest_loss_streak(all_trades)
    pf = abs(sum(t["pnl"] for t in all_trades if t["pnl"] > 0)) / \
         max(abs(sum(t["pnl"] for t in all_trades if t["pnl"] < 0)), 0.001)
    recovery = net / max(max_dd_abs, 0.001)

    total_days = (all_trades[-1]["exit_time"] - all_trades[0]["entry_time"]).days
    annual_return = (net / START_BALANCE) / max(total_days / 365, 0.001) * 100
    calmar = annual_return / max(max_dd, 0.001)

    print(f"\n{'='*60}")
    print("  COMBINED SUMMARY -- ALL WINDOWS")
    print(f"{'='*60}")
    print(f"  Total trades:     {total}  ({wins}W / {losses}L / {total-wins-losses} other)")
    print(f"  Win rate:         {wr:.1f}%")
    print(f"  Net P&L:          ${net:>+8.2f}  on $100 account")
    print(f"  Net on $1,000:    ${net*10:>+9.2f}")
    print(f"  Net on $10,000:   ${net*100:>+10.2f}")
    print(f"  Avg R per trade:  {avg_r:>+.3f}R")
    print()
    print(f"  -- DRAWDOWN ------------------------------------------")
    print(f"  Max drawdown:     {max_dd:.2f}%  (${max_dd_abs:.2f} from peak)")
    print(f"  DD duration:      {dd_dur} balance updates (~ trades)")
    print(f"  Longest L-streak: {streak} consecutive losses")
    print()
    print(f"  -- RISK METRICS --------------------------------------")
    print(f"  Profit factor:    {pf:.2f}")
    print(f"  Recovery factor:  {recovery:.2f}  (net profit / max DD $)")
    print(f"  Annualised return:{annual_return:.1f}%  (over {total_days} days)")
    print(f"  Calmar ratio:     {calmar:.2f}  (ann. return / max DD %)")
    print()

    # Risk distribution
    risk_dist = {}
    for t in all_trades:
        k = t.get("risk_pct", 0.05)
        risk_dist[k] = risk_dist.get(k, 0) + 1
    total_tr = sum(risk_dist.values())
    print("  -- DYNAMIC RISK USAGE --------------------------------")
    for k in sorted(risk_dist):
        pct = risk_dist[k] / total_tr * 100
        label = "COLD(3%)" if k == 0.03 else ("HOT(8%)" if k == 0.08 else "BASE(5%)")
        print(f"  {label}: {risk_dist[k]} trades  ({pct:.1f}%)")


# -- SCENARIO RUNNER ----------------------------------------------------------

def run_scenario(windows_data, risk_hot):
    all_trades = []
    all_curves = []
    for trend_raw, entry_raw in windows_data:
        trades, curve = simulate(trend_raw, entry_raw, risk_hot=risk_hot)
        all_trades.extend(trades)
        all_curves.append(curve)
    all_trades.sort(key=lambda t: t["entry_time"])
    return all_trades, all_curves


def scenario_summary(all_trades, all_curves):
    if not all_trades:
        return {}
    total  = len(all_trades)
    wins   = sum(1 for t in all_trades if t["result"] == "WIN")
    losses = sum(1 for t in all_trades if t["result"] == "LOSS")
    net    = sum(t["pnl"] for t in all_trades)
    wr     = wins / total * 100
    avg_r  = sum(t["r_multiple"] for t in all_trades) / total

    combined_curve = sorted([pt for c in all_curves for pt in c], key=lambda x: x[0])
    max_dd, max_dd_abs, dd_dur = compute_drawdown(combined_curve)
    streak = longest_loss_streak(all_trades)
    pf = abs(sum(t["pnl"] for t in all_trades if t["pnl"] > 0)) / \
         max(abs(sum(t["pnl"] for t in all_trades if t["pnl"] < 0)), 0.001)
    recovery = net / max(max_dd_abs, 0.001)
    total_days = (all_trades[-1]["exit_time"] - all_trades[0]["entry_time"]).days
    annual_return = (net / START_BALANCE) / max(total_days / 365, 0.001) * 100
    calmar = annual_return / max(max_dd, 0.001)

    risk_dist = {}
    for t in all_trades:
        k = t.get("risk_pct", 0.05)
        risk_dist[k] = risk_dist.get(k, 0) + 1

    return {
        "total": total, "wins": wins, "losses": losses,
        "net": net, "wr": wr, "avg_r": avg_r,
        "max_dd": max_dd, "max_dd_abs": max_dd_abs, "dd_dur": dd_dur,
        "streak": streak, "pf": pf, "recovery": recovery,
        "annual_return": annual_return, "calmar": calmar,
        "risk_dist": risk_dist,
    }


def print_comparison(s1, label1, s2, label2):
    def arrow(v1, v2, higher_is_better=True):
        if higher_is_better:
            return "  <-- BETTER" if v1 > v2 else ("  <-- BETTER" if v2 > v1 else "")
        else:
            return "  <-- BETTER" if v1 < v2 else ("  <-- BETTER" if v2 < v1 else "")

    w = 12
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<26} {label1:>{w}}   {label2:>{w}}")
    print(f"  {'-'*60}")

    metrics = [
        ("Net P&L ($100 acc)",  f"${s1['net']:>+8.2f}",   f"${s2['net']:>+8.2f}",   s1['net'],   s2['net'],   True),
        ("Net P&L ($1,000)",    f"${s1['net']*10:>+8.2f}", f"${s2['net']*10:>+8.2f}", s1['net'],  s2['net'],   True),
        ("Win rate",            f"{s1['wr']:>8.1f}%",      f"{s2['wr']:>8.1f}%",      s1['wr'],   s2['wr'],    True),
        ("Avg R/trade",         f"{s1['avg_r']:>+8.3f}R",  f"{s2['avg_r']:>+8.3f}R",  s1['avg_r'],s2['avg_r'], True),
        ("Max Drawdown",        f"{s1['max_dd']:>7.2f}%",  f"{s2['max_dd']:>7.2f}%",  s1['max_dd'],s2['max_dd'],False),
        ("Max DD ($)",          f"${s1['max_dd_abs']:>7.2f}", f"${s2['max_dd_abs']:>7.2f}", s1['max_dd_abs'],s2['max_dd_abs'],False),
        ("Longest L-streak",    f"{s1['streak']:>8}",      f"{s2['streak']:>8}",       s1['streak'],s2['streak'],False),
        ("Profit factor",       f"{s1['pf']:>8.2f}",       f"{s2['pf']:>8.2f}",        s1['pf'],   s2['pf'],    True),
        ("Recovery factor",     f"{s1['recovery']:>8.2f}", f"{s2['recovery']:>8.2f}",  s1['recovery'],s2['recovery'],True),
        ("Ann. return",         f"{s1['annual_return']:>7.1f}%", f"{s2['annual_return']:>7.1f}%", s1['annual_return'],s2['annual_return'],True),
        ("Calmar ratio",        f"{s1['calmar']:>8.2f}",   f"{s2['calmar']:>8.2f}",    s1['calmar'],s2['calmar'],True),
    ]

    for name, v1_str, v2_str, v1, v2, hib in metrics:
        best = ""
        if v1 != v2:
            best = f"  << {label1}" if (hib and v1 > v2) or (not hib and v1 < v2) else f"  << {label2}"
        print(f"  {name:<26} {v1_str:>{w}}   {v2_str:>{w}}{best}")

    print(f"\n  Dynamic risk breakdown:")
    all_keys = sorted(set(list(s1['risk_dist'].keys()) + list(s2['risk_dist'].keys())))
    for k in all_keys:
        lbl = "COLD(3%)" if k == 0.03 else ("HOT" if k == RISK_HOT or k != 0.05 else "BASE(5%)")
        if k != 0.03 and k != 0.05:
            lbl = f"HOT({int(k*100)}%)"
        n1 = s1['risk_dist'].get(k, 0)
        n2 = s2['risk_dist'].get(k, 0)
        t1 = s1['total']
        t2 = s2['total']
        print(f"  {lbl:<12}: {n1:>3} trades ({n1/t1*100:.1f}%)   vs   {n2:>3} trades ({n2/t2*100:.1f}%)")


# -- MAIN ---------------------------------------------------------------------

def main():
    if not connect():
        return

    minutes_per_window = BARS * 5
    days_per_window    = minutes_per_window / (60 * 24)
    total_days         = days_per_window * WINDOWS
    total_months       = total_days / 30.44

    print(f"\nStrategy  : step_trend  |  Symbol: {SYMBOL}")
    print(f"Session   : {SESSION_START:02d}:00-{SESSION_END:02d}:00 GMT  (skip 11, 14)")
    print(f"Exit      : Chandelier {CHANDELIER_MULT}xATR  |  Partial 50% @ {PARTIAL_R}R  |  Max hold {MAX_HOLD_CANDLES} M5 candles ({MAX_HOLD_CANDLES*5//60}h)")
    print(f"Windows   : {WINDOWS} x {BARS:,} M5 bars = {total_days:.0f} calendar days ~{total_months:.1f} months")
    print(f"Comparing : HOT risk 8% vs HOT risk 6%  (cold=3%, base=5% unchanged)")

    # Fetch all data once — both scenarios reuse the same candles
    print("\nFetching data...")
    windows_data = []
    for w in range(WINDOWS):
        start_pos = w * BARS
        trend_raw = fetch(TREND_TF, BARS, start_pos // 3)
        entry_raw = fetch(ENTRY_TF, BARS, start_pos)
        if trend_raw is not None and entry_raw is not None:
            windows_data.append((trend_raw, entry_raw))
        else:
            print(f"  Window {w+1}: not enough data, skipped")

    print(f"  Loaded {len(windows_data)}/{WINDOWS} windows. Running simulations...")

    # Scenario A: original 8% hot
    trades_a, curves_a = run_scenario(windows_data, risk_hot=0.08)
    stats_a = scenario_summary(trades_a, curves_a)

    # Scenario B: capped 6% hot
    trades_b, curves_b = run_scenario(windows_data, risk_hot=0.06)
    stats_b = scenario_summary(trades_b, curves_b)

    # Head-to-head
    print_comparison(stats_a, "8% HOT", stats_b, "6% HOT")

    # Monthly breakdown for both
    print(f"\n{'='*70}")
    print("  MONTHLY BREAKDOWN -- 8% HOT (original)")
    print(f"{'='*70}")
    print_monthly_breakdown(trades_a)

    print(f"\n{'='*70}")
    print("  MONTHLY BREAKDOWN -- 6% HOT (capped)")
    print(f"{'='*70}")
    print_monthly_breakdown(trades_b)

    # Verdict
    print(f"\n{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    dd_saved = stats_a['max_dd'] - stats_b['max_dd']
    profit_lost = stats_a['net'] - stats_b['net']
    calmar_gain = stats_b['calmar'] - stats_a['calmar']
    print(f"  Capping HOT from 8% to 6%:")
    print(f"    Drawdown reduced by : {dd_saved:+.2f}%  (from {stats_a['max_dd']:.2f}% to {stats_b['max_dd']:.2f}%)")
    print(f"    Profit change       : ${profit_lost:+.2f} on $100  (from ${stats_a['net']:.2f} to ${stats_b['net']:.2f})")
    print(f"    Calmar ratio change : {calmar_gain:+.2f}  (from {stats_a['calmar']:.2f} to {stats_b['calmar']:.2f})")
    if dd_saved > 0 and stats_b['calmar'] >= stats_a['calmar']:
        print(f"  RECOMMENDATION: Use 6% -- lower risk, same or better risk-adjusted return.")
    elif dd_saved > 0 and profit_lost < stats_a['net'] * 0.15:
        print(f"  RECOMMENDATION: Use 6% -- saves {dd_saved:.1f}% drawdown for only ${abs(profit_lost):.2f} less profit.")
    else:
        print(f"  RECOMMENDATION: Trade-off is significant -- review monthly breakdown above.")

    mt5.shutdown()


if __name__ == "__main__":
    main()

"""
Monster Trade Optimiser -- Step Index
Compares two exit strategies on the same historical data:

  CURRENT  : Partial 50% at 2R  |  Chandelier fixed 3xATR
  NEW      : Partial 25% at 2R  |  Progressive Chandelier
               0-2R  -> 3.0xATR  (loose, let it breathe)
               2-4R  -> 2.5xATR  (protect partial gains)
               4R+   -> 1.5xATR  (lock in the monster)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
import pandas as pd
import strategy_step_trend as strategy

SYMBOL          = "Step Index"
TREND_TF        = mt5.TIMEFRAME_M15
ENTRY_TF        = mt5.TIMEFRAME_M5
BARS            = 15000
WINDOWS         = 5
START_BALANCE   = 100.0

RISK_BASE       = 0.05
RISK_HOT        = 0.08
RISK_COLD       = 0.03
STREAK_THRESHOLD= 2
SL_ATR_MULT     = 1.0
MAX_HOLD_CANDLES= 96
MAX_TRADES_PER_DAY = 6
DAILY_DD_LIMIT  = 0.03
ACCOUNT_DD_LIMIT= 0.15
SESSION_START   = 9
SESSION_END     = 19
SKIP_HOURS      = {11, 14}

# ── EXIT CONFIGS ──────────────────────────────────────────────────────────────
CURRENT = {
    "label":        "CURRENT  (50% partial @ 2R, fixed 3xATR)",
    "partial_pct":  0.50,
    "partial_r":    2.0,
    "chand_tiers":  [(0, 3.0)],          # always 3xATR
}

NEW = {
    "label":        "NEW      (25% partial @ 2R, progressive Chandelier 3/2.5/2x)",
    "partial_pct":  0.25,
    "partial_r":    2.0,
    "chand_tiers":  [(0, 3.0), (2.0, 2.5), (4.0, 2.0)],
}


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


def current_risk(cw, cl):
    if cw >= STREAK_THRESHOLD: return RISK_HOT
    if cl >= STREAK_THRESHOLD: return RISK_COLD
    return RISK_BASE


def chandelier_mult(r_so_far, tiers):
    """Pick the tightest applicable tier based on current R reached."""
    mult = tiers[0][1]
    for min_r, m in tiers:
        if r_so_far >= min_r:
            mult = m
    return mult


def simulate(trend_df, entry_df, cfg):
    trend_df = strategy.calculate_indicators(trend_df)
    entry_df = strategy.calculate_indicators(entry_df)

    PARTIAL_PCT = cfg["partial_pct"]
    PARTIAL_R   = cfg["partial_r"]
    TIERS       = cfg["chand_tiers"]

    trades       = []
    balance      = START_BALANCE
    peak_balance = START_BALANCE
    open_trade   = None
    tbd = {}; dsb = {}
    cw = 0; cl = 0

    def get_bias(t):
        slc = trend_df[trend_df["time"] <= t]
        if len(slc) < 220: return "NEUTRAL"
        return strategy.analyze_setup(slc).get("checks", {}).get("trend_bias", "NEUTRAL")

    for i in range(220, len(entry_df) - 1):
        c  = entry_df.iloc[i]
        nc = entry_df.iloc[i + 1]
        day  = c["time"].date()
        hour = c["time"].hour
        tbd.setdefault(day, 0)
        dsb.setdefault(day, balance)

        if balance < peak_balance * (1 - ACCOUNT_DD_LIMIT):
            break

        if open_trade is not None:
            open_trade["age"] += 1
            t   = open_trade["type"]
            atr = open_trade["atr"]
            d   = 1 if t == "BUY" else -1
            hi, lo = c["high"], c["low"]

            # Partial close (one-time)
            if not open_trade["partial_done"]:
                pt = open_trade["entry"] + d * atr * PARTIAL_R
                if (t == "BUY" and hi >= pt) or (t == "SELL" and lo <= pt):
                    open_trade["locked_pnl"] = (pt - open_trade["entry"]) * d * (open_trade["size"] * PARTIAL_PCT)
                    open_trade["size"]       *= (1.0 - PARTIAL_PCT)
                    open_trade["partial_done"] = True

            # Update peak
            if t == "BUY":
                open_trade["peak"] = max(open_trade["peak"], hi)
            else:
                open_trade["peak"] = min(open_trade["peak"], lo)

            # Current R reached (used to pick Chandelier tier)
            r_now = abs(open_trade["peak"] - open_trade["entry"]) / atr
            cm    = chandelier_mult(r_now, TIERS)

            # Progressive Chandelier SL -- only moves in favour
            csl = open_trade["peak"] - d * atr * cm
            if t == "BUY":
                open_trade["sl"] = max(open_trade["sl"], csl)
            else:
                open_trade["sl"] = min(open_trade["sl"], csl)

            sl_hit = (t == "BUY" and lo <= open_trade["sl"]) or \
                     (t == "SELL" and hi >= open_trade["sl"])
            timed  = open_trade["age"] >= MAX_HOLD_CANDLES

            if sl_hit or timed:
                ep        = open_trade["sl"] if sl_hit else c["close"]
                raw       = (ep - open_trade["entry"]) * d * open_trade["size"]
                total_pnl = raw + open_trade["locked_pnl"]

                if total_pnl > 0: cw += 1; cl = 0
                else: cl += 1; cw = 0

                balance     += total_pnl
                peak_balance = max(peak_balance, balance)
                r_mult       = total_pnl / open_trade["risk_amount"] if open_trade["risk_amount"] else 0
                peak_r       = abs(open_trade["peak"] - open_trade["entry"]) / atr

                trades.append({
                    "entry_time":  open_trade["entry_time"],
                    "exit_time":   c["time"],
                    "type":        t,
                    "entry":       open_trade["entry"],
                    "exit":        ep,
                    "risk_pct":    open_trade["risk_pct"],
                    "risk_amount": open_trade["risk_amount"],
                    "pnl":         round(total_pnl, 4),
                    "r_multiple":  round(r_mult, 3),
                    "peak_r":      round(peak_r, 3),
                    "result":      "WIN" if total_pnl > 0 else "LOSS",
                    "partial":     open_trade["partial_done"],
                    "hold_hours":  round(open_trade["age"] * 5 / 60, 1),
                    "balance":     round(balance, 4),
                })
                tbd[open_trade["entry_time"].date()] += 1
                open_trade = None

        if open_trade is not None: continue
        if not (SESSION_START <= hour < SESSION_END) or hour in SKIP_HOURS: continue
        if balance < dsb[day] * (1 - DAILY_DD_LIMIT): continue
        if tbd[day] >= MAX_TRADES_PER_DAY: continue

        bias = get_bias(c["time"])
        if bias not in {"STRONG_BUY", "STRONG_SELL"}: continue
        es  = entry_df.iloc[:i + 1].copy()
        er  = strategy.analyze_setup(es)
        sig = er.get("signal", "WAIT")
        if sig == "WAIT": continue
        req = "BUY" if bias == "STRONG_BUY" else "SELL"
        if sig != req: continue

        atr = float(entry_df.iloc[i]["atr"])
        if pd.isna(atr) or atr <= 0: continue

        rp  = current_risk(cw, cl)
        ra  = balance * rp
        sz  = ra / atr
        ep2 = float(nc["open"])
        d2  = 1 if sig == "BUY" else -1
        sl2 = ep2 - d2 * atr * SL_ATR_MULT

        open_trade = {
            "type": sig, "entry": ep2, "sl": sl2, "atr": atr,
            "size": sz, "risk_pct": rp, "risk_amount": ra,
            "entry_time": nc["time"], "age": 0,
            "peak": ep2, "partial_done": False, "locked_pnl": 0.0,
        }

    if open_trade is not None:
        f   = entry_df.iloc[-1]
        raw = (float(f["close"]) - open_trade["entry"]) * (1 if open_trade["type"] == "BUY" else -1) * open_trade["size"]
        pnl = raw + open_trade["locked_pnl"]
        balance += pnl
        r_mult = pnl / open_trade["risk_amount"] if open_trade["risk_amount"] else 0
        trades.append({
            "entry_time": open_trade["entry_time"], "exit_time": f["time"],
            "type": open_trade["type"], "entry": open_trade["entry"], "exit": float(f["close"]),
            "risk_pct": open_trade["risk_pct"], "risk_amount": open_trade["risk_amount"],
            "pnl": round(pnl, 4), "r_multiple": round(r_mult, 3),
            "peak_r": round(abs(open_trade["peak"] - open_trade["entry"]) / open_trade["atr"], 3),
            "result": "WIN" if pnl > 0 else "LOSS",
            "partial": open_trade["partial_done"], "hold_hours": round(open_trade["age"]*5/60, 1),
            "balance": round(balance, 4),
        })

    return trades


def compute_max_dd(trades):
    if not trades: return 0.0, 0.0
    peak = trades[0]["balance"]
    max_dd = 0.0; max_dd_abs = 0.0
    for t in trades:
        b = t["balance"]
        peak = max(peak, b)
        dd = (peak - b) / peak * 100 if peak else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_abs = peak - b
    return round(max_dd, 2), round(max_dd_abs, 2)


def stats(trades):
    if not trades: return {}
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    net    = sum(t["pnl"] for t in trades)
    wr     = len(wins) / len(trades) * 100
    avg_r  = sum(t["r_multiple"] for t in trades) / len(trades)
    avg_win_r   = sum(t["r_multiple"] for t in wins) / max(len(wins), 1)
    avg_loss_r  = sum(t["r_multiple"] for t in losses) / max(len(losses), 1)
    max_win_r   = max((t["r_multiple"] for t in wins), default=0)
    max_peak_r  = max((t["peak_r"] for t in trades), default=0)
    monsters    = [t for t in wins if t["r_multiple"] >= 3.0]
    pf = sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.001)
    max_dd, max_dd_abs = compute_max_dd(trades)
    final_bal = trades[-1]["balance"]
    return {
        "total": len(trades), "wins": len(wins), "losses": len(losses),
        "net": net, "wr": wr, "avg_r": avg_r,
        "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "max_win_r": max_win_r, "max_peak_r": max_peak_r,
        "monsters": len(monsters), "monster_pnl": sum(t["pnl"] for t in monsters),
        "pf": pf, "max_dd": max_dd, "max_dd_abs": max_dd_abs,
        "final_bal": final_bal,
    }


def print_head_to_head(sa, sb, label_a, label_b):
    def mark(va, vb, higher_better=True):
        if va == vb: return ""
        a_better = (va > vb) if higher_better else (va < vb)
        return "  << WIN" if a_better else "       << WIN"

    w = 14
    print(f"\n{'='*72}")
    print("  HEAD-TO-HEAD")
    print(f"{'='*72}")
    print(f"  {'Metric':<28} {'CURRENT':>{w}}   {'NEW':>{w}}")
    print(f"  {'-'*64}")

    rows = [
        ("Net P&L ($100 acc)",    f"${sa['net']:>+8.2f}",      f"${sb['net']:>+8.2f}",      sa['net'],        sb['net'],        True),
        ("Net P&L ($1,000)",      f"${sa['net']*10:>+9.2f}",   f"${sb['net']*10:>+9.2f}",   sa['net'],        sb['net'],        True),
        ("Final balance",         f"${sa['final_bal']:>8.2f}",  f"${sb['final_bal']:>8.2f}",  sa['final_bal'],  sb['final_bal'],  True),
        ("Win rate",              f"{sa['wr']:>8.1f}%",         f"{sb['wr']:>8.1f}%",         sa['wr'],         sb['wr'],         True),
        ("Avg R per trade",       f"{sa['avg_r']:>+8.3f}R",     f"{sb['avg_r']:>+8.3f}R",     sa['avg_r'],      sb['avg_r'],      True),
        ("Avg winner R",          f"{sa['avg_win_r']:>+8.3f}R", f"{sb['avg_win_r']:>+8.3f}R", sa['avg_win_r'],  sb['avg_win_r'],  True),
        ("Best single trade R",   f"{sa['max_win_r']:>+8.3f}R", f"{sb['max_win_r']:>+8.3f}R", sa['max_win_r'],  sb['max_win_r'],  True),
        ("Highest peak R hit",    f"{sa['max_peak_r']:>8.3f}R", f"{sb['max_peak_r']:>8.3f}R", sa['max_peak_r'], sb['max_peak_r'], True),
        ("Monster trades (3R+)",  f"{sa['monsters']:>8}",       f"{sb['monsters']:>8}",        sa['monsters'],   sb['monsters'],   True),
        ("Monster trade profit",  f"${sa['monster_pnl']:>+8.2f}",f"${sb['monster_pnl']:>+8.2f}",sa['monster_pnl'],sb['monster_pnl'],True),
        ("Profit factor",         f"{sa['pf']:>8.2f}",          f"{sb['pf']:>8.2f}",           sa['pf'],         sb['pf'],         True),
        ("Max Drawdown",          f"{sa['max_dd']:>7.2f}%",     f"{sb['max_dd']:>7.2f}%",      sa['max_dd'],     sb['max_dd'],     False),
        ("Max DD ($)",            f"${sa['max_dd_abs']:>7.2f}", f"${sb['max_dd_abs']:>7.2f}",  sa['max_dd_abs'], sb['max_dd_abs'], False),
    ]

    for name, va_s, vb_s, va, vb, hib in rows:
        best = ""
        if va != vb:
            a_wins = (va > vb) if hib else (va < vb)
            best = "  << CURRENT" if a_wins else "  << NEW"
        print(f"  {name:<28} {va_s:>{w}}   {vb_s:>{w}}{best}")


def print_trade_table(trades, label):
    print(f"\n  {'='*72}")
    print(f"  ALL TRADES -- {label}")
    print(f"  {'='*72}")
    print(f"  {'#':<3} {'Date':<12} {'Dir':<5} {'Risk':<5} {'PeakR':>7} {'FinalR':>7} {'PnL':>8}  {'Hold':>5}  {'Partial'}")
    print("  " + "-"*70)
    for idx, t in enumerate(trades, 1):
        flag = "  *** MONSTER" if t["r_multiple"] >= 3.0 else ""
        partial_str = f"YES @{t['risk_pct']*100:.0f}%" if t["partial"] else "no"
        print(f"  {idx:<3} {str(t['entry_time'])[:10]:<12} {t['type']:<5} "
              f"{t['risk_pct']*100:.0f}%   {t['peak_r']:>6.2f}R  {t['r_multiple']:>+6.3f}R  "
              f"${t['pnl']:>+7.2f}  {t['hold_hours']:>4.1f}h  {partial_str}{flag}")


def main():
    if not connect(): return

    print(f"\nFetching {WINDOWS} windows of data...")
    windows_data = []
    for w in range(WINDOWS):
        sp = w * BARS
        tr = fetch(TREND_TF, BARS, sp // 3)
        er = fetch(ENTRY_TF, BARS, sp)
        if tr is not None and er is not None:
            windows_data.append((tr, er))
    print(f"  Loaded {len(windows_data)}/{WINDOWS} windows.")

    print("\nRunning CURRENT strategy...")
    trades_a = []
    for tr, er in windows_data:
        trades_a.extend(simulate(tr, er, CURRENT))
    trades_a.sort(key=lambda t: t["entry_time"])

    print("Running NEW strategy (progressive Chandelier)...")
    trades_b = []
    for tr, er in windows_data:
        trades_b.extend(simulate(tr, er, NEW))
    trades_b.sort(key=lambda t: t["entry_time"])

    sa = stats(trades_a)
    sb = stats(trades_b)

    print_head_to_head(sa, sb, "CURRENT", "NEW")
    print_trade_table(trades_a, CURRENT["label"])
    print_trade_table(trades_b, NEW["label"])

    # Monster trade deep dive
    print(f"\n{'='*72}")
    print("  MONSTER TRADES DEEP DIVE (3R+ exits)")
    print(f"{'='*72}")
    monsters_a = sorted([t for t in trades_a if t["r_multiple"] >= 3.0], key=lambda t: t["r_multiple"], reverse=True)
    monsters_b = sorted([t for t in trades_b if t["r_multiple"] >= 3.0], key=lambda t: t["r_multiple"], reverse=True)

    print(f"\n  CURRENT -- {len(monsters_a)} monster trades:")
    for t in monsters_a:
        print(f"    {str(t['entry_time'])[:10]}  {t['type']:<5}  peak={t['peak_r']:.2f}R  exit={t['r_multiple']:+.3f}R  ${t['pnl']:+.2f}  {t['hold_hours']:.1f}h")

    print(f"\n  NEW -- {len(monsters_b)} monster trades:")
    for t in monsters_b:
        print(f"    {str(t['entry_time'])[:10]}  {t['type']:<5}  peak={t['peak_r']:.2f}R  exit={t['r_multiple']:+.3f}R  ${t['pnl']:+.2f}  {t['hold_hours']:.1f}h")

    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")
    net_diff = sb['net'] - sa['net']
    dd_diff  = sa['max_dd'] - sb['max_dd']
    mon_diff = sb['monster_pnl'] - sa['monster_pnl']
    print(f"  Net profit change    : ${net_diff:>+.2f}  ({'MORE' if net_diff>0 else 'LESS'} with NEW)")
    print(f"  Drawdown change      : {dd_diff:>+.2f}%  ({'LOWER' if dd_diff>0 else 'HIGHER'} drawdown with NEW)")
    print(f"  Monster profit change: ${mon_diff:>+.2f}  ({'MORE' if mon_diff>0 else 'LESS'} from monster trades with NEW)")
    if net_diff > 0 and dd_diff >= 0:
        print(f"\n  RESULT: NEW is STRICTLY BETTER -- more profit AND less drawdown.")
    elif net_diff > 0 and dd_diff < 0:
        print(f"\n  RESULT: NEW makes more profit but with {abs(dd_diff):.1f}% more drawdown.")
    elif net_diff < 0 and dd_diff > 0:
        print(f"\n  RESULT: NEW is safer (less DD) but gives up ${abs(net_diff):.2f} in profit.")
    else:
        print(f"\n  RESULT: CURRENT wins on both metrics -- keep original settings.")

    mt5.shutdown()


if __name__ == "__main__":
    main()

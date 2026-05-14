"""
Pyramid Backtest -- Step Index
Tests three scenarios on identical data:

  A  CURRENT  : 50% partial @ 2R  |  fixed 3xATR chandelier
  B  NEW      : 25% partial @ 2R  |  progressive 3/2.5/2xATR
  C  PYRAMID  : NEW + add-on position at 2R (risking 50% of original)

Pyramid rules:
  - Triggers at same candle as partial close (price hits +2R)
  - Add-on size sized so it risks PYRAMID_RISK_PCT x original_risk_amount
  - Add-on SL = current chandelier SL of original at time of add
  - Add-on follows same progressive chandelier from that point
  - Both original and add-on close together when chandelier catches price
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

PYRAMID_RISK_PCT = 0.50   # add-on risks 50% of original risk amount

CONFIGS = {
    "A_CURRENT": {
        "label":       "A: CURRENT (50% partial, fixed 3xATR)",
        "partial_pct": 0.50,
        "partial_r":   2.0,
        "tiers":       [(0, 3.0)],
        "pyramid":     False,
    },
    "B_NEW": {
        "label":       "B: NEW    (25% partial, progressive 3/2.5/2xATR)",
        "partial_pct": 0.25,
        "partial_r":   2.0,
        "tiers":       [(0, 3.0), (2.0, 2.5), (4.0, 2.0)],
        "pyramid":     False,
    },
    "C_PYRAMID": {
        "label":       "C: PYRAMID (NEW + add-on at 2R risking 50% extra)",
        "partial_pct": 0.25,
        "partial_r":   2.0,
        "tiers":       [(0, 3.0), (2.0, 2.5), (4.0, 2.0)],
        "pyramid":     True,
    },
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


def chand_mult(r_now, tiers):
    m = tiers[0][1]
    for min_r, mult in tiers:
        if r_now >= min_r:
            m = mult
    return m


def simulate(trend_df, entry_df, cfg):
    trend_df = strategy.calculate_indicators(trend_df)
    entry_df = strategy.calculate_indicators(entry_df)

    PARTIAL_PCT = cfg["partial_pct"]
    PARTIAL_R   = cfg["partial_r"]
    TIERS       = cfg["tiers"]
    USE_PYRAMID = cfg["pyramid"]

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

            # ── Partial close + pyramid add (one-time at PARTIAL_R) ───────────
            if not open_trade["partial_done"]:
                pt = open_trade["entry"] + d * atr * PARTIAL_R
                hit = (t == "BUY" and hi >= pt) or (t == "SELL" and lo <= pt)
                if hit:
                    # Lock partial profit
                    open_trade["locked_pnl"] = (pt - open_trade["entry"]) * d * \
                                               (open_trade["size"] * PARTIAL_PCT)
                    open_trade["size"]       *= (1.0 - PARTIAL_PCT)
                    open_trade["partial_done"] = True

                    # ── Pyramid add-on ────────────────────────────────────────
                    if USE_PYRAMID:
                        # Current chandelier SL becomes pyramid's initial SL
                        r_at_add  = PARTIAL_R
                        cm        = chand_mult(r_at_add, TIERS)
                        pyr_sl    = pt - d * atr * cm
                        sl_dist   = abs(pt - pyr_sl)
                        pyr_risk  = open_trade["risk_amount"] * PYRAMID_RISK_PCT
                        pyr_size  = pyr_risk / sl_dist if sl_dist > 0 else 0
                        open_trade["pyramid"] = {
                            "entry":  pt,
                            "sl":     pyr_sl,
                            "size":   pyr_size,
                            "risk":   pyr_risk,
                            "peak":   pt,
                            "locked": 0.0,
                        }

            # ── Update peak (shared across original + pyramid) ────────────────
            if t == "BUY":
                open_trade["peak"] = max(open_trade["peak"], hi)
            else:
                open_trade["peak"] = min(open_trade["peak"], lo)

            r_now = abs(open_trade["peak"] - open_trade["entry"]) / atr
            cm    = chand_mult(r_now, TIERS)

            # ── Chandelier SL (original) ──────────────────────────────────────
            csl = open_trade["peak"] - d * atr * cm
            if t == "BUY":
                open_trade["sl"] = max(open_trade["sl"], csl)
            else:
                open_trade["sl"] = min(open_trade["sl"], csl)

            # ── Pyramid: update its peak + SL in sync ─────────────────────────
            pyr = open_trade.get("pyramid")
            if pyr is not None:
                if t == "BUY":
                    pyr["peak"] = max(pyr["peak"], hi)
                else:
                    pyr["peak"] = min(pyr["peak"], lo)
                pyr_r  = abs(pyr["peak"] - pyr["entry"]) / atr
                pyr_cm = chand_mult(pyr_r + PARTIAL_R, TIERS)  # offset by 2R since entry is at 2R
                pyr_csl = pyr["peak"] - d * atr * pyr_cm
                if t == "BUY":
                    pyr["sl"] = max(pyr["sl"], pyr_csl)
                else:
                    pyr["sl"] = min(pyr["sl"], pyr_csl)

            # ── Check exit ────────────────────────────────────────────────────
            sl_hit = (t == "BUY" and lo <= open_trade["sl"]) or \
                     (t == "SELL" and hi >= open_trade["sl"])
            timed  = open_trade["age"] >= MAX_HOLD_CANDLES

            if sl_hit or timed:
                ep        = open_trade["sl"] if sl_hit else c["close"]
                raw       = (ep - open_trade["entry"]) * d * open_trade["size"]
                total_pnl = raw + open_trade["locked_pnl"]

                # Add pyramid P&L
                pyr_pnl = 0.0
                if pyr is not None:
                    pyr_ep   = max(ep, open_trade["sl"]) if t == "BUY" else min(ep, open_trade["sl"])
                    pyr_pnl  = (pyr_ep - pyr["entry"]) * d * pyr["size"] + pyr["locked"]
                    total_pnl += pyr_pnl

                if total_pnl > 0: cw += 1; cl = 0
                else: cl += 1; cw = 0

                balance     += total_pnl
                peak_balance = max(peak_balance, balance)
                r_mult  = total_pnl / open_trade["risk_amount"] if open_trade["risk_amount"] else 0
                peak_r  = abs(open_trade["peak"] - open_trade["entry"]) / atr

                trades.append({
                    "entry_time":  open_trade["entry_time"],
                    "exit_time":   c["time"],
                    "type":        t,
                    "entry":       open_trade["entry"],
                    "exit":        ep,
                    "risk_pct":    open_trade["risk_pct"],
                    "risk_amount": open_trade["risk_amount"],
                    "pnl":         round(total_pnl, 4),
                    "pyr_pnl":     round(pyr_pnl, 4),
                    "r_multiple":  round(r_mult, 3),
                    "peak_r":      round(peak_r, 3),
                    "result":      "WIN" if total_pnl > 0 else "LOSS",
                    "partial":     open_trade["partial_done"],
                    "pyramided":   pyr is not None,
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
            "peak": ep2, "partial_done": False,
            "locked_pnl": 0.0, "pyramid": None,
        }

    # Force close
    if open_trade is not None:
        f   = entry_df.iloc[-1]
        raw = (float(f["close"]) - open_trade["entry"]) * \
              (1 if open_trade["type"] == "BUY" else -1) * open_trade["size"]
        pyr_pnl = 0.0
        pyr = open_trade.get("pyramid")
        if pyr is not None:
            pyr_pnl = (float(f["close"]) - pyr["entry"]) * \
                      (1 if open_trade["type"] == "BUY" else -1) * pyr["size"]
        pnl = raw + open_trade["locked_pnl"] + pyr_pnl
        balance += pnl
        r_mult = pnl / open_trade["risk_amount"] if open_trade["risk_amount"] else 0
        trades.append({
            "entry_time": open_trade["entry_time"], "exit_time": f["time"],
            "type": open_trade["type"], "entry": open_trade["entry"],
            "exit": float(f["close"]),
            "risk_pct": open_trade["risk_pct"], "risk_amount": open_trade["risk_amount"],
            "pnl": round(pnl, 4), "pyr_pnl": round(pyr_pnl, 4),
            "r_multiple": round(r_mult, 3),
            "peak_r": round(abs(open_trade["peak"] - open_trade["entry"]) / open_trade["atr"], 3),
            "result": "WIN" if pnl > 0 else "LOSS",
            "partial": open_trade["partial_done"],
            "pyramided": pyr is not None,
            "hold_hours": round(open_trade["age"] * 5 / 60, 1),
            "balance": round(balance, 4),
        })

    return trades


def compute_dd(trades):
    if not trades: return 0.0, 0.0
    peak = START_BALANCE; max_dd = 0.0; max_dd_abs = 0.0
    for t in trades:
        b = t["balance"]
        peak = max(peak, b)
        dd = (peak - b) / peak * 100 if peak else 0
        if dd > max_dd:
            max_dd = dd; max_dd_abs = peak - b
    return round(max_dd, 2), round(max_dd_abs, 2)


def get_stats(trades):
    if not trades: return {}
    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    net    = sum(t["pnl"] for t in trades)
    wr     = len(wins) / len(trades) * 100
    avg_r  = sum(t["r_multiple"] for t in trades) / len(trades)
    avg_wr = sum(t["r_multiple"] for t in wins) / max(len(wins), 1)
    best_r = max((t["r_multiple"] for t in wins), default=0)
    monsters = [t for t in trades if t["r_multiple"] >= 3.0]
    monster_pnl = sum(t["pnl"] for t in monsters)
    pf = sum(t["pnl"] for t in wins) / max(abs(sum(t["pnl"] for t in losses)), 0.001)
    max_dd, max_dd_abs = compute_dd(trades)
    pyr_trades = [t for t in trades if t.get("pyramided")]
    pyr_pnl    = sum(t["pyr_pnl"] for t in pyr_trades)
    streak = 0; best_streak = 0
    for t in trades:
        if t["result"] == "LOSS": streak += 1; best_streak = max(best_streak, streak)
        else: streak = 0
    days = (trades[-1]["exit_time"] - trades[0]["entry_time"]).days
    ann  = (net / START_BALANCE) / max(days / 365, 0.001) * 100
    calmar = ann / max(max_dd, 0.001)
    return {
        "n": len(trades), "wins": len(wins), "losses": len(losses),
        "net": net, "wr": wr, "avg_r": avg_r, "avg_wr": avg_wr,
        "best_r": best_r, "monsters": len(monsters), "monster_pnl": monster_pnl,
        "pf": pf, "max_dd": max_dd, "max_dd_abs": max_dd_abs,
        "streak": best_streak, "ann": ann, "calmar": calmar,
        "final": trades[-1]["balance"],
        "pyr_count": len(pyr_trades), "pyr_pnl": pyr_pnl,
    }


def print_comparison(sa, sb, sc):
    W = 13
    print(f"\n{'='*72}")
    print("  3-WAY COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Metric':<26} {'A:CURRENT':>{W}} {'B:NEW':>{W}} {'C:PYRAMID':>{W}}")
    print(f"  {'-'*68}")

    def best3(va, vb, vc, hib=True):
        vals = [va, vb, vc]
        best = max(vals) if hib else min(vals)
        return ["  ***" if v == best else "     " for v in vals]

    rows = [
        ("Net P&L ($100)",       f"${sa['net']:>+7.2f}",    f"${sb['net']:>+7.2f}",    f"${sc['net']:>+7.2f}",    sa['net'],    sb['net'],    sc['net'],    True),
        ("Net P&L ($1,000)",     f"${sa['net']*10:>+8.2f}", f"${sb['net']*10:>+8.2f}", f"${sc['net']*10:>+8.2f}", sa['net'],    sb['net'],    sc['net'],    True),
        ("Final balance",        f"${sa['final']:>8.2f}",   f"${sb['final']:>8.2f}",   f"${sc['final']:>8.2f}",   sa['final'],  sb['final'],  sc['final'],  True),
        ("Win rate",             f"{sa['wr']:>8.1f}%",      f"{sb['wr']:>8.1f}%",      f"{sc['wr']:>8.1f}%",      sa['wr'],     sb['wr'],     sc['wr'],     True),
        ("Avg R/trade",          f"{sa['avg_r']:>+7.3f}R",  f"{sb['avg_r']:>+7.3f}R",  f"{sc['avg_r']:>+7.3f}R",  sa['avg_r'],  sb['avg_r'],  sc['avg_r'],  True),
        ("Avg winner R",         f"{sa['avg_wr']:>+7.3f}R", f"{sb['avg_wr']:>+7.3f}R", f"{sc['avg_wr']:>+7.3f}R", sa['avg_wr'], sb['avg_wr'], sc['avg_wr'], True),
        ("Best single R",        f"{sa['best_r']:>+7.3f}R", f"{sb['best_r']:>+7.3f}R", f"{sc['best_r']:>+7.3f}R", sa['best_r'], sb['best_r'], sc['best_r'], True),
        ("Monster trades (3R+)", f"{sa['monsters']:>9}",    f"{sb['monsters']:>9}",    f"{sc['monsters']:>9}",    sa['monsters'],sb['monsters'],sc['monsters'],True),
        ("Monster total $",      f"${sa['monster_pnl']:>+7.2f}", f"${sb['monster_pnl']:>+7.2f}", f"${sc['monster_pnl']:>+7.2f}", sa['monster_pnl'],sb['monster_pnl'],sc['monster_pnl'],True),
        ("Profit factor",        f"{sa['pf']:>9.2f}",       f"{sb['pf']:>9.2f}",       f"{sc['pf']:>9.2f}",       sa['pf'],     sb['pf'],     sc['pf'],     True),
        ("Max Drawdown %",       f"{sa['max_dd']:>8.2f}%",  f"{sb['max_dd']:>8.2f}%",  f"{sc['max_dd']:>8.2f}%",  sa['max_dd'], sb['max_dd'], sc['max_dd'], False),
        ("Max DD ($)",           f"${sa['max_dd_abs']:>7.2f}", f"${sb['max_dd_abs']:>7.2f}", f"${sc['max_dd_abs']:>7.2f}", sa['max_dd_abs'],sb['max_dd_abs'],sc['max_dd_abs'],False),
        ("Worst L-streak",       f"{sa['streak']:>9}",      f"{sb['streak']:>9}",      f"{sc['streak']:>9}",      sa['streak'], sb['streak'], sc['streak'], False),
        ("Ann. return",          f"{sa['ann']:>7.1f}%",     f"{sb['ann']:>7.1f}%",     f"{sc['ann']:>7.1f}%",     sa['ann'],    sb['ann'],    sc['ann'],    True),
        ("Calmar ratio",         f"{sa['calmar']:>9.2f}",   f"{sb['calmar']:>9.2f}",   f"{sc['calmar']:>9.2f}",   sa['calmar'], sb['calmar'], sc['calmar'], True),
    ]

    for name, va_s, vb_s, vc_s, va, vb, vc, hib in rows:
        marks = best3(va, vb, vc, hib)
        print(f"  {name:<26} {va_s:>{W}}{marks[0]} {vb_s:>{W}}{marks[1]} {vc_s:>{W}}{marks[2]}")

    print(f"\n  Pyramid stats (C only):")
    print(f"    Trades that got a pyramid add : {sc['pyr_count']}")
    print(f"    Total profit FROM pyramid adds: ${sc['pyr_pnl']:+.2f}")
    print(f"    Profit per pyramid add        : ${sc['pyr_pnl']/max(sc['pyr_count'],1):+.2f}")


def print_all_trades(trades, label):
    print(f"\n  {'='*74}")
    print(f"  {label}")
    print(f"  {'='*74}")
    print(f"  {'#':<3} {'Date':<12} {'Dir':<5} {'Risk':<5} {'PeakR':>6} {'R':>7} {'PnL':>8} {'PyrPnL':>8} {'Hold':>5}  Notes")
    print("  " + "-"*74)
    for idx, t in enumerate(trades, 1):
        notes = []
        if t.get("pyramided"): notes.append("PYRAMID")
        if t["r_multiple"] >= 3.0: notes.append("*** MONSTER")
        if t["result"] == "LOSS": notes.append("LOSS")
        print(f"  {idx:<3} {str(t['entry_time'])[:10]:<12} {t['type']:<5} "
              f"{t['risk_pct']*100:.0f}%   {t['peak_r']:>5.2f}R {t['r_multiple']:>+6.3f}R "
              f"${t['pnl']:>+7.2f} ${t['pyr_pnl']:>+7.2f}  {t['hold_hours']:>4.1f}h  {'  '.join(notes)}")


def main():
    if not connect(): return

    print(f"\nFetching {WINDOWS} windows...")
    windows_data = []
    for w in range(WINDOWS):
        sp = w * BARS
        tr = fetch(TREND_TF, BARS, sp // 3)
        er = fetch(ENTRY_TF, BARS, sp)
        if tr is not None and er is not None:
            windows_data.append((tr, er))
    print(f"  Loaded {len(windows_data)}/{WINDOWS} windows.")

    results = {}
    for key, cfg in CONFIGS.items():
        print(f"Running {cfg['label']}...")
        all_trades = []
        for tr, er in windows_data:
            all_trades.extend(simulate(tr, er, cfg))
        all_trades.sort(key=lambda t: t["entry_time"])
        results[key] = all_trades

    sa = get_stats(results["A_CURRENT"])
    sb = get_stats(results["B_NEW"])
    sc = get_stats(results["C_PYRAMID"])

    print_comparison(sa, sb, sc)

    for key, cfg in CONFIGS.items():
        print_all_trades(results[key], cfg["label"])

    # Verdict
    print(f"\n{'='*72}")
    print("  VERDICT")
    print(f"{'='*72}")
    best_net = max(sa['net'], sb['net'], sc['net'])
    best_cal = max(sa['calmar'], sb['calmar'], sc['calmar'])
    print(f"  Highest profit  : {'C:PYRAMID' if sc['net']==best_net else ('B:NEW' if sb['net']==best_net else 'A:CURRENT')}  (${best_net:.2f})")
    print(f"  Best Calmar     : {'C:PYRAMID' if sc['calmar']==best_cal else ('B:NEW' if sb['calmar']==best_cal else 'A:CURRENT')}  ({best_cal:.2f})")
    print(f"  Pyramid adds    : ${sc['pyr_pnl']:+.2f} extra profit from {sc['pyr_count']} add-ons")
    print(f"  DD cost         : {sc['max_dd'] - sa['max_dd']:+.2f}% vs CURRENT  |  {sc['max_dd'] - sb['max_dd']:+.2f}% vs NEW")

    mt5.shutdown()


if __name__ == "__main__":
    main()

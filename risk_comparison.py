"""
risk_comparison.py
Monthly equity breakdown: 1% vs 2% risk, best config (Chandelier x1.6, hold=6).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from commodity_backtest import load_raw, build_merged, CHAND_TIERS

m5 = load_raw('XAUUSD', 'M5')
h1 = load_raw('XAUUSD', 'H1')
df = build_merged(m5, h1, 21)

op  = df['open'].values;  hi = df['high'].values
lo  = df['low'].values;   cl = df['close'].values
atr = df['atr'].values;   h1c = df['h1_close'].values
h1e = df['h1_ema21'].values
hrs = df['time'].dt.hour.values
agp = df['after_gap'].values.astype(bool)
dts = df['time'].dt.date.astype(str).values
mth = df['time'].dt.tz_localize(None).dt.to_period('M').astype(str).values
N   = len(df)

ACTIVE   = frozenset(range(7, 21))
cooldown = 2
max_day  = 12

m5_ema  = pd.Series(cl).ewm(span=50, adjust=False).mean().values
h1_bull = h1c > h1e
h1_bear = h1c < h1e
bull_c  = cl > op
bear_c  = cl < op
near    = np.abs(cl - m5_ema) <= 0.50 * atr
base    = (~agp) & (atr > 0) & ~np.isnan(m5_ema) & np.isin(hrs, list(ACTIVE))
buy_m   = base & h1_bull & near & bull_c
sell_m  = base & h1_bear & near & bear_c
all_sigs = sorted(
    [(i, 'BUY')  for i in np.where(buy_m)[0]] +
    [(i, 'SELL') for i in np.where(sell_m)[0]]
)

CHAND_MULT = 1.6
HOLD       = 6


def sim_trade(entry, av, fwd_h, fwd_l, fwd_c, hold, is_buy, chand_mult=1.6):
    """Simplified sim: no BE/pyr, just Chandelier x chand_mult + 50% partial at 2R."""
    if is_buy:
        sl = entry - av
        size = 1.0; locked = 0.0; partial_done = False; peak = entry
        n = min(hold, len(fwd_h))
        for i in range(n):
            lo_b, hi_b = fwd_l[i], fwd_h[i]
            if lo_b <= sl:
                return round(size * (-1.0) + locked, 3), 'SL'
            if not partial_done and hi_b >= entry + av * 2.0:
                locked += size * 0.50 * 2.0
                size   *= 0.50
                partial_done = True
            peak = max(peak, hi_b)
            pr   = (peak - entry) / av
            cm   = CHAND_TIERS[0][1]
            for mr, tm in CHAND_TIERS:
                if pr >= mr: cm = tm
            cm  *= chand_mult
            csl  = peak - av * cm
            if lo_b <= csl:
                return round(size * ((csl - entry) / av) + locked, 3), 'CHAND'
        last = fwd_c[n-1] if n > 0 else entry
        return round(size * ((last - entry) / av) + locked, 3), 'TIME'
    else:
        sl = entry + av
        size = 1.0; locked = 0.0; partial_done = False; trough = entry
        n = min(hold, len(fwd_h))
        for i in range(n):
            lo_b, hi_b = fwd_l[i], fwd_h[i]
            if hi_b >= sl:
                return round(size * (-1.0) + locked, 3), 'SL'
            if not partial_done and lo_b <= entry - av * 2.0:
                locked += size * 0.50 * 2.0
                size   *= 0.50
                partial_done = True
            trough = min(trough, lo_b)
            tr_r   = (entry - trough) / av
            cm     = CHAND_TIERS[0][1]
            for mr, tm in CHAND_TIERS:
                if tr_r >= mr: cm = tm
            cm  *= chand_mult
            csl  = trough + av * cm
            if hi_b >= csl:
                return round(size * ((entry - csl) / av) + locked, 3), 'CHAND'
        last = fwd_c[n-1] if n > 0 else entry
        return round(size * ((entry - last) / av) + locked, 3), 'TIME'


# --- collect all trades ---
traded = set(); tpd = {}; lsi = -999; trades = []
for idx, sig in all_sigs:
    if idx + HOLD + 1 >= N: break
    if idx in traded: continue
    if (idx - lsi) <= cooldown: lsi = idx; continue
    lsi = idx
    date = dts[idx]
    if tpd.get(date, 0) >= max_day: continue
    entry = float(op[idx + 1])
    av    = float(atr[idx])
    if av <= 0: continue
    fe = min(idx + 1 + HOLD, N)
    fh = hi[idx+1:fe]; fl = lo[idx+1:fe]; fc = cl[idx+1:fe]
    if len(fh) < 4: continue
    r, reason = sim_trade(entry, av, fh, fl, fc, HOLD, sig == 'BUY', CHAND_MULT)
    tpd[date] = tpd.get(date, 0) + 1
    trades.append({'month': mth[idx], 'r': r, 'exit': reason})
    for k in range(idx, fe): traded.add(k)

bt = pd.DataFrame(trades)

# --- monthly compounding for a given risk % ---
def monthly_table(bt, risk_pct, start=50.0):
    eq = start
    rows = []
    for month, grp in bt.groupby('month', sort=True):
        eq_start = eq
        dollar_risk = eq * risk_pct
        r_sum = grp['r'].sum()
        pnl   = r_sum * dollar_risk
        eq   += pnl
        w     = (grp['r'] > 0).sum()
        n     = len(grp)
        rows.append({
            'Month': month, 'Trades': n,
            'WR': f'{w/n*100:.0f}%',
            'R sum': f'{r_sum:+.1f}R',
            'Risk/trade': f'${dollar_risk:,.2f}',
            'P&L': f'${pnl:+,.0f}',
            'Equity': f'${eq:,.0f}',
        })
    return rows, eq

rows1, end1 = monthly_table(bt, 0.01)
rows2, end2 = monthly_table(bt, 0.02)

n  = len(bt); w = (bt['r'] > 0).sum()
gw = bt[bt['r'] > 0]['r'].sum()
gl = abs(bt[bt['r'] <= 0]['r'].sum())
pf = gw / gl if gl > 0 else float('inf')

print()
print('=' * 110)
print(f'  XAUUSD M5 EMA Pullback — Chandelier x{CHAND_MULT}  |  N={n}  WR={w/n*100:.1f}%  PF={pf:.3f}')
print('=' * 110)

# 1% table
print()
print(f'  ── 1% RISK PER TRADE  ($50 start) ──')
print(f'  {"Month":<10} {"Trades":>7} {"WR":>5} {"R sum":>8} {"Risk/trade":>12} {"P&L":>10} {"Equity":>12}')
print(f'  {"-"*10} {"-"*7} {"-"*5} {"-"*8} {"-"*12} {"-"*10} {"-"*12}')
for r in rows1:
    print(f'  {r["Month"]:<10} {r["Trades"]:>7} {r["WR"]:>5} {r["R sum"]:>8} {r["Risk/trade"]:>12} {r["P&L"]:>10} {r["Equity"]:>12}')
print(f'  {"FINAL":<10} {"":>7} {"":>5} {"":>8} {"":>12} {"":>10} ${end1:>11,.0f}')

# 2% table
print()
print(f'  ── 2% RISK PER TRADE  ($50 start) ──')
print(f'  {"Month":<10} {"Trades":>7} {"WR":>5} {"R sum":>8} {"Risk/trade":>12} {"P&L":>10} {"Equity":>12}')
print(f'  {"-"*10} {"-"*7} {"-"*5} {"-"*8} {"-"*12} {"-"*10} {"-"*12}')
for r in rows2:
    print(f'  {r["Month"]:<10} {r["Trades"]:>7} {r["WR"]:>5} {r["R sum"]:>8} {r["Risk/trade"]:>12} {r["P&L"]:>10} {r["Equity"]:>12}')
print(f'  {"FINAL":<10} {"":>7} {"":>5} {"":>8} {"":>12} {"":>10} ${end2:>11,.0f}')

print()
print(f'  SUMMARY:  1% risk → ${end1:,.0f}   |   2% risk → ${end2:,.0f}   (starting $50)')
print('=' * 110)

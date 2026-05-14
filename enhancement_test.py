"""
enhancement_test.py
Tests BE stop, pyramiding, wider Chandelier on the best XAUUSD pullback config.
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

ACTIVE = frozenset(range(7, 21))
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


def sim_buy(entry, av, fwd_h, fwd_l, fwd_c, hold,
            use_be=False, use_pyr=False, chand_mult=1.0):
    sl = entry - av
    size = 1.0; locked = 0.0; partial_done = False; be_done = False
    pyr_done = False; pyr_size = 0.0; pyr_entry = 0.0; pyr_locked = 0.0; pyr_partial = False
    peak = entry
    n = min(hold, len(fwd_h))

    for i in range(n):
        lo_b, hi_b = fwd_l[i], fwd_h[i]

        # SL hit
        if lo_b <= sl:
            pyr_r = 0.0
            if pyr_done and pyr_size > 0:
                pyr_r = pyr_size * ((sl - pyr_entry) / av) + pyr_locked
            return round(size * (-1.0) + locked + pyr_r, 3), 'SL'

        # Pyramid SL (always at original entry = breakeven for original)
        if pyr_done and pyr_size > 0 and lo_b <= entry:
            pyr_locked += pyr_size * ((entry - pyr_entry) / av)
            pyr_size = 0.0

        # Breakeven stop on original
        if use_be and not be_done and hi_b >= entry + av:
            sl = entry
            be_done = True

        # Add pyramid at +1R
        if use_pyr and not pyr_done and hi_b >= entry + av:
            pyr_done  = True
            pyr_size  = 0.50
            pyr_entry = hi_b

        # Partial on original at 2R
        if not partial_done and hi_b >= entry + av * 2.0:
            locked += size * 0.50 * 2.0
            size   *= 0.50
            partial_done = True

        # Partial on pyramid at 2R (from pyramid entry)
        if pyr_done and not pyr_partial and pyr_size > 0 and hi_b >= pyr_entry + av * 2.0:
            pyr_locked += pyr_size * 0.50 * 2.0
            pyr_size   *= 0.50
            pyr_partial = True

        peak = max(peak, hi_b)
        pr   = (peak - entry) / av
        cm   = CHAND_TIERS[0][1]
        for mr, tm in CHAND_TIERS:
            if pr >= mr: cm = tm
        cm *= chand_mult
        csl = peak - av * cm

        if lo_b <= csl:
            main_r = size * ((csl - entry) / av) + locked
            pyr_r  = pyr_size * ((csl - pyr_entry) / av) + pyr_locked if pyr_done else 0.0
            return round(main_r + pyr_r, 3), 'CHANDELIER'

    last   = fwd_c[n-1] if n > 0 else entry
    pyr_r  = pyr_size * ((last - pyr_entry) / av) + pyr_locked if pyr_done else 0.0
    return round(size * ((last - entry) / av) + locked + pyr_r, 3), 'TIME'


def sim_sell(entry, av, fwd_h, fwd_l, fwd_c, hold,
             use_be=False, use_pyr=False, chand_mult=1.0):
    sl = entry + av
    size = 1.0; locked = 0.0; partial_done = False; be_done = False
    pyr_done = False; pyr_size = 0.0; pyr_entry = 0.0; pyr_locked = 0.0; pyr_partial = False
    trough = entry
    n = min(hold, len(fwd_h))

    for i in range(n):
        lo_b, hi_b = fwd_l[i], fwd_h[i]

        if hi_b >= sl:
            pyr_r = pyr_size * ((pyr_entry - sl) / av) + pyr_locked if (pyr_done and pyr_size > 0) else 0.0
            return round(size * (-1.0) + locked + pyr_r, 3), 'SL'

        if pyr_done and pyr_size > 0 and hi_b >= entry:
            pyr_locked += pyr_size * ((pyr_entry - entry) / av)
            pyr_size = 0.0

        if use_be and not be_done and lo_b <= entry - av:
            sl = entry
            be_done = True

        if use_pyr and not pyr_done and lo_b <= entry - av:
            pyr_done  = True
            pyr_size  = 0.50
            pyr_entry = lo_b

        if not partial_done and lo_b <= entry - av * 2.0:
            locked += size * 0.50 * 2.0
            size   *= 0.50
            partial_done = True

        if pyr_done and not pyr_partial and pyr_size > 0 and lo_b <= pyr_entry - av * 2.0:
            pyr_locked += pyr_size * 0.50 * 2.0
            pyr_size   *= 0.50
            pyr_partial = True

        trough = min(trough, lo_b)
        tr_r   = (entry - trough) / av
        cm     = CHAND_TIERS[0][1]
        for mr, tm in CHAND_TIERS:
            if tr_r >= mr: cm = tm
        cm *= chand_mult
        csl = trough + av * cm

        if hi_b >= csl:
            main_r = size * ((entry - csl) / av) + locked
            pyr_r  = pyr_size * ((pyr_entry - csl) / av) + pyr_locked if pyr_done else 0.0
            return round(main_r + pyr_r, 3), 'CHANDELIER'

    last  = fwd_c[n-1] if n > 0 else entry
    pyr_r = pyr_size * ((pyr_entry - last) / av) + pyr_locked if pyr_done else 0.0
    return round(size * ((entry - last) / av) + locked + pyr_r, 3), 'TIME'


def run_mode(label, hold, use_be=False, use_pyr=False, chand_mult=1.0):
    traded = set(); tpd = {}; lsi = -999; trades = []
    for idx, sig in all_sigs:
        if idx + hold + 1 >= N: break
        if idx in traded: continue
        if (idx - lsi) <= cooldown: lsi = idx; continue
        lsi = idx
        date = dts[idx]
        if tpd.get(date, 0) >= max_day: continue
        entry = float(op[idx + 1])
        av    = float(atr[idx])
        if av <= 0: continue
        fe    = min(idx + 1 + hold, N)
        fh    = hi[idx+1:fe]; fl = lo[idx+1:fe]; fc = cl[idx+1:fe]
        if len(fh) < 4: continue

        if sig == 'BUY':
            r, reason = sim_buy(entry, av, fh, fl, fc, hold, use_be, use_pyr, chand_mult)
        else:
            r, reason = sim_sell(entry, av, fh, fl, fc, hold, use_be, use_pyr, chand_mult)

        tpd[date] = tpd.get(date, 0) + 1
        trades.append({'month': mth[idx], 'r': r, 'exit': reason})
        for k in range(idx, fe): traded.add(k)

    bt = pd.DataFrame(trades)
    if bt.empty:
        print(f'  {label}: no trades')
        return

    n  = len(bt); w = (bt['r'] > 0).sum()
    gw = bt[bt['r'] > 0]['r'].sum()
    gl = abs(bt[bt['r'] <= 0]['r'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    months  = bt['month'].nunique()
    oos     = bt.iloc[int(n * 0.75):]
    ow = oos[oos['r'] > 0]['r'].sum()
    ol = abs(oos[oos['r'] <= 0]['r'].sum())
    oos_pf  = ow / ol if ol > 0 else float('inf')

    # Compound from $50 at 1% risk
    eq = 50.0
    for r in bt['r']:
        eq += r * (eq * 0.01)

    print(f'  {label:<40} N={n:>5}  WR={w/n*100:>4.1f}%  PF={pf:>5.3f}  OOS={oos_pf:>5.3f}'
          f'  AvgR={bt["r"].mean():>+.3f}  {n/months:>5.0f}/mo  $50->>${eq:>10,.0f}')
    exits = bt['exit'].value_counts()
    for ex, cnt in exits.items():
        ar = bt[bt['exit'] == ex]['r'].mean()
        print(f'      {ex:<14}{cnt:>5} ({cnt/n*100:>4.1f}%)  avg R {ar:>+.3f}')
    print()


print()
print('=' * 105)
print('  XAUUSD PULLBACK — Enhancement Comparison  (1% compound from $50, zone=0.50 m5ema=50 h1ema=21)')
print('=' * 105)
print()
run_mode('A) Baseline                          (hold=6)',  6)
run_mode('B) BE stop at 1R                     (hold=6)',  6,  use_be=True)
run_mode('C) Pyramid at 1R                     (hold=6)',  6,  use_pyr=True)
run_mode('D) BE + Pyramid                      (hold=6)',  6,  use_be=True,  use_pyr=True)
run_mode('E) Wider Chandelier x1.4             (hold=6)',  6,  chand_mult=1.4)
run_mode('F) BE + Pyramid                      (hold=12)', 12, use_be=True,  use_pyr=True)
run_mode('G) Pyramid only                      (hold=12)', 12, use_pyr=True)
run_mode('H) Wide Chandelier + Pyramid         (hold=12)', 12, use_pyr=True,  chand_mult=1.4)
run_mode('I) BE + Pyramid + Wide Chandelier    (hold=12)', 12, use_be=True,  use_pyr=True, chand_mult=1.4)
run_mode('J) BE + Pyramid + Wide Chandelier    (hold=24)', 24, use_be=True,  use_pyr=True, chand_mult=1.4)
run_mode('K) Pyramid + Wide Chandelier         (hold=24)', 24, use_pyr=True,  chand_mult=1.4)
print('=' * 105)

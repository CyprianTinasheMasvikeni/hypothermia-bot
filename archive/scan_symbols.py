"""
Scan MT5 for all available Step Index variants and show their properties.
"""
import MetaTrader5 as mt5
import pandas as pd

def main():
    if not mt5.initialize():
        print("MT5 init failed:", mt5.last_error())
        return

    all_symbols = mt5.symbols_get()
    step_symbols = [s for s in all_symbols if "step" in s.name.lower()]

    if not step_symbols:
        print("No Step Index variants found.")
        mt5.shutdown()
        return

    print(f"\nFound {len(step_symbols)} Step Index variant(s):\n")
    print(f"  {'Symbol':<30} {'Digits':>6} {'Spread':>8} {'Min Lot':>9} {'Lot Step':>9} {'Min Vol':>9}")
    print("  " + "-"*75)

    for s in step_symbols:
        info = mt5.symbol_info(s.name)
        if info is None:
            continue
        print(f"  {s.name:<30} {info.digits:>6} {info.spread:>8} {info.volume_min:>9.4f} {info.volume_step:>9.4f} {info.volume_min:>9.4f}")

    print()
    print("Checking recent data availability:")
    print(f"  {'Symbol':<30} {'M5 bars':>10} {'M15 bars':>10} {'Last candle':<25}")
    print("  " + "-"*75)

    for s in step_symbols:
        m5  = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M5,  0, 100)
        m15 = mt5.copy_rates_from_pos(s.name, mt5.TIMEFRAME_M15, 0, 100)
        m5_count  = len(m5)  if m5  is not None else 0
        m15_count = len(m15) if m15 is not None else 0
        if m5 is not None and len(m5) > 0:
            import datetime
            last = datetime.datetime.utcfromtimestamp(m5[-1]["time"]).strftime("%Y-%m-%d %H:%M UTC")
        else:
            last = "NO DATA"
        print(f"  {s.name:<30} {m5_count:>10} {m15_count:>10} {last:<25}")

    mt5.shutdown()

if __name__ == "__main__":
    main()

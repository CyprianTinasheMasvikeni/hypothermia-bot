import MetaTrader5 as mt5
import pandas as pd
from config import PAIR, TIMEFRAME
from strategy_sniper_v4 import calculate_indicators, generate_signal

ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE_PERCENT = 1.0
MAX_TRADES_PER_DAY = 3
RR_MULTIPLIER = 2.0  # TP = 2x ATR

def connect_mt5():
    if not mt5.initialize():
        print("❌ MT5 initialization failed:", mt5.last_error())
        return False
    print("✅ Connected to MT5")
    return True

def fetch_data(symbol, timeframe, bars=15000):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }
    if timeframe not in tf_map:
        print("❌ Invalid timeframe")
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    if rates is None or len(rates) == 0:
        print(f"⚠️ No data returned for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def simulate_sniper(df):
    trades = []
    df = calculate_indicators(df)
    position = None
    balance = ACCOUNT_BALANCE
    daily_trades = {}

    for i in range(60, len(df)):
        slice_df = df.iloc[:i+1].copy()
        candle = df.iloc[i]
        date = candle['time'].date()
        daily_trades.setdefault(date, 0)

        if daily_trades[date] >= MAX_TRADES_PER_DAY:
            continue

        signal = generate_signal(slice_df)
        atr = slice_df['atr'].iloc[-1]
        if pd.isna(atr) or atr == 0:
            continue

        risk_amount = (RISK_PER_TRADE_PERCENT / 100) * balance
        sl_points = atr
        tp_points = atr * RR_MULTIPLIER
        lot_size = risk_amount / sl_points if sl_points != 0 else 0

        if position is None and signal in ["BUY", "SELL"]:
            entry = candle['close']
            sl = entry - sl_points if signal == "BUY" else entry + sl_points
            tp = entry + tp_points if signal == "BUY" else entry - tp_points
            partial_tp = entry + atr if signal == "BUY" else entry - atr
            position = {
                "type": signal,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "partial_tp": partial_tp,
                "entry_time": candle['time'],
                "entry_index": i,
                "lot_size": lot_size,
                "risk_amount": risk_amount,
                "breakeven_set": False,
                "trail_triggered": False,
                "partial_taken": False,
                "age": 0
            }

        elif position:
            candle_high = candle['high']
            candle_low = candle['low']
            position['age'] += 1
            direction = 1 if position['type'] == 'BUY' else -1

            # Partial TP at +1R
            if not position['partial_taken']:
                if (direction == 1 and candle_high >= position['partial_tp']) or \
                   (direction == -1 and candle_low <= position['partial_tp']):
                    position['partial_taken'] = True
                    locked_profit = atr * position['lot_size'] * 0.5
                    balance += locked_profit
                    position['risk_amount'] *= 0.5  # halve risk
                    position['lot_size'] *= 0.5

            # Breakeven SL after +1.5R move
            if not position['breakeven_set']:
                if (direction == 1 and candle_high >= position['entry'] + 1.5 * atr) or \
                   (direction == -1 and candle_low <= position['entry'] - 1.5 * atr):
                    position['sl'] = position['entry']
                    position['breakeven_set'] = True

            # Trail SL after +2R move
            if position['breakeven_set'] and not position['trail_triggered']:
                if direction == 1 and candle_high >= position['entry'] + 2 * atr:
                    position['sl'] = max(position['sl'], candle_low - 0.5 * atr)
                    position['trail_triggered'] = True
                elif direction == -1 and candle_low <= position['entry'] - 2 * atr:
                    position['sl'] = min(position['sl'], candle_high + 0.5 * atr)
                    position['trail_triggered'] = True

            # Exit on SL
            if (position['type'] == 'BUY' and candle_low <= position['sl']) or \
               (position['type'] == 'SELL' and candle_high >= position['sl']):
                position['exit'] = position['sl']
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                pnl = (position['exit'] - position['entry']) * direction
                position['result'] = "LOSS" if pnl < 0 else "BE"
                position['pnl'] = pnl * position['lot_size']
                balance += position['pnl']
                trades.append(position)
                position = None
                daily_trades[date] += 1
                continue

            # Exit on TP
            if (position['type'] == 'BUY' and candle_high >= position['tp']) or \
               (position['type'] == 'SELL' and candle_low <= position['tp']):
                position['exit'] = position['tp']
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                pnl = (position['exit'] - position['entry']) * direction
                position['result'] = "WIN"
                position['pnl'] = pnl * position['lot_size']
                balance += position['pnl']
                trades.append(position)
                position = None
                daily_trades[date] += 1
                continue

            # Exit on age timeout
            if position['age'] >= 20:
                position['exit'] = candle['close']
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                pnl = (position['exit'] - position['entry']) * direction
                position['result'] = "EXIT"
                position['pnl'] = pnl * position['lot_size']
                balance += position['pnl']
                trades.append(position)
                position = None
                daily_trades[date] += 1

    return trades

if __name__ == "__main__":
    if connect_mt5():
        df = fetch_data(PAIR, TIMEFRAME, bars=15000)
        if df is not None:
            trades = simulate_sniper(df)

            wins = sum(1 for t in trades if t['pnl'] > 0)
            losses = sum(1 for t in trades if t['pnl'] < 0)
            exits = sum(1 for t in trades if t['result'] == 'EXIT')
            be = sum(1 for t in trades if t['result'] == 'BE')
            total = len(trades)
            net_profit = sum(t['pnl'] for t in trades)
            win_rate = (wins / total) * 100 if total > 0 else 0

            print(f"\n📊 Total Trades: {total}")
            print(f"✅ Wins: {wins}")
            print(f"❌ Losses: {losses}")
            print(f"➖ Breakeven: {be}")
            print(f"🚪 Exits: {exits}")
            print(f"💰 Net Profit: ${net_profit:.2f}")
            print(f"🏆 Win Rate: {win_rate:.2f}%")

            df_out = pd.DataFrame(trades)
            df_out.to_csv("strategy_sniper_v4_backtest.csv", index=False)
            print("📦 Trades exported to strategy_sniper_v4_backtest.csv")

        mt5.shutdown()

import MetaTrader5 as mt5
import pandas as pd
from config import PAIR, TIMEFRAME
from strategy_sniper_v4_1 import calculate_indicators, generate_signal

# Settings
ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE_PERCENT = 1.0
MAX_TRADES_PER_DAY = 3
TP1_MULTIPLIER = 1.0
TP2_MULTIPLIER = 1.5
TRAIL_AFTER_MULTIPLIER = 1.5
MAX_CANDLE_AGE = 15

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
        tp1_points = atr * TP1_MULTIPLIER
        tp2_points = atr * TP2_MULTIPLIER
        lot_size = risk_amount / sl_points if sl_points != 0 else 0

        if position is None and signal in ["BUY", "SELL"]:
            entry = candle['close']
            sl = entry - sl_points if signal == "BUY" else entry + sl_points
            tp1 = entry + tp1_points if signal == "BUY" else entry - tp1_points
            tp2 = entry + tp2_points if signal == "BUY" else entry - tp2_points
            position = {
                "type": signal,
                "entry": entry,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "entry_time": candle['time'],
                "entry_index": i,
                "lot_size": lot_size,
                "risk_amount": risk_amount,
                "breakeven_set": False,
                "trail_triggered": False,
                "tp1_hit": False,
                "age": 0
            }

        elif position:
            candle_high = candle['high']
            candle_low = candle['low']
            position['age'] += 1
            direction = 1 if position['type'] == 'BUY' else -1

            # TP1 Hit
            if not position['tp1_hit']:
                if (direction == 1 and candle_high >= position['tp1']) or (direction == -1 and candle_low <= position['tp1']):
                    position['tp1_hit'] = True
                    balance += position['risk_amount'] * 0.5  # Lock half
                    position['sl'] = position['entry']  # Move SL to breakeven

            # Full TP2 Hit
            if (position['type'] == 'BUY' and candle_high >= position['tp2']) or \
               (position['type'] == 'SELL' and candle_low <= position['tp2']):
                position['exit'] = position['tp2']
                position['result'] = "WIN"
                pnl = (position['exit'] - position['entry']) * direction * position['lot_size']
                balance += pnl * 0.5  # Other half
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['pnl'] = pnl
                trades.append(position)
                position = None
                daily_trades[date] += 1
                continue

            # Trailing Stop Trigger
            if position['tp1_hit'] and not position['trail_triggered']:
                if (direction == 1 and candle_high >= position['entry'] + TRAIL_AFTER_MULTIPLIER * atr) or \
                   (direction == -1 and candle_low <= position['entry'] - TRAIL_AFTER_MULTIPLIER * atr):
                    position['trail_triggered'] = True

            # Active Trailing
            if position['trail_triggered']:
                if position['type'] == 'BUY':
                    position['sl'] = max(position['sl'], candle_low - 0.5 * atr)
                elif position['type'] == 'SELL':
                    position['sl'] = min(position['sl'], candle_high + 0.5 * atr)

            # SL Hit
            if (position['type'] == 'BUY' and candle_low <= position['sl']) or \
               (position['type'] == 'SELL' and candle_high >= position['sl']):
                position['exit'] = position['sl']
                pnl = (position['exit'] - position['entry']) * direction * position['lot_size']
                position['result'] = "BE" if pnl == 0 else "LOSS"
                balance += pnl
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['pnl'] = pnl
                trades.append(position)
                position = None
                daily_trades[date] += 1
                continue

            # Max candle limit
            if position['age'] >= MAX_CANDLE_AGE:
                position['exit'] = candle['close']
                pnl = (position['exit'] - position['entry']) * direction * position['lot_size']
                position['result'] = "EXIT"
                balance += pnl
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['pnl'] = pnl
                trades.append(position)
                position = None
                daily_trades[date] += 1

    return trades

if __name__ == "__main__":
    if connect_mt5():
        df = fetch_data(PAIR, TIMEFRAME, bars=15000)
        if df is not None:
            trades = simulate_sniper(df)

            wins = sum(1 for t in trades if t['pnl'] > 0 and t['result'] == "WIN")
            losses = sum(1 for t in trades if t['pnl'] < 0 and t['result'] == "LOSS")
            breakevens = sum(1 for t in trades if t['result'] == "BE")
            exits = sum(1 for t in trades if t['result'] == "EXIT")
            total = len(trades)
            net_profit = sum(t['pnl'] for t in trades)
            win_rate = (wins / total) * 100 if total > 0 else 0

            print(f"\n📊 Total Trades: {total}")
            print(f"✅ Wins: {wins}")
            print(f"❌ Losses: {losses}")
            print(f"➖ Breakeven: {breakevens}")
            print(f"🚪 Exits: {exits}")
            print(f"💰 Net Profit: ${net_profit:.2f}")
            print(f"🏆 Win Rate: {win_rate:.2f}%")

            df_out = pd.DataFrame(trades)
            df_out.to_csv("strategy_sniper_v4_1_backtest.csv", index=False)
            print("📦 Trades exported to strategy_sniper_v4_1_backtest.csv")

        mt5.shutdown()

import MetaTrader5 as mt5
import pandas as pd
from config import PAIR, TIMEFRAME
from strategy_scalper import calculate_indicators, generate_signal

ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE_PERCENT = 1.0

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

def simulate_scalping(df):
    trades = []
    df = calculate_indicators(df)
    position = None
    balance = ACCOUNT_BALANCE

    for i in range(50, len(df)):
        slice_df = df.iloc[:i+1].copy()
        candle = df.iloc[i]
        signal = generate_signal(slice_df)
        atr = slice_df['atr'].iloc[-1]

        if pd.isna(atr) or atr == 0:
            continue

        sl_points = atr
        tp_points = atr
        risk_amount = (RISK_PER_TRADE_PERCENT / 100) * balance
        lot_size = risk_amount / sl_points if sl_points != 0 else 0

        if position is None and signal in ["BUY", "SELL"]:
            entry = candle['close']
            sl = entry - sl_points if signal == "BUY" else entry + sl_points
            tp = entry + tp_points if signal == "BUY" else entry - tp_points
            position = {
                "type": signal,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "entry_time": candle['time'],
                "entry_index": i,
                "lot_size": lot_size,
                "risk_amount": risk_amount,
                "age": 0,
                "breakeven_set": False,
                "profit_locked": False
            }

        elif position:
            candle_high = candle['high']
            candle_low = candle['low']
            position['age'] += 1

            if position['type'] == 'BUY':
                profit = candle_high - position['entry']

                if not position['breakeven_set'] and profit >= 5:
                    position['sl'] = position['entry']
                    position['breakeven_set'] = True

                if profit >= 8 and not position.get("profit_locked"):
                    position['sl'] = max(position['sl'], position['entry'] + 5)
                    position['profit_locked'] = True

                if profit >= 20:
                    position['sl'] = max(position['sl'], candle_low - 10)

                if candle_low <= position['sl']:
                    position['exit'] = position['sl']
                    position['result'] = 'LOSS' if position['sl'] < position['entry'] else 'BE'
                    if position['result'] == 'LOSS':
                        balance -= position['risk_amount']
                elif candle_high >= position['tp']:
                    position['exit'] = position['tp']
                    position['result'] = 'WIN'
                    balance += position['risk_amount']

            elif position['type'] == 'SELL':
                profit = position['entry'] - candle_low

                if not position['breakeven_set'] and profit >= 5:
                    position['sl'] = position['entry']
                    position['breakeven_set'] = True

                if profit >= 8 and not position.get("profit_locked"):
                    position['sl'] = min(position['sl'], position['entry'] - 5)
                    position['profit_locked'] = True

                if profit >= 20:
                    position['sl'] = min(position['sl'], candle_high + 10)

                if candle_high >= position['sl']:
                    position['exit'] = position['sl']
                    position['result'] = 'LOSS' if position['sl'] > position['entry'] else 'BE'
                    if position['result'] == 'LOSS':
                        balance -= position['risk_amount']
                elif candle_low <= position['tp']:
                    position['exit'] = position['tp']
                    position['result'] = 'WIN'
                    balance += position['risk_amount']

            if position['age'] >= 10 and 'result' not in position:
                position['exit'] = candle['close']
                position['result'] = 'EXIT'

            if 'result' in position:
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['final_balance'] = balance
                trades.append(position)
                position = None

    return trades

if __name__ == "__main__":
    if connect_mt5():
        df = fetch_data(PAIR, TIMEFRAME, bars=15000)
        if df is not None:
            trades = simulate_scalping(df)

            wins = sum(1 for t in trades if t['result'] == 'WIN')
            losses = sum(1 for t in trades if t['result'] == 'LOSS')
            breakevens = sum(1 for t in trades if t['result'] == 'BE')
            exits = sum(1 for t in trades if t['result'] == 'EXIT')
            total = len(trades)
            win_rate = (wins / total) * 100 if total > 0 else 0

            print(f"\n📊 Total Trades: {total}")
            print(f"✅ Wins: {wins}")
            print(f"❌ Losses: {losses}")
            print(f"➖ Breakeven: {breakevens}")
            print(f"🚪 Exits: {exits}")
            print(f"🏆 Win Rate: {win_rate:.2f}%")

            df_out = pd.DataFrame(trades)
            df_out.to_csv("strategy_scalper_backtest.csv", index=False)
            print("📦 Trades exported to strategy_scalper_backtest.csv")

        mt5.shutdown()

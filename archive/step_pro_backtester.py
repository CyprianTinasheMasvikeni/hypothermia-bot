import MetaTrader5 as mt5
import pandas as pd
from config import PAIR, TIMEFRAME
from strategy_step_pro import calculate_indicators, generate_signal

# === Config ===
ACCOUNT_BALANCE = 100.0
RISK_PER_TRADE_PERCENT = 1.0
MAX_TRADES_PER_DAY = 3

# === Trade Management ===
TP_MULTIPLIER = 2.5
TRAIL_AFTER_R = 1.5
TRAIL_STEP = 0.5
MAX_AGE = 15

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
        "H1": mt5.TIMEFRAME_H1,
    }
    if timeframe not in tf_map:
        print("❌ Invalid timeframe")
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
    if rates is None or rates.size == 0:
        print(f"⚠️ No data returned for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def simulate_trades(df):
    df = calculate_indicators(df)
    trades = []
    position = None
    balance = ACCOUNT_BALANCE
    daily_trades = {}

    for i in range(60, len(df)):
        candle = df.iloc[i]
        slice_df = df.iloc[:i+1]
        date = candle['time'].date()
        daily_trades.setdefault(date, 0)

        if daily_trades[date] >= MAX_TRADES_PER_DAY:
            continue

        signal = generate_signal(slice_df)
        atr = slice_df['atr'].iloc[-1]
        if pd.isna(atr) or atr == 0:
            continue

        sl = atr
        tp = atr * TP_MULTIPLIER
        risk = (RISK_PER_TRADE_PERCENT / 100) * balance
        lot = risk / sl if sl > 0 else 0

        if position is None and signal in ["BUY", "SELL"]:
            entry = candle['close']
            sl_price = entry - sl if signal == "BUY" else entry + sl
            tp_price = entry + tp if signal == "BUY" else entry - tp
            position = {
                "type": signal,
                "entry": entry,
                "sl": sl_price,
                "tp": tp_price,
                "entry_time": candle['time'],
                "entry_index": i,
                "lot_size": lot,
                "risk_amount": risk,
                "breakeven_set": False,
                "trail_started": False,
                "age": 0
            }

        elif position:
            direction = 1 if position['type'] == "BUY" else -1
            high = candle['high']
            low = candle['low']
            position['age'] += 1

            move_from_entry = (high - position['entry']) if position['type'] == "BUY" else (position['entry'] - low)

            # Start trailing
            if not position['trail_started'] and move_from_entry >= TRAIL_AFTER_R * atr:
                position['trail_started'] = True

            if position['trail_started']:
                if position['type'] == "BUY":
                    new_sl = high - TRAIL_STEP * atr
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                else:
                    new_sl = low + TRAIL_STEP * atr
                    if new_sl < position['sl']:
                        position['sl'] = new_sl

            # Check SL or TP hits
            exit_hit = False
            if (position['type'] == "BUY" and low <= position['sl']):
                position['exit'] = position['sl']
                exit_hit = True
            elif (position['type'] == "SELL" and high >= position['sl']):
                position['exit'] = position['sl']
                exit_hit = True
            elif (position['type'] == "BUY" and high >= position['tp']):
                position['exit'] = position['tp']
                exit_hit = True
            elif (position['type'] == "SELL" and low <= position['tp']):
                position['exit'] = position['tp']
                exit_hit = True

            if exit_hit:
                pnl = (position['exit'] - position['entry']) * direction * lot
                position['result'] = "WIN" if pnl > 0 else "LOSS"
                balance += pnl
                daily_trades[date] += 1
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['pnl'] = pnl
                position['final_balance'] = balance
                trades.append(position)
                position = None

            # Max timeout
            if position and position['age'] >= MAX_AGE:
                position['exit'] = candle['close']
                pnl = (position['exit'] - position['entry']) * direction * lot
                position['result'] = "EXIT"
                balance += pnl
                daily_trades[date] += 1
                position['exit_time'] = candle['time']
                position['exit_index'] = i
                position['pnl'] = pnl
                position['final_balance'] = balance
                trades.append(position)
                position = None

    return trades

if __name__ == "__main__":
    if connect_mt5():
        df = fetch_data(PAIR, TIMEFRAME, bars=15000)
        if df is not None:
            trades = simulate_trades(df)

            wins = sum(1 for t in trades if t['result'] == "WIN")
            losses = sum(1 for t in trades if t['result'] == "LOSS")
            exits = sum(1 for t in trades if t['result'] == "EXIT")
            total = len(trades)
            net_profit = sum(t['pnl'] for t in trades)
            win_rate = (wins / total) * 100 if total > 0 else 0

            print(f"\n📊 Total Trades: {total}")
            print(f"✅ Wins: {wins}")
            print(f"❌ Losses: {losses}")
            print(f"🚪 Exits: {exits}")
            print(f"💰 Net Profit: ${net_profit:.2f}")
            print(f"🏆 Win Rate: {win_rate:.2f}%")

            pd.DataFrame(trades).to_csv("strategy_step_pro_backtest.csv", index=False)
            print("📦 Trades exported to strategy_step_pro_backtest.csv")

        mt5.shutdown()

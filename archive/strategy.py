import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from config import EMA_FAST, EMA_SLOW, RSI_PERIOD


def get_data(symbol, timeframe, candles=500):
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1
    }

    if timeframe not in timeframe_map:
        print(f"❌ Invalid timeframe: {timeframe}")
        return None

    bars = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, candles)

    if bars is None or len(bars) == 0:
        print(f"⚠️ No data received for {symbol} on {timeframe}")
        return None

    df = pd.DataFrame(bars)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def calculate_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def generate_signal(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    if (
        latest['ema_fast'] > latest['ema_slow'] and
        previous['ema_fast'] <= previous['ema_slow'] and
        latest['rsi'] > 50
    ):
        return "BUY"

    elif (
        latest['ema_fast'] < latest['ema_slow'] and
        previous['ema_fast'] >= previous['ema_slow'] and
        latest['rsi'] < 50
    ):
        return "SELL"

    else:
        return "WAIT"

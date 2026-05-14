import pandas as pd
import numpy as np

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

def calculate_indicators(df):
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi'].diff()

    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    df['atr_40p'] = df['atr'].rolling(50).quantile(0.4)
    return df

def strong_candle(candle):
    body = abs(candle['close'] - candle['open'])
    range_total = candle['high'] - candle['low']
    if range_total == 0:
        return False
    body_ratio = body / range_total
    wick_size = range_total - body
    wick_ratio = wick_size / range_total
    return body_ratio >= 0.5 and wick_ratio <= 0.5

def generate_signal(df):
    if len(df) < 60:
        return "WAIT"

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Filters
    if pd.isna(latest['atr']) or latest['atr'] < latest['atr_40p']:
        return "WAIT"

    if not strong_candle(latest):
        return "WAIT"

    ema_slope = df['ema_fast'].iloc[-1] - df['ema_fast'].iloc[-5]
    if abs(ema_slope) < 1.0:
        return "WAIT"

    if abs(latest['rsi_slope']) < 0.3:
        return "WAIT"

    # BUY
    if (
        latest['ema_fast'] > latest['ema_slow'] and
        latest['close'] > latest['ema_fast'] and
        latest['close'] > latest['ema_slow'] and
        latest['close'] > latest['open'] and
        latest['rsi'] > 50 and latest['rsi'] > prev['rsi']
    ):
        return "BUY"

    # SELL
    if (
        latest['ema_fast'] < latest['ema_slow'] and
        latest['close'] < latest['ema_fast'] and
        latest['close'] < latest['ema_slow'] and
        latest['close'] < latest['open'] and
        latest['rsi'] < 50 and latest['rsi'] < prev['rsi']
    ):
        return "SELL"

    return "WAIT"

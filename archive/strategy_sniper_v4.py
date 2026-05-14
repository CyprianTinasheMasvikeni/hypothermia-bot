import pandas as pd
import numpy as np

# Strategy settings
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14

def calculate_indicators(df):
    df = df.copy()

    # Moving Averages
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    # RSI and slope
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi'].diff()

    # ATR and 40% filter
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    df['atr_40p'] = df['atr'].rolling(50).quantile(0.4)

    return df

def strong_engulfing_candle(current, previous):
    body_current = abs(current['close'] - current['open'])
    body_previous = abs(previous['close'] - previous['open'])

    return (
        body_current > body_previous and
        current['high'] > previous['high'] and
        current['low'] < previous['low']
    )

def generate_signal(df):
    if len(df) < 60:
        return "WAIT"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    # Filters
    if pd.isna(current['atr']) or pd.isna(current['atr_40p']) or current['atr'] < current['atr_40p']:
        return "WAIT"

    if abs(current['rsi_slope']) < 0.1:
        return "WAIT"

    # BUY logic
    if (
        current['ema_fast'] > current['ema_slow'] and
        current['close'] > current['ema_fast'] and
        current['rsi'] > 50 and
        current['rsi'] > previous['rsi'] and
        strong_engulfing_candle(current, previous)
    ):
        return "BUY"

    # SELL logic
    if (
        current['ema_fast'] < current['ema_slow'] and
        current['close'] < current['ema_fast'] and
        current['rsi'] < 50 and
        current['rsi'] < previous['rsi'] and
        strong_engulfing_candle(current, previous)
    ):
        return "SELL"

    return "WAIT"

import pandas as pd
import numpy as np
from config import EMA_FAST, EMA_SLOW, RSI_PERIOD


def calculate_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df


def is_bullish_rejection(candle, prev_candle):
    body = candle['close'] - candle['open']
    wick = candle['open'] - candle['low']
    prev_body = abs(prev_candle['close'] - prev_candle['open'])

    return (
        body > 0 and
        wick > body and
        body > prev_body * 1.2
    )


def is_bearish_rejection(candle, prev_candle):
    body = candle['open'] - candle['close']
    wick = candle['high'] - candle['open']
    prev_body = abs(prev_candle['close'] - prev_candle['open'])

    return (
        body > 0 and
        wick > body and
        body > prev_body * 1.2
    )


def recent_high_break(df, lookback=10):
    highs = df['high'].iloc[-lookback-2:-2]
    return df['close'].iloc[-2] > highs.max()


def recent_low_break(df, lookback=10):
    lows = df['low'].iloc[-lookback-2:-2]
    return df['close'].iloc[-2] < lows.min()


def generate_signal_v2(df):
    df = calculate_indicators(df)

    if len(df) < 4:
        return "WAIT"

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    prev_candle = df.iloc[-3]

    # Trend check
    if latest['ema_fast'] > latest['ema_slow']:
        trend = "BUY"
    elif latest['ema_fast'] < latest['ema_slow']:
        trend = "SELL"
    else:
        return "WAIT"

    # Entry conditions with RSI filter
    if trend == "BUY":
        if (
            latest['rsi'] > 55 and
            recent_high_break(df) and
            is_bullish_rejection(latest, prev_candle)
        ):
            return "BUY"

    elif trend == "SELL":
        if (
            latest['rsi'] < 45 and
            recent_low_break(df) and
            is_bearish_rejection(latest, prev_candle)
        ):
            return "SELL"

    return "WAIT"

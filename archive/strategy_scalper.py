import pandas as pd
import numpy as np

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
MIN_BODY_RATIO = 0.45
MIN_RSI_SLOPE = 0.1
MIN_ATR_RATIO = 0.00008      # ATR as % of price — works at any price level
MIN_SLOPE_ATR_RATIO = 0.3    # EMA slope must be at least 30% of one ATR


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

    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1)),
        ),
    )
    df['atr'] = df['tr'].rolling(window=ATR_PERIOD).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    df['ema_fast_slope'] = df['ema_fast'] - df['ema_fast'].shift(5)

    return df


def strong_candle(candle):
    body = abs(candle['close'] - candle['open'])
    range_total = candle['high'] - candle['low']
    if range_total == 0:
        return False
    return body / range_total >= MIN_BODY_RATIO


def classify_trend_bias(latest, prev):
    ema_bullish = latest['ema_fast'] > latest['ema_slow']
    ema_bearish = latest['ema_fast'] < latest['ema_slow']
    rsi_rising = latest['rsi'] > prev['rsi']
    rsi_falling = latest['rsi'] < prev['rsi']
    above_fast = latest['close'] > latest['ema_fast']
    below_fast = latest['close'] < latest['ema_fast']

    if ema_bullish and above_fast and latest['rsi'] > 50 and rsi_rising:
        return "STRONG_BUY"
    if ema_bearish and below_fast and latest['rsi'] < 50 and rsi_falling:
        return "STRONG_SELL"
    if ema_bullish and latest['close'] > latest['ema_slow']:
        return "WEAK_BUY"
    if ema_bearish and latest['close'] < latest['ema_slow']:
        return "WEAK_SELL"
    return "NEUTRAL"


def analyze_setup(df):
    if len(df) < 60:
        return {"signal": "WAIT", "reason": "Not enough data.", "checks": {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    atr = latest['atr']
    atr_ratio = latest['atr_ratio']

    # Normalized slope — works at any price level, not absolute price units
    slope_atr_ratio = abs(latest['ema_fast_slope']) / atr if atr > 0 else 0

    market_choppy = (
        pd.isna(atr_ratio)
        or atr_ratio < MIN_ATR_RATIO
        or slope_atr_ratio < MIN_SLOPE_ATR_RATIO
    )

    trend_bias = classify_trend_bias(latest, prev)

    checks = {
        "trend_bias": trend_bias,
        "market_choppy": market_choppy,
        "atr": round(float(atr), 2),
        "atr_ratio": round(float(atr_ratio), 6),
        "slope_atr_ratio": round(float(slope_atr_ratio), 3),
        "rsi": round(float(latest['rsi']), 2),
        "rsi_slope": round(float(latest['rsi_slope']), 2),
        "ema_fast": round(float(latest['ema_fast']), 2),
        "ema_slow": round(float(latest['ema_slow']), 2),
    }

    if market_choppy:
        return {"signal": "WAIT", "reason": "Market choppy — slope or ATR too low.", "checks": checks}

    candle_strong = strong_candle(latest)
    rsi_momentum_ok = abs(latest['rsi_slope']) >= MIN_RSI_SLOPE
    rsi_rising = latest['rsi'] > prev['rsi']
    rsi_falling = latest['rsi'] < prev['rsi']

    buy_ready = (
        latest['ema_fast'] > latest['ema_slow']
        and latest['close'] > latest['ema_fast']
        and latest['close'] > latest['open']
        and latest['rsi'] > 47
        and rsi_rising
        and candle_strong
        and rsi_momentum_ok
    )
    sell_ready = (
        latest['ema_fast'] < latest['ema_slow']
        and latest['close'] < latest['ema_fast']
        and latest['close'] < latest['open']
        and latest['rsi'] < 53
        and rsi_falling
        and candle_strong
        and rsi_momentum_ok
    )

    if buy_ready:
        return {
            "signal": "BUY",
            "reason": "EMA bullish stack + strong candle + RSI momentum confirmed.",
            "checks": checks,
        }
    if sell_ready:
        return {
            "signal": "SELL",
            "reason": "EMA bearish stack + strong candle + RSI momentum confirmed.",
            "checks": checks,
        }
    return {"signal": "WAIT", "reason": "Conditions not fully aligned.", "checks": checks}


def generate_signal(df):
    return analyze_setup(df)["signal"]

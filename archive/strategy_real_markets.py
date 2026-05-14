import numpy as np
import pandas as pd

EMA_FAST_PERIOD = 21
EMA_PERIOD = 50
EMA_LONG_PERIOD = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
SR_LOOKBACK = 20
TREND_LOOKBACK = 5

MIN_EMA_SLOPE_ATR   = 0.10
WEAK_EMA_SLOPE_ATR  = 0.04
MIN_ATR_RATIO = 0.00008
MAX_PULLBACK_DISTANCE_ATR = 1.2
MIN_CONFIRMATION_BODY_RATIO = 0.5
MIN_CONTINUATION_CLOSE_RATIO = 0.7


def calculate_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema"]      = df["close"].ewm(span=EMA_PERIOD,      adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df["rsi"]       = 100 - (100 / (1 + gain / loss))
    df["rsi_slope"] = df["rsi"].diff()
    df["tr"] = np.maximum(df["high"] - df["low"],
                  np.maximum(abs(df["high"] - df["close"].shift(1)),
                             abs(df["low"]  - df["close"].shift(1))))
    df["atr"]            = df["tr"].rolling(ATR_PERIOD).mean()
    df["recent_high"]    = df["high"].rolling(SR_LOOKBACK).max()
    df["recent_low"]     = df["low"].rolling(SR_LOOKBACK).min()
    df["ema_slope"]      = df["ema"]      - df["ema"].shift(TREND_LOOKBACK)
    df["ema_long_slope"] = df["ema_long"] - df["ema_long"].shift(TREND_LOOKBACK)
    df["ema_fast_slope"] = df["ema_fast"] - df["ema_fast"].shift(TREND_LOOKBACK)
    df["atr_ratio"]          = df["atr"] / df["close"]
    atr_safe = df["atr"].replace(0, np.nan)
    df["ema_slope_atr"]      = df["ema_slope"]      / atr_safe
    df["ema_long_slope_atr"] = df["ema_long_slope"] / atr_safe
    return df


def is_engulfing(latest, prev):
    bullish = (latest["close"] > latest["open"] and prev["close"] < prev["open"]
               and latest["close"] > prev["open"] and latest["open"] < prev["close"])
    bearish = (latest["close"] < latest["open"] and prev["close"] > prev["open"]
               and latest["close"] < prev["open"] and latest["open"] > prev["close"])
    return bullish or bearish


def candle_body_ratio(c):
    r = c["high"] - c["low"]
    return abs(c["close"] - c["open"]) / r if r > 0 else 0.0


def candle_close_position(c):
    r = c["high"] - c["low"]
    return (c["close"] - c["low"]) / r if r > 0 else 0.5


def wick_ratios(c):
    r = c["high"] - c["low"]
    if r <= 0:
        return 0.0, 0.0
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    return upper / r, lower / r


def classify_trend_bias(latest):
    atr = latest["atr"]
    if atr <= 0 or np.isnan(atr):
        return "NEUTRAL"
    s50  = latest["ema_slope_atr"]
    s200 = latest["ema_long_slope_atr"]
    price_above_200    = latest["close"] > latest["ema_long"]
    price_below_200    = latest["close"] < latest["ema_long"]
    ema_stack_bullish  = latest["ema_fast"] > latest["ema"] > latest["ema_long"]
    ema_stack_bearish  = latest["ema_fast"] < latest["ema"] < latest["ema_long"]
    ema50_rising   = s50  >  MIN_EMA_SLOPE_ATR
    ema50_falling  = s50  < -MIN_EMA_SLOPE_ATR
    ema200_rising  = s200 >  WEAK_EMA_SLOPE_ATR
    ema200_falling = s200 < -WEAK_EMA_SLOPE_ATR

    if price_above_200 and ema_stack_bullish and ema50_rising  and ema200_rising:
        return "STRONG_BUY"
    if price_below_200 and ema_stack_bearish and ema50_falling and ema200_falling:
        return "STRONG_SELL"
    if (price_above_200 and latest["ema"] > latest["ema_long"] and s200 > 0
            and latest["ema_fast"] >= latest["ema"] - latest["atr"] * 0.35):
        return "WEAK_BUY"
    if (price_below_200 and latest["ema"] < latest["ema_long"] and s200 < 0
            and latest["ema_fast"] <= latest["ema"] + latest["atr"] * 0.35):
        return "WEAK_SELL"
    return "NEUTRAL"


def analyze_setup(df):
    if len(df) < max(EMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD, SR_LOOKBACK) + 10:
        return {"signal": "WAIT", "reason": "Not enough data.", "checks": {}}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    atr    = latest["atr"]
    s50    = latest["ema_slope_atr"]

    body_ratio  = candle_body_ratio(latest)
    close_pos   = candle_close_position(latest)
    bullish_candle = latest["close"] > latest["open"]
    bearish_candle = latest["close"] < latest["open"]
    engulfing      = is_engulfing(latest, prev)

    bullish_continuation = (bullish_candle and body_ratio >= MIN_CONFIRMATION_BODY_RATIO
                            and close_pos >= MIN_CONTINUATION_CLOSE_RATIO)
    bearish_continuation = (bearish_candle and body_ratio >= MIN_CONFIRMATION_BODY_RATIO
                            and close_pos <= 1 - MIN_CONTINUATION_CLOSE_RATIO)

    bullish_confirmation = engulfing or bullish_continuation
    bearish_confirmation = engulfing or bearish_continuation
    confirmation_type = ("engulfing" if engulfing
                         else "continuation" if (bullish_continuation or bearish_continuation)
                         else "none")

    trend_up   = latest["close"] > latest["ema"] and s50 >  MIN_EMA_SLOPE_ATR
    trend_down = latest["close"] < latest["ema"] and s50 < -MIN_EMA_SLOPE_ATR

    market_choppy = (np.isnan(latest["atr_ratio"])
                     or latest["atr_ratio"] < MIN_ATR_RATIO
                     or abs(s50) < WEAK_EMA_SLOPE_ATR)

    near_support    = abs(latest["low"]  - latest["recent_low"])  <= atr * MAX_PULLBACK_DISTANCE_ATR
    near_resistance = abs(latest["high"] - latest["recent_high"]) <= atr * MAX_PULLBACK_DISTANCE_ATR
    pulled_into_ema = abs(latest["close"] - latest["ema"])        <= atr * MAX_PULLBACK_DISTANCE_ATR
    valid_pullback_buy  = pulled_into_ema or near_support
    valid_pullback_sell = pulled_into_ema or near_resistance

    momentum_up   = latest["rsi"] > 50 and latest["rsi_slope"] > 0 and df["rsi_slope"].iloc[-2] > 0
    momentum_down = latest["rsi"] < 50 and latest["rsi_slope"] < 0 and df["rsi_slope"].iloc[-2] < 0

    trend_bias  = classify_trend_bias(latest)
    strong_bias = trend_bias in {"STRONG_BUY", "STRONG_SELL"}

    buy_ready = all([trend_up,   strong_bias, not market_choppy,
                     valid_pullback_buy,  bullish_candle, bullish_confirmation, momentum_up])
    sell_ready = all([trend_down, strong_bias, not market_choppy,
                      valid_pullback_sell, bearish_candle, bearish_confirmation, momentum_down])

    checks = {
        "trend_bias":        trend_bias,
        "market_choppy":     market_choppy,
        "confirmation_type": confirmation_type,
        "momentum_up":       momentum_up,
        "momentum_down":     momentum_down,
        "rsi":          round(float(latest["rsi"]), 2),
        "atr":          round(float(atr), 6),
        "ema_slope_atr": round(float(s50), 4) if not np.isnan(s50) else 0,
    }

    if buy_ready:
        return {"signal": "BUY",  "reason": f"Strong bullish + pullback + {confirmation_type}.", "checks": checks}
    if sell_ready:
        return {"signal": "SELL", "reason": f"Strong bearish + pullback + {confirmation_type}.", "checks": checks}
    return {"signal": "WAIT", "reason": "Setup not aligned.", "checks": checks}


def generate_signal(df):
    return analyze_setup(df)["signal"]

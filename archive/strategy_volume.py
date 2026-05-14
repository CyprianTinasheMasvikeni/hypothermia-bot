import numpy as np
import pandas as pd


EMA_FAST = 9
EMA_SLOW = 21
EMA_LONG = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 6
MIN_ATR_RATIO = 0.00005
MIN_BODY_RATIO = 0.35
MIN_CLOSE_POSITION = 0.65


def calculate_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi"].diff()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr"] = df["tr"].rolling(window=ATR_PERIOD).mean()
    df["atr_ratio"] = df["atr"] / df["close"]

    df["recent_high"] = df["high"].rolling(window=BREAKOUT_LOOKBACK).max()
    df["recent_low"] = df["low"].rolling(window=BREAKOUT_LOOKBACK).min()
    return df


def candle_body_ratio(candle):
    total_range = candle["high"] - candle["low"]
    if total_range <= 0:
        return 0.0
    return abs(candle["close"] - candle["open"]) / total_range


def close_position_ratio(candle):
    total_range = candle["high"] - candle["low"]
    if total_range <= 0:
        return 0.5
    return (candle["close"] - candle["low"]) / total_range


def analyze_setup(df):
    if len(df) < max(EMA_LONG, ATR_PERIOD, BREAKOUT_LOOKBACK) + 5:
        return {
            "signal": "WAIT",
            "reason": "Not enough candle history yet.",
            "checks": {"data_ready": False},
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    body_ratio = candle_body_ratio(latest)
    close_pos = close_position_ratio(latest)
    bullish_candle = latest["close"] > latest["open"]
    bearish_candle = latest["close"] < latest["open"]

    fast_above_slow = latest["ema_fast"] > latest["ema_slow"]
    fast_below_slow = latest["ema_fast"] < latest["ema_slow"]
    price_above_long = latest["close"] > latest["ema_long"]
    price_below_long = latest["close"] < latest["ema_long"]
    momentum_up = latest["rsi"] > 52 and latest["rsi_slope"] > 0
    momentum_down = latest["rsi"] < 48 and latest["rsi_slope"] < 0
    enough_range = not pd.isna(latest["atr_ratio"]) and latest["atr_ratio"] >= MIN_ATR_RATIO

    broke_recent_high = latest["close"] >= df["high"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max()
    broke_recent_low = latest["close"] <= df["low"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].min()

    bullish_confirmation = bullish_candle and body_ratio >= MIN_BODY_RATIO and close_pos >= MIN_CLOSE_POSITION
    bearish_confirmation = bearish_candle and body_ratio >= MIN_BODY_RATIO and close_pos <= (1 - MIN_CLOSE_POSITION)

    strong_buy = all(
        [
            fast_above_slow,
            price_above_long,
            bullish_confirmation,
            momentum_up,
            broke_recent_high,
            enough_range,
            latest["close"] > prev["close"],
        ]
    )
    strong_sell = all(
        [
            fast_below_slow,
            price_below_long,
            bearish_confirmation,
            momentum_down,
            broke_recent_low,
            enough_range,
            latest["close"] < prev["close"],
        ]
    )

    trend_bias = "NEUTRAL"
    if fast_above_slow and price_above_long:
        trend_bias = "STRONG_BUY"
    elif fast_below_slow and price_below_long:
        trend_bias = "STRONG_SELL"

    checks = {
        "trend_up": fast_above_slow and price_above_long,
        "trend_down": fast_below_slow and price_below_long,
        "weak_trend_up": fast_above_slow,
        "weak_trend_down": fast_below_slow,
        "trend_bias": trend_bias,
        "market_choppy": not enough_range,
        "price_above_200": price_above_long,
        "price_below_200": price_below_long,
        "ema_stack_bullish": fast_above_slow and latest["ema_slow"] > latest["ema_long"],
        "ema_stack_bearish": fast_below_slow and latest["ema_slow"] < latest["ema_long"],
        "valid_pullback_buy": fast_above_slow,
        "valid_pullback_sell": fast_below_slow,
        "bullish_candle": bullish_candle,
        "bearish_candle": bearish_candle,
        "engulfing": False,
        "bullish_continuation": bullish_confirmation,
        "bearish_continuation": bearish_confirmation,
        "bullish_pinbar": False,
        "bearish_pinbar": False,
        "bullish_confirmation": bullish_confirmation,
        "bearish_confirmation": bearish_confirmation,
        "confirmation_type": "continuation" if bullish_confirmation or bearish_confirmation else "none",
        "broke_high": broke_recent_high,
        "broke_low": broke_recent_low,
        "bullish_breakout_ok": broke_recent_high,
        "bearish_breakout_ok": broke_recent_low,
        "strong_confirmation": bullish_confirmation or bearish_confirmation,
        "momentum_up": momentum_up,
        "momentum_down": momentum_down,
        "ema_fast": round(float(latest["ema_fast"]), 2),
        "ema": round(float(latest["ema_slow"]), 2),
        "ema_long": round(float(latest["ema_long"]), 2),
        "ema_fast_slope": round(float(latest["ema_fast"] - df["ema_fast"].iloc[-4]), 2),
        "ema_slope": round(float(latest["ema_slow"] - df["ema_slow"].iloc[-4]), 2),
        "ema_long_slope": round(float(latest["ema_long"] - df["ema_long"].iloc[-4]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "rsi_slope": round(float(latest["rsi_slope"]), 2),
        "atr": round(float(latest["atr"]), 2),
        "atr_ratio": round(float(latest["atr_ratio"]), 6),
        "recent_high": round(float(latest["recent_high"]), 2),
        "recent_low": round(float(latest["recent_low"]), 2),
        "body_ratio": round(body_ratio, 3),
        "close_position": round(close_pos, 3),
        "upper_wick_ratio": round((latest["high"] - max(latest["open"], latest["close"])) / max(latest["high"] - latest["low"], 1e-9), 3),
        "lower_wick_ratio": round((min(latest["open"], latest["close"]) - latest["low"]) / max(latest["high"] - latest["low"], 1e-9), 3),
    }

    if strong_buy:
        return {
            "signal": "BUY",
            "reason": "Fast momentum breakout aligns with bullish trend structure.",
            "checks": checks,
        }
    if strong_sell:
        return {
            "signal": "SELL",
            "reason": "Fast momentum breakout aligns with bearish trend structure.",
            "checks": checks,
        }

    if not enough_range:
        reason = "No trade: volatility is too low for the volume strategy."
    elif trend_bias == "NEUTRAL":
        reason = "No trade: fast trend structure is unclear."
    elif trend_bias == "STRONG_BUY" and not bullish_confirmation:
        reason = "No trade: bullish continuation candle is too weak."
    elif trend_bias == "STRONG_SELL" and not bearish_confirmation:
        reason = "No trade: bearish continuation candle is too weak."
    elif trend_bias == "STRONG_BUY" and not broke_recent_high:
        reason = "No trade: bullish breakout did not clear recent highs."
    elif trend_bias == "STRONG_SELL" and not broke_recent_low:
        reason = "No trade: bearish breakout did not clear recent lows."
    else:
        reason = "No trade: setup is incomplete."

    return {"signal": "WAIT", "reason": reason, "checks": checks}


def generate_signal(df):
    return analyze_setup(df)["signal"]

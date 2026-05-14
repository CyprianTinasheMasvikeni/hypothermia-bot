import numpy as np
import pandas as pd


EMA_FAST_PERIOD = 21
EMA_PERIOD = 50
EMA_LONG_PERIOD = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
SR_LOOKBACK = 20
TREND_LOOKBACK = 5
MIN_EMA_SLOPE = 0.6
WEAK_EMA_SLOPE = 0.2
MIN_ATR_RATIO = 0.00008
MAX_PULLBACK_DISTANCE_ATR = 1.2
MIN_CONFIRMATION_BODY_RATIO = 0.5
MIN_CONTINUATION_CLOSE_RATIO = 0.7
MIN_PINBAR_WICK_RATIO = 0.45


def calculate_indicators(df):
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()

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

    df["recent_high"] = df["high"].rolling(window=SR_LOOKBACK).max()
    df["recent_low"] = df["low"].rolling(window=SR_LOOKBACK).min()
    df["ema_fast_slope"] = df["ema_fast"] - df["ema_fast"].shift(TREND_LOOKBACK)
    df["ema_slope"] = df["ema"] - df["ema"].shift(TREND_LOOKBACK)
    df["ema_long_slope"] = df["ema_long"] - df["ema_long"].shift(TREND_LOOKBACK)
    df["atr_ratio"] = df["atr"] / df["close"]

    return df


def is_engulfing(latest, prev):
    bullish = (
        latest["close"] > latest["open"]
        and prev["close"] < prev["open"]
        and latest["close"] > prev["open"]
        and latest["open"] < prev["close"]
    )
    bearish = (
        latest["close"] < latest["open"]
        and prev["close"] > prev["open"]
        and latest["close"] < prev["open"]
        and latest["open"] > prev["close"]
    )
    return bullish or bearish


def candle_body_ratio(candle):
    total_range = candle["high"] - candle["low"]
    if total_range <= 0:
        return 0.0
    body = abs(candle["close"] - candle["open"])
    return body / total_range


def candle_close_position(candle):
    total_range = candle["high"] - candle["low"]
    if total_range <= 0:
        return 0.5
    return (candle["close"] - candle["low"]) / total_range


def wick_ratios(candle):
    total_range = candle["high"] - candle["low"]
    if total_range <= 0:
        return 0.0, 0.0

    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    return upper_wick / total_range, lower_wick / total_range


def classify_trend_bias(latest):
    price_above_200 = latest["close"] > latest["ema_long"]
    price_below_200 = latest["close"] < latest["ema_long"]
    ema_stack_bullish = latest["ema_fast"] > latest["ema"] > latest["ema_long"]
    ema_stack_bearish = latest["ema_fast"] < latest["ema"] < latest["ema_long"]
    ema50_rising = latest["ema_slope"] > MIN_EMA_SLOPE
    ema50_falling = latest["ema_slope"] < -MIN_EMA_SLOPE
    ema200_rising = latest["ema_long_slope"] > WEAK_EMA_SLOPE
    ema200_falling = latest["ema_long_slope"] < -WEAK_EMA_SLOPE

    strong_buy = price_above_200 and ema_stack_bullish and ema50_rising and ema200_rising
    strong_sell = price_below_200 and ema_stack_bearish and ema50_falling and ema200_falling

    weak_buy = (
        not strong_buy
        and price_above_200
        and latest["ema"] > latest["ema_long"]
        and latest["ema_long_slope"] > 0
        and latest["ema_fast"] >= latest["ema"] - max(latest["atr"] * 0.35, 0.1)
    )
    weak_sell = (
        not strong_sell
        and price_below_200
        and latest["ema"] < latest["ema_long"]
        and latest["ema_long_slope"] < 0
        and latest["ema_fast"] <= latest["ema"] + max(latest["atr"] * 0.35, 0.1)
    )

    if strong_buy:
        return "STRONG_BUY"
    if strong_sell:
        return "STRONG_SELL"
    if weak_buy:
        return "WEAK_BUY"
    if weak_sell:
        return "WEAK_SELL"
    return "NEUTRAL"


def analyze_setup(df):
    if len(df) < max(EMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD, SR_LOOKBACK) + 10:
        return {
            "signal": "WAIT",
            "reason": "Not enough candle history yet.",
            "checks": {"data_ready": False},
        }

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    body_ratio = candle_body_ratio(latest)
    close_position = candle_close_position(latest)
    upper_wick_ratio, lower_wick_ratio = wick_ratios(latest)
    bullish_candle = latest["close"] > latest["open"]
    bearish_candle = latest["close"] < latest["open"]
    engulfing = is_engulfing(latest, prev)

    bullish_continuation = (
        bullish_candle
        and body_ratio >= MIN_CONFIRMATION_BODY_RATIO
        and close_position >= MIN_CONTINUATION_CLOSE_RATIO
    )
    bearish_continuation = (
        bearish_candle
        and body_ratio >= MIN_CONFIRMATION_BODY_RATIO
        and close_position <= (1 - MIN_CONTINUATION_CLOSE_RATIO)
    )
    bullish_pinbar = (
        bullish_candle
        and lower_wick_ratio >= MIN_PINBAR_WICK_RATIO
        and close_position >= 0.55
    )
    bearish_pinbar = (
        bearish_candle
        and upper_wick_ratio >= MIN_PINBAR_WICK_RATIO
        and close_position <= 0.45
    )

    bullish_confirmation = engulfing or bullish_continuation
    bearish_confirmation = engulfing or bearish_continuation
    confirmation_type = "none"
    if engulfing:
        confirmation_type = "engulfing"
    elif bullish_continuation or bearish_continuation:
        confirmation_type = "continuation"

    trend_up = latest["close"] > latest["ema"] and latest["ema_slope"] > MIN_EMA_SLOPE
    trend_down = latest["close"] < latest["ema"] and latest["ema_slope"] < -MIN_EMA_SLOPE
    weak_trend_up = latest["close"] > latest["ema_long"] and latest["ema_long_slope"] > 0
    weak_trend_down = latest["close"] < latest["ema_long"] and latest["ema_long_slope"] < 0

    market_choppy = (
        pd.isna(latest["atr_ratio"])
        or latest["atr_ratio"] < MIN_ATR_RATIO
        or (
            abs(latest["ema_slope"]) < WEAK_EMA_SLOPE
            and abs(latest["ema_long_slope"]) < WEAK_EMA_SLOPE
        )
    )

    near_support = abs(latest["low"] - latest["recent_low"]) <= latest["atr"] * MAX_PULLBACK_DISTANCE_ATR
    near_resistance = abs(latest["high"] - latest["recent_high"]) <= latest["atr"] * MAX_PULLBACK_DISTANCE_ATR
    pulled_into_ema = abs(latest["close"] - latest["ema"]) <= latest["atr"] * MAX_PULLBACK_DISTANCE_ATR
    valid_pullback_buy = pulled_into_ema or near_support
    valid_pullback_sell = pulled_into_ema or near_resistance

    broke_high = latest["close"] > prev["high"]
    broke_low = latest["close"] < prev["low"]
    bullish_breakout_ok = broke_high or bullish_pinbar
    bearish_breakout_ok = broke_low or bearish_pinbar
    momentum_up = latest["rsi"] > 50 and latest["rsi_slope"] > 0 and df["rsi_slope"].iloc[-2] > 0
    momentum_down = latest["rsi"] < 50 and latest["rsi_slope"] < 0 and df["rsi_slope"].iloc[-2] < 0
    strong_confirmation = bullish_confirmation or bearish_confirmation
    trend_bias = classify_trend_bias(latest)

    strong_bias = trend_bias in {"STRONG_BUY", "STRONG_SELL"}

    buy_ready = all(
        [
            trend_up,
            strong_bias,
            not market_choppy,
            valid_pullback_buy,
            bullish_candle,
            bullish_confirmation,
            momentum_up,
        ]
    )
    sell_ready = all(
        [
            trend_down,
            strong_bias,
            not market_choppy,
            valid_pullback_sell,
            bearish_candle,
            bearish_confirmation,
            momentum_down,
        ]
    )

    checks = {
        "trend_up": trend_up,
        "trend_down": trend_down,
        "weak_trend_up": weak_trend_up,
        "weak_trend_down": weak_trend_down,
        "trend_bias": trend_bias,
        "market_choppy": market_choppy,
        "price_above_200": latest["close"] > latest["ema_long"],
        "price_below_200": latest["close"] < latest["ema_long"],
        "ema_stack_bullish": latest["ema_fast"] > latest["ema"] > latest["ema_long"],
        "ema_stack_bearish": latest["ema_fast"] < latest["ema"] < latest["ema_long"],
        "near_support": near_support,
        "near_resistance": near_resistance,
        "pulled_into_ema": pulled_into_ema,
        "valid_pullback_buy": valid_pullback_buy,
        "valid_pullback_sell": valid_pullback_sell,
        "bullish_candle": bullish_candle,
        "bearish_candle": bearish_candle,
        "engulfing": engulfing,
        "bullish_continuation": bullish_continuation,
        "bearish_continuation": bearish_continuation,
        "bullish_pinbar": bullish_pinbar,
        "bearish_pinbar": bearish_pinbar,
        "bullish_confirmation": bullish_confirmation,
        "bearish_confirmation": bearish_confirmation,
        "confirmation_type": confirmation_type,
        "broke_high": broke_high,
        "broke_low": broke_low,
        "bullish_breakout_ok": bullish_breakout_ok,
        "bearish_breakout_ok": bearish_breakout_ok,
        "strong_confirmation": strong_confirmation,
        "momentum_up": momentum_up,
        "momentum_down": momentum_down,
        "body_ratio": round(body_ratio, 3),
        "close_position": round(close_position, 3),
        "upper_wick_ratio": round(upper_wick_ratio, 3),
        "lower_wick_ratio": round(lower_wick_ratio, 3),
        "ema_fast": round(float(latest["ema_fast"]), 2),
        "ema": round(float(latest["ema"]), 2),
        "ema_long": round(float(latest["ema_long"]), 2),
        "ema_fast_slope": round(float(latest["ema_fast_slope"]), 2),
        "ema_slope": round(float(latest["ema_slope"]), 2),
        "ema_long_slope": round(float(latest["ema_long_slope"]), 2),
        "rsi": round(float(latest["rsi"]), 2),
        "rsi_slope": round(float(latest["rsi_slope"]), 2),
        "atr": round(float(latest["atr"]), 2),
        "atr_ratio": round(float(latest["atr_ratio"]), 6),
        "recent_high": round(float(latest["recent_high"]), 2),
        "recent_low": round(float(latest["recent_low"]), 2),
    }

    if buy_ready:
        return {
            "signal": "BUY",
            "reason": (
                f"Trend, pullback, and bullish {confirmation_type} confirmation "
                "aligned on a closed candle."
            ),
            "checks": checks,
        }

    if sell_ready:
        return {
            "signal": "SELL",
            "reason": (
                f"Trend, pullback, and bearish {confirmation_type} confirmation "
                "aligned on a closed candle."
            ),
            "checks": checks,
        }

    if market_choppy and trend_bias == "NEUTRAL":
        reason = "No trade: market is too choppy and the higher structure is neutral."
    elif not (trend_up or trend_down or weak_trend_up or weak_trend_down):
        reason = "No trade: trend is unclear."
    elif (trend_up or weak_trend_up) and not valid_pullback_buy:
        reason = "No trade: bullish structure exists, but no clean pullback into EMA/support."
    elif (trend_down or weak_trend_down) and not valid_pullback_sell:
        reason = "No trade: bearish structure exists, but no clean pullback into EMA/resistance."
    elif (trend_up or weak_trend_up) and not bullish_confirmation:
        reason = "No trade: bullish confirmation candle is not strong enough."
    elif (trend_down or weak_trend_down) and not bearish_confirmation:
        reason = "No trade: bearish confirmation candle is not strong enough."
    elif (trend_up or weak_trend_up) and not bullish_breakout_ok:
        reason = "No trade: bullish confirmation did not break or reject strongly enough."
    elif (trend_down or weak_trend_down) and not bearish_breakout_ok:
        reason = "No trade: bearish confirmation did not break or reject strongly enough."
    elif (trend_up or weak_trend_up) and not momentum_up:
        reason = "No trade: bullish momentum is not strong enough."
    elif (trend_down or weak_trend_down) and not momentum_down:
        reason = "No trade: bearish momentum is not strong enough."
    else:
        reason = "No trade: setup is incomplete."

    return {"signal": "WAIT", "reason": reason, "checks": checks}


def generate_signal(df):
    return analyze_setup(df)["signal"]

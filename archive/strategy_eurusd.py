import numpy as np
import pandas as pd

# EURUSD Strategy
# Regime-adaptive: trend mode (34/55 EMA pullback) + ranging mode (range edge reversion)
# H4 for regime detection, M15 for entry

# EMAs
EMA_TREND_FAST  = 34
EMA_TREND_SLOW  = 55
EMA_BIAS        = 200

# Indicators
ADX_PERIOD      = 14
RSI_PERIOD      = 14
ATR_PERIOD      = 14

# Regime thresholds
ADX_TREND       = 28
ADX_RANGE       = 22    # slightly raised — catches more ranging conditions

# Range detection
RANGE_LOOKBACK  = 28    # shorter lookback = more ranges detected (was 40)
RANGE_EDGE_ATR  = 0.7   # wider edge zone = more entries triggered (was 0.4)

# Entry confirmation
MIN_BODY_RATIO  = 0.50
MIN_RSI_BUY     = 45
MAX_RSI_SELL    = 55

# Trend mode: crossover must be recent
CROSSOVER_LOOKBACK = 10  # EMA crossover must have happened within last 10 candles

# Volatility filter (ATR-based news proxy)
ATR_SPIKE_MULT  = 2.5   # skip if ATR > 2.5x its 20-period average (news spike)
ATR_DEAD_RATIO  = 0.00003  # skip if ATR/price < this (dead market)

# Slope normalized by ATR
MIN_EMA_SLOPE_ATR = 0.08


def calculate_indicators(df):
    df = df.copy()

    df["ema_fast"] = df["close"].ewm(span=EMA_TREND_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_TREND_SLOW, adjust=False).mean()
    df["ema_bias"] = df["close"].ewm(span=EMA_BIAS,       adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df["rsi"]       = 100 - (100 / (1 + gain / loss))
    df["rsi_slope"] = df["rsi"].diff()

    # ATR
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1)),
        ),
    )
    df["atr"]       = df["tr"].rolling(ATR_PERIOD).mean()
    df["atr_avg"]   = df["atr"].rolling(20).mean()   # baseline ATR for spike detection
    df["atr_ratio"] = df["atr"] / df["close"]

    # ADX
    up_move   = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_s      = df["tr"].rolling(ADX_PERIOD).sum()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(ADX_PERIOD).sum() / tr_s
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(ADX_PERIOD).sum() / tr_s
    dx        = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["adx"]       = dx.rolling(ADX_PERIOD).mean()
    df["plus_di"]   = plus_di
    df["minus_di"]  = minus_di

    # EMA slope normalized by ATR
    df["ema_fast_slope"]     = df["ema_fast"] - df["ema_fast"].shift(5)
    df["ema_fast_slope_atr"] = df["ema_fast_slope"] / df["atr"].replace(0, np.nan)

    # Range boundaries (for ranging mode)
    df["range_high"] = df["high"].shift(1).rolling(RANGE_LOOKBACK).max()
    df["range_low"]  = df["low"].shift(1).rolling(RANGE_LOOKBACK).min()
    df["range_mid"]  = (df["range_high"] + df["range_low"]) / 2

    return df


def candle_body_ratio(candle):
    r = candle["high"] - candle["low"]
    return abs(candle["close"] - candle["open"]) / r if r > 0 else 0.0


def candle_close_position(candle):
    r = candle["high"] - candle["low"]
    return (candle["close"] - candle["low"]) / r if r > 0 else 0.5


def is_engulfing(latest, prev):
    bullish = (latest["close"] > latest["open"] and prev["close"] < prev["open"]
               and latest["close"] > prev["open"] and latest["open"] < prev["close"])
    bearish = (latest["close"] < latest["open"] and prev["close"] > prev["open"]
               and latest["close"] < prev["open"] and latest["open"] > prev["close"])
    return bullish or bearish


def is_rejection_candle(candle, direction):
    body   = candle_body_ratio(candle)
    cl_pos = candle_close_position(candle)
    r = candle["high"] - candle["low"]
    if r <= 0:
        return False
    upper_wick = (candle["high"] - max(candle["open"], candle["close"])) / r
    lower_wick = (min(candle["open"], candle["close"]) - candle["low"]) / r
    if direction == "BUY":
        # Bullish rejection: close in upper half, decent body or long lower wick
        return (candle["close"] > candle["open"] and body >= MIN_BODY_RATIO and cl_pos >= 0.55) \
               or (lower_wick >= 0.45 and cl_pos >= 0.5)
    else:
        return (candle["close"] < candle["open"] and body >= MIN_BODY_RATIO and cl_pos <= 0.45) \
               or (upper_wick >= 0.45 and cl_pos <= 0.5)


def get_regime(latest):
    adx = latest["adx"]
    if pd.isna(adx):
        return "NONE"
    if adx > ADX_TREND:
        return "TREND"
    if adx < ADX_RANGE:
        return "RANGE"
    return "NONE"  # ambiguous zone 20-25


def analyze_setup(df, h4_bias="NEUTRAL"):
    if len(df) < max(EMA_BIAS, ADX_PERIOD * 2, RANGE_LOOKBACK) + 10:
        return {"signal": "WAIT", "reason": "Not enough data.", "checks": {}}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    atr    = latest["atr"]

    # Volatility filter — skip news spikes and dead markets
    atr_spiked = not pd.isna(latest["atr_avg"]) and atr > latest["atr_avg"] * ATR_SPIKE_MULT
    atr_dead   = pd.isna(latest["atr_ratio"]) or latest["atr_ratio"] < ATR_DEAD_RATIO
    if atr_spiked:
        return {"signal": "WAIT", "reason": "ATR spike — likely news event, standing down.", "checks": {"atr_spike": True}}
    if atr_dead:
        return {"signal": "WAIT", "reason": "Market dead — ATR too low.", "checks": {"atr_dead": True}}

    regime = get_regime(latest)
    price_above_bias = latest["close"] > latest["ema_bias"]
    price_below_bias = latest["close"] < latest["ema_bias"]

    # H4 bias overrides local bias if provided
    if h4_bias == "BULLISH":
        htf_allows_buy, htf_allows_sell = True, False
    elif h4_bias == "BEARISH":
        htf_allows_buy, htf_allows_sell = False, True
    else:
        # Use M15 200 EMA as fallback bias
        htf_allows_buy  = price_above_bias
        htf_allows_sell = price_below_bias

    signal    = "WAIT"
    reason    = "No setup."
    conf_type = "none"

    # ── TREND MODE ──────────────────────────────────────────────────────────
    if regime == "TREND" and False:  # disabled — trend mode not reliable on Gold
        ema_bullish_stack = latest["ema_fast"] > latest["ema_slow"]
        ema_bearish_stack = latest["ema_fast"] < latest["ema_slow"]

        # Momentum breakout: close breaks prev candle high/low with strong body
        broke_high = latest["close"] > prev["high"]
        broke_low  = latest["close"] < prev["low"]

        # Price must be on the right side of both EMAs
        above_both = latest["close"] > latest["ema_fast"] and latest["close"] > latest["ema_slow"]
        below_both = latest["close"] < latest["ema_fast"] and latest["close"] < latest["ema_slow"]

        rsi_ok_buy  = latest["rsi"] > MIN_RSI_BUY  and latest["rsi"] < 75
        rsi_ok_sell = latest["rsi"] < MAX_RSI_SELL and latest["rsi"] > 25
        rejection_buy  = is_rejection_candle(latest, "BUY")
        rejection_sell = is_rejection_candle(latest, "SELL")

        buy_ready  = all([ema_bullish_stack, broke_high, above_both, htf_allows_buy,  rsi_ok_buy,  rejection_buy])
        sell_ready = all([ema_bearish_stack, broke_low,  below_both, htf_allows_sell, rsi_ok_sell, rejection_sell])

        if buy_ready:
            signal, reason, conf_type = "BUY", "Trend mode: EMA pullback + rejection confirmed.", "trend_pullback"
        elif sell_ready:
            signal, reason, conf_type = "SELL", "Trend mode: EMA pullback + rejection confirmed.", "trend_pullback"
        else:
            reason = "Trend mode: waiting for pullback to EMA."

    # ── RANGE MODE ──────────────────────────────────────────────────────────
    elif regime == "RANGE":
        range_high = latest["range_high"]
        range_low  = latest["range_low"]
        range_size = range_high - range_low if not pd.isna(range_high) else 0

        if range_size > atr * 2:  # range must be meaningful
            at_top    = abs(latest["high"] - range_high) <= atr * RANGE_EDGE_ATR
            at_bottom = abs(latest["low"]  - range_low)  <= atr * RANGE_EDGE_ATR
            stretched_up   = latest["close"] > latest["range_mid"] + range_size * 0.3
            stretched_down = latest["close"] < latest["range_mid"] - range_size * 0.3

            rsi_overbought  = latest["rsi"] > 60
            rsi_oversold    = latest["rsi"] < 40
            rejection_buy   = is_rejection_candle(latest, "BUY")
            rejection_sell  = is_rejection_candle(latest, "SELL")

            sell_at_top = all([at_top, stretched_up,   rsi_overbought, rejection_sell, htf_allows_sell])
            buy_at_bot  = all([at_bottom, stretched_down, rsi_oversold,   rejection_buy,  htf_allows_buy])

            if buy_at_bot:
                signal, reason, conf_type = "BUY", "Range mode: price at support edge + reversal.", "range_reversion"
            elif sell_at_top:
                signal, reason, conf_type = "SELL", "Range mode: price at resistance edge + reversal.", "range_reversion"
            else:
                reason = "Range mode: waiting for edge + reversal candle."
        else:
            reason = "Range mode: range too small to trade."

    checks = {
        "regime":         regime,
        "h4_bias":        h4_bias,
        "adx":            round(float(latest["adx"]), 2) if not pd.isna(latest["adx"]) else 0,
        "rsi":            round(float(latest["rsi"]), 2),
        "atr":            round(float(atr), 6),
        "atr_spiked":     atr_spiked,
        "conf_type":      conf_type,
        "price_vs_bias":  "above" if price_above_bias else "below",
    }

    return {"signal": signal, "reason": reason, "checks": checks}


def get_h4_bias(h4_df):
    if h4_df is None or len(h4_df) < EMA_BIAS + 10:
        return "NEUTRAL"
    h4 = calculate_indicators(h4_df)
    latest = h4.iloc[-1]
    if latest["close"] > latest["ema_bias"] and latest["adx"] > ADX_RANGE:
        return "BULLISH"
    if latest["close"] < latest["ema_bias"] and latest["adx"] > ADX_RANGE:
        return "BEARISH"
    return "NEUTRAL"


def generate_signal(df):
    return analyze_setup(df)["signal"]

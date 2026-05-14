import numpy as np
import pandas as pd

# Gold (XAUUSD) Breakout Strategy
# Catches momentum breakouts during London/NY overlap (13:00-17:00 UTC)

EMA_PERIOD    = 50
RSI_PERIOD    = 14
ATR_PERIOD    = 14
RANGE_LOOKBACK = 24     # 24 x M15 = 6 hours consolidation range
MIN_BODY_RATIO = 0.55   # breakout candle must be strong
MIN_ATR_RATIO  = 0.0002 # Gold minimum volatility filter
MIN_RSI_BUY    = 52     # RSI must show bullish momentum
MAX_RSI_SELL   = 48     # RSI must show bearish momentum
MIN_SLOPE_ATR  = 0.05   # EMA slope normalized by ATR


def calculate_indicators(df):
    df = df.copy()
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df["rsi"]       = 100 - (100 / (1 + gain / loss))
    df["rsi_slope"] = df["rsi"].diff()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1)),
        ),
    )
    df["atr"]       = df["tr"].rolling(ATR_PERIOD).mean()
    df["atr_ratio"] = df["atr"] / df["close"]

    # Consolidation range — high/low over last RANGE_LOOKBACK candles (excluding current)
    df["range_high"] = df["high"].shift(1).rolling(RANGE_LOOKBACK).max()
    df["range_low"]  = df["low"].shift(1).rolling(RANGE_LOOKBACK).min()
    df["range_size"] = df["range_high"] - df["range_low"]

    # EMA slope normalized by ATR
    df["ema_slope"]     = df["ema"] - df["ema"].shift(5)
    df["ema_slope_atr"] = df["ema_slope"] / df["atr"].replace(0, np.nan)

    # Tick volume average for momentum confirmation
    df["vol_avg"] = df["tick_volume"].rolling(20).mean()

    return df


def candle_body_ratio(candle):
    r = candle["high"] - candle["low"]
    return abs(candle["close"] - candle["open"]) / r if r > 0 else 0.0


def analyze_setup(df):
    if len(df) < RANGE_LOOKBACK + ATR_PERIOD + 10:
        return {"signal": "WAIT", "reason": "Not enough data.", "checks": {}}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    atr       = latest["atr"]
    atr_ratio = latest["atr_ratio"]
    body_ratio = candle_body_ratio(latest)

    # Market too quiet — skip
    market_quiet = pd.isna(atr_ratio) or atr_ratio < MIN_ATR_RATIO
    if market_quiet:
        return {"signal": "WAIT", "reason": "Gold too quiet — no edge.", "checks": {"market_quiet": True}}

    range_high = latest["range_high"]
    range_low  = latest["range_low"]
    range_size = latest["range_size"]

    if pd.isna(range_high) or pd.isna(range_low) or range_size <= 0:
        return {"signal": "WAIT", "reason": "Range not established.", "checks": {}}

    # Breakout conditions
    bullish_breakout = latest["close"] > range_high
    bearish_breakout = latest["close"] < range_low

    # Strong candle confirmation
    strong_candle    = body_ratio >= MIN_BODY_RATIO
    bullish_candle   = latest["close"] > latest["open"]
    bearish_candle   = latest["close"] < latest["open"]

    # RSI momentum
    rsi_bullish = latest["rsi"] > MIN_RSI_BUY and latest["rsi_slope"] > 0
    rsi_bearish = latest["rsi"] < MAX_RSI_SELL and latest["rsi_slope"] < 0

    # EMA trend alignment (loose — just direction)
    ema_bullish = latest["close"] > latest["ema"] and latest["ema_slope_atr"] > MIN_SLOPE_ATR
    ema_bearish = latest["close"] < latest["ema"] and latest["ema_slope_atr"] < -MIN_SLOPE_ATR

    # Volume spike (breakout should have above-average volume)
    vol_spike = latest["tick_volume"] >= latest["vol_avg"] * 0.8

    buy_ready = all([
        bullish_breakout,
        bullish_candle,
        strong_candle,
        rsi_bullish,
        ema_bullish,
        vol_spike,
    ])
    sell_ready = all([
        bearish_breakout,
        bearish_candle,
        strong_candle,
        rsi_bearish,
        ema_bearish,
        vol_spike,
    ])

    checks = {
        "bullish_breakout": bullish_breakout,
        "bearish_breakout": bearish_breakout,
        "strong_candle":    strong_candle,
        "rsi_bullish":      rsi_bullish,
        "rsi_bearish":      rsi_bearish,
        "ema_bullish":      ema_bullish,
        "ema_bearish":      ema_bearish,
        "vol_spike":        vol_spike,
        "range_high":       round(float(range_high), 2),
        "range_low":        round(float(range_low), 2),
        "range_size":       round(float(range_size), 2),
        "rsi":              round(float(latest["rsi"]), 2),
        "atr":              round(float(atr), 2),
        "body_ratio":       round(body_ratio, 3),
    }

    if buy_ready:
        return {"signal": "BUY",  "reason": "Bullish breakout above range with momentum.", "checks": checks}
    if sell_ready:
        return {"signal": "SELL", "reason": "Bearish breakout below range with momentum.", "checks": checks}
    return {"signal": "WAIT", "reason": "No breakout confirmed.", "checks": checks}


def generate_signal(df):
    return analyze_setup(df)["signal"]

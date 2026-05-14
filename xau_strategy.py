# xau_strategy.py
# Pure signal / indicator functions -- no WebSocket, no side effects.
# The bot passes DataFrames built from state.m5_candles / state.h1_candles.
# iloc[-1] = currently-forming bar (excluded from decisions)
# iloc[-2] = last CLOSED bar  <-- everything is evaluated here

import numpy as np
import pandas as pd
from xau_config import (
    M5_EMA_PERIOD, H1_EMA_PERIOD, ATR_PERIOD,
    ZONE_ATR_MULT, SESSION_START, SESSION_END,
    CHANDELIER_TIERS, CHAND_MULT,
)


def calc_m5(df: pd.DataFrame) -> pd.DataFrame:
    """Add ema50 and atr14 to an M5 OHLC DataFrame."""
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=M5_EMA_PERIOD, adjust=False).mean()
    high_low  = df["high"] - df["low"]
    high_prev = (df["high"] - df["close"].shift(1)).abs()
    low_prev  = (df["low"]  - df["close"].shift(1)).abs()
    df["tr"]  = np.maximum(high_low, np.maximum(high_prev, low_prev))
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
    return df


def calc_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Add ema21 to an H1 OHLC DataFrame."""
    df = df.copy()
    df["ema21"] = df["close"].ewm(span=H1_EMA_PERIOD, adjust=False).mean()
    return df


def get_signal(m5: pd.DataFrame, h1: pd.DataFrame):
    """
    Evaluate last CLOSED M5 bar for an EMA Pullback entry.
    Returns (direction, atr) where direction is 'BUY', 'SELL', or 'WAIT'.

    Entry rules (all must pass):
      1. Session  : UTC hour of last closed bar in [SESSION_START, SESSION_END]
      2. No gap   : time gap from previous bar < 6 hours (skip first bar after weekend)
      3. Proximity: close within ZONE_ATR_MULT * ATR of M5 EMA50
      4. H1 trend : h1_close > h1_ema21 for BUY  |  h1_close < h1_ema21 for SELL
      5. Candle   : bullish close (close > open) for BUY  |  bearish for SELL
    """
    if len(m5) < M5_EMA_PERIOD + ATR_PERIOD + 5:
        return "WAIT", 0.0
    if len(h1) < H1_EMA_PERIOD + 5:
        return "WAIT", 0.0

    bar  = m5.iloc[-2]   # last closed bar
    prev = m5.iloc[-3]

    atr   = float(bar["atr"])
    ema50 = float(bar["ema50"])
    if pd.isna(atr) or atr <= 0 or pd.isna(ema50):
        return "WAIT", 0.0

    # 1. Session
    hour = bar["time"].hour
    if not (SESSION_START <= hour <= SESSION_END):
        return "WAIT", 0.0

    # 2. After-gap filter
    if not pd.isna(prev["time"]):
        gap_h = (bar["time"] - prev["time"]).total_seconds() / 3600.0
        if gap_h > 6.0:
            return "WAIT", 0.0

    # 3. EMA proximity
    if abs(float(bar["close"]) - ema50) > ZONE_ATR_MULT * atr:
        return "WAIT", 0.0

    # 4 + 5. H1 trend + candle direction (use last CLOSED H1 bar)
    h1_bar   = h1.iloc[-2]
    h1_close = float(h1_bar["close"])
    h1_ema21 = float(h1_bar["ema21"])
    if pd.isna(h1_ema21):
        return "WAIT", 0.0

    bull_c = float(bar["close"]) > float(bar["open"])
    bear_c = float(bar["close"]) < float(bar["open"])

    if h1_close > h1_ema21 and bull_c:
        return "BUY", atr
    if h1_close < h1_ema21 and bear_c:
        return "SELL", atr
    return "WAIT", 0.0


def chandelier_stop(entry: float, atr: float, extreme: float, direction: str) -> float:
    """
    Chandelier trailing stop price.
      direction='BUY'  : extreme = peak (max high seen)
                         stop = peak  - atr * tier_mult * CHAND_MULT
      direction='SELL' : extreme = trough (min low seen)
                         stop = trough + atr * tier_mult * CHAND_MULT
    """
    if direction == "BUY":
        peak_r = (extreme - entry) / atr if atr > 0 else 0.0
    else:
        peak_r = (entry - extreme) / atr if atr > 0 else 0.0

    tier = CHANDELIER_TIERS[0][1]
    for min_r, m in CHANDELIER_TIERS:
        if peak_r >= min_r:
            tier = m

    trail = atr * tier * CHAND_MULT
    return (extreme - trail) if direction == "BUY" else (extreme + trail)

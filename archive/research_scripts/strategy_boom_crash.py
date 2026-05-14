"""
Boom/Crash Spike Rider Strategy
- Boom (BOOM300N, BOOM500, BOOM1000): BUY only — spikes go UP
- Crash (CRASH300N, CRASH500, CRASH1000): SELL only — spikes go DOWN
- Entry: price drifted into EMA zone + momentum confirmation candle
- Exit: chandelier trailing stop rides the spike
"""
import numpy as np
import pandas as pd

EMA_FAST_PERIOD = 21
EMA_PERIOD      = 50
EMA_LONG_PERIOD = 200
RSI_PERIOD      = 14
ATR_PERIOD      = 14
SR_LOOKBACK     = 20

BOOM_SYMBOLS  = {"BOOM300N", "BOOM500", "BOOM1000"}
CRASH_SYMBOLS = {"CRASH300N", "CRASH500", "CRASH1000"}

SPIKE_WICK_RATIO  = 0.60  # spike wick must be >= 60% of candle range
SPIKE_MIN_ATR     = 0.25  # spike wick must be >= 0.25x ATR (filters dust)
COOLDOWN_CANDLES  = 2     # candles to wait after spike before re-entry
REENTRY_WINDOW    = 25    # max candles after spike to still consider entry valid
RSI_BOOM_MAX      = 68    # Boom: RSI ceiling
RSI_CRASH_MIN     = 32    # Crash: RSI floor


def get_direction(symbol: str) -> str:
    return "BUY" if "BOOM" in symbol.upper() else "SELL"


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]       = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema"]            = df["close"].ewm(span=EMA_PERIOD,      adjust=False).mean()
    df["ema_long"]       = df["close"].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs    = gain / loss
    df["rsi"]       = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi"].diff()

    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1)),
        ),
    )
    df["atr"]            = df["tr"].rolling(ATR_PERIOD).mean()
    df["recent_high"]    = df["high"].rolling(SR_LOOKBACK).max()
    df["recent_low"]     = df["low"].rolling(SR_LOOKBACK).min()
    df["ema_slope"]      = df["ema"]      - df["ema"].shift(5)
    df["ema_long_slope"] = df["ema_long"] - df["ema_long"].shift(5)
    df["atr_ratio"]      = df["atr"] / df["close"]
    return df


def _body_ratio(c) -> float:
    r = c["high"] - c["low"]
    return abs(c["close"] - c["open"]) / r if r > 0 else 0.0


def _is_engulfing(latest, prev) -> tuple:
    bull = (latest["close"] > latest["open"] and prev["close"] < prev["open"]
            and latest["close"] > prev["open"] and latest["open"] < prev["close"])
    bear = (latest["close"] < latest["open"] and prev["close"] > prev["open"]
            and latest["close"] < prev["open"] and latest["open"] > prev["close"])
    return bull, bear


def is_spike_candle(candle, atr: float, direction: str) -> bool:
    """Detect a spike candle using wick dominance — not ATR magnitude."""
    total_range = candle["high"] - candle["low"]
    if total_range <= 0 or atr <= 0:
        return False
    if direction == "BUY":
        wick = candle["high"] - max(candle["open"], candle["close"])
    else:
        wick = min(candle["open"], candle["close"]) - candle["low"]
    return (wick / total_range >= SPIKE_WICK_RATIO) and (wick >= atr * SPIKE_MIN_ATR)


def candles_since_spike(df: pd.DataFrame, direction: str, lookback: int = 30) -> int:
    """How many candles since last spike. Returns lookback+1 if none found."""
    atr = float(df.iloc[-1]["atr"]) if not pd.isna(df.iloc[-1]["atr"]) else 0
    slice_ = df.iloc[-lookback:] if len(df) >= lookback else df
    for i in range(len(slice_) - 2, -1, -1):
        if is_spike_candle(slice_.iloc[i], atr, direction):
            return len(slice_) - 2 - i
    return lookback + 1


def analyze_setup(df: pd.DataFrame, direction: str) -> dict:
    min_len = max(EMA_LONG_PERIOD, RSI_PERIOD, ATR_PERIOD, SR_LOOKBACK) + 10
    if len(df) < min_len:
        return {"signal": "WAIT", "reason": "Not enough history", "checks": {}}

    latest = df.iloc[-1]
    atr    = float(latest["atr"])
    rsi    = float(latest["rsi"])
    close  = float(latest["close"])
    ema    = float(latest["ema"])

    since_spike = candles_since_spike(df, direction)
    # Re-entry window: after spike cooldown but not too long after
    in_reentry_window = COOLDOWN_CANDLES <= since_spike <= REENTRY_WINDOW

    checks = {
        "direction":     direction,
        "rsi":           round(rsi, 1),
        "since_spike":   since_spike,
        "in_window":     in_reentry_window,
        "atr":           round(atr, 6),
        "close":         round(close, 4),
        "ema":           round(ema, 4),
    }

    if direction == "BUY":
        rsi_ok = rsi < RSI_BOOM_MAX
        ready  = in_reentry_window and rsi_ok
        reason = ("Boom: post-spike re-entry window open" if ready else
                  f"RSI too high ({rsi:.0f})" if not rsi_ok else
                  f"Not in re-entry window (since_spike={since_spike})")
    else:
        rsi_ok = rsi > RSI_CRASH_MIN
        ready  = in_reentry_window and rsi_ok
        reason = ("Crash: post-spike re-entry window open" if ready else
                  f"RSI too low ({rsi:.0f})" if not rsi_ok else
                  f"Not in re-entry window (since_spike={since_spike})")

    checks["ready"] = ready
    return {"signal": direction if ready else "WAIT", "reason": reason, "checks": checks}

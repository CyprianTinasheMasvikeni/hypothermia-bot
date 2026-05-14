# === CONFIGURATION FILE ===

# Strategy module to use for setup analysis
STRATEGY = "step_trend"

# Market to trade
PAIR = "Step Index"

# Multi-timeframe model
TREND_TIMEFRAME = "M15"
ENTRY_TIMEFRAME = "M5"

# Legacy compatibility
TIMEFRAME = ENTRY_TIMEFRAME

# Indicator settings used by some older modules
EMA_FAST = 50
EMA_SLOW = 200
RSI_PERIOD = 14

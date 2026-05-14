RISK_PER_TRADE = 0.05
TP_R = 1.5
MAX_HOLD_CANDLES = 12
MAX_TRADES_PER_DAY = 6
START_BALANCE = 100.0


def build_trade_plan(signal, entry_price, atr, balance, tp_r=TP_R):
    if signal not in {"BUY", "SELL"}:
        raise ValueError("Trade plan requires BUY or SELL signal.")
    if atr is None or atr <= 0:
        raise ValueError("ATR must be positive for trade planning.")

    risk_amount = balance * RISK_PER_TRADE
    size = risk_amount / atr

    if signal == "BUY":
        sl = entry_price - atr
        tp = entry_price + (atr * tp_r)
    else:
        sl = entry_price + atr
        tp = entry_price - (atr * tp_r)

    return {
        "type": signal,
        "entry": entry_price,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "risk_amount": risk_amount,
        "size": size,
        "tp_r": tp_r,
    }


def close_trade(trade, exit_price, exit_time, result, balance):
    direction = 1 if trade["type"] == "BUY" else -1
    pnl = (exit_price - trade["entry"]) * direction * trade["size"]
    final_balance = balance + pnl

    closed_trade = trade.copy()
    closed_trade["exit"] = exit_price
    closed_trade["exit_time"] = exit_time
    closed_trade["result"] = result
    closed_trade["pnl"] = pnl
    closed_trade["r_multiple"] = pnl / trade["risk_amount"] if trade["risk_amount"] else 0
    closed_trade["final_balance"] = final_balance
    closed_trade["closed"] = True
    return closed_trade

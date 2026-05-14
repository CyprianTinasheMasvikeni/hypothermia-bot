import MetaTrader5 as mt5
from config import RISK_PER_TRADE, TP_MULTIPLIER
from journal import log_trade
from notifier import send_telegram_message


def calculate_lot(symbol, stop_loss_points):
    account_info = mt5.account_info()
    balance = account_info.balance
    risk_amount = balance * RISK_PER_TRADE

    # Tick value and size assumptions
    tick_value = 1.0
    tick_size = 1.0
    point_value = tick_value / tick_size

    lot = risk_amount / (stop_loss_points * point_value)
    return round(lot, 2)


def place_trade(symbol, signal, sl_points=100, magic=10001):
    symbol_info = mt5.symbol_info(symbol)

    if not symbol_info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if signal == "BUY" else tick.bid

    sl = price - sl_points if signal == "BUY" else price + sl_points
    tp = price + (sl_points * TP_MULTIPLIER) if signal == "BUY" else price - (sl_points * TP_MULTIPLIER)

    lot = calculate_lot(symbol, stop_loss_points=sl_points)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "deviation": 20,
        "comment": "Neural Sniper Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade failed for {symbol}: {result.retcode}")
        return None
    else:
        print(f"✅ {signal} trade placed on {symbol} @ {price} | SL: {sl} | TP: {tp}")
        log_trade(symbol, signal, price, sl, tp, lot, signal)

        # 🚨 Send Telegram alert
        send_telegram_message(
            f"✅ {signal} trade placed on {symbol}\nEntry: {price:.2f}\nSL: {sl:.2f} | TP: {tp:.2f} | Lot: {lot}"
        )

        return result

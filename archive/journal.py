import csv
from datetime import datetime

# Path to your journal file
JOURNAL_FILE = "trade_journal.csv"


def log_trade(symbol, direction, price, sl, tp, lot, signal):
    now = datetime.now()
    data = [
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M:%S"),
        symbol,
        direction,
        price,
        sl,
        tp,
        lot,
        signal
    ]

    # Add headers if file is empty/new
    try:
        with open(JOURNAL_FILE, "r") as file:
            pass
    except FileNotFoundError:
        with open(JOURNAL_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Date", "Time", "Symbol", "Direction", "Entry Price",
                "SL", "TP", "Lot Size", "Signal Type"
            ])

    # Append new row
    with open(JOURNAL_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

    print(f"📓 Trade logged to journal: {symbol} | {direction} | {price}")

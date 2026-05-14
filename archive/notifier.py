import requests

# ✅ Your real bot token and chat ID
BOT_TOKEN = "7594639413:AAF80KyzltOD2fc_RSG-nCmq6v-6Fyjc4X0"
CHAT_ID = "6646264971"  # This is your personal chat ID

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("📲 Telegram alert sent.")
        else:
            print(f"❌ Failed to send alert: {response.text}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")


# ✅ TEST MODE: Send a message when you run this file directly
if __name__ == "__main__":
    send_telegram_message("🚀 Test alert from SniperTraderCypBot is working!")

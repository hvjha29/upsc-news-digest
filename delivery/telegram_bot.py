# delivery/telegram_bot.py
import os
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import CommandHandler, Updater

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TOKEN:
    print("Set TELEGRAM_TOKEN in .env to test the bot.")
    raise SystemExit(1)

def start(update, context):
    update.message.reply_text("Hi — UPSC news digest bot (alpha). Use /digest to get today's sample.")

def digest(update, context):
    # placeholder: in prod you would fetch today's summaries from DB
    sample = "Today's headlines (sample):\n1) Topic A — short bullet\n2) Topic B — short bullet"
    update.message.reply_text(sample)

def main():
    updater = Updater(TOKEN)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("digest", digest))
    print("Starting Telegram bot...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
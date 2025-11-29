# delivery/send_daily_digest.py
"""
Run daily: process sources, index & summarize (using existing pipeline),
then fetch recent summaries and send a compact digest to Telegram.

Usage (local):
  export TELEGRAM_TOKEN=...
  export TELEGRAM_CHAT_ID=...   # optional: if you want to send to a specific chat/channel
  python delivery/send_daily_digest.py

In GitHub Actions, set TELEGRAM_TOKEN & OPENAI_API_KEY as secrets.
"""
import os
import time
import json
import logging
from index.chroma_client import get_client, get_or_create_collection
from run_pipeline import process_and_index
from dotenv import load_dotenv
from telegram import Bot

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # can be your bot DM id or channel id (e.g., @channelname)
SOURCES_FILE = os.path.join(os.path.dirname(__file__), "sources.txt")
MAX_SUMMARIES_TO_SEND = int(os.getenv("MAX_SUMMARIES_TO_SEND", "5"))

if not TELEGRAM_TOKEN:
    logging.error("TELEGRAM_TOKEN not set. Exiting.")
    raise SystemExit(1)

bot = Bot(token=TELEGRAM_TOKEN)

def load_sources(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
    return lines

def collect_today_summaries(limit=10):
    """
    Read summaries collection from Chroma and return the most recent documents.
    (Chroma's local persistence doesn't have a created_at by default; we assume insertion order).
    """
    client = get_client()
    summaries_col = get_or_create_collection(client, "summaries")
    try:
        res = summaries_col.get()
    except Exception as e:
        logging.warning("Could not fetch summaries from Chroma: %s", e)
        return []

    documents = res.get("documents", []) or []
    metadatas = res.get("metadatas", []) or []
    ids = res.get("ids", []) or []
    items = list(zip(ids, documents, metadatas))
    if not items:
        logging.info("No summaries found in Chroma.")
        return []
    # take the last 'limit' items (assume last appended are latest)
    last_items = items[-limit:]
    # return reversed so newest first
    return list(reversed(last_items))

def assemble_message(items):
    """
    Build a compact digest message (plain text). Keep within Telegram message limits.
    """
    header = f"🗞️ UPSC News Digest — {time.strftime('%Y-%m-%d')}\n\n"
    parts = [header]
    for _id, doc, md in items:
        title = md.get("title") or "Untitled"
        source = md.get("source") or ""
        # doc is expected to be JSON string from LLM; try to parse and format
        brief = ""
        try:
            parsed = json.loads(doc)
            gist = parsed.get("gist") or (parsed.get("gist", "") if isinstance(parsed, dict) else "")
            facts = parsed.get("facts", [])
            # Make a short snippet
            brief = (gist if isinstance(gist, str) else json.dumps(gist))[:400]
            # include 1–2 bullets from facts
            bullets = ""
            if isinstance(facts, list) and len(facts) > 0:
                bullets = "\n• " + "\n• ".join(facts[:2])
        except Exception:
            # fallback: doc is raw text
            brief = (doc[:400] + ("..." if len(doc) > 400 else ""))

        parts.append(f"🔹 *{title}*\nSource: {source}\n{brief}{bullets}\n\n")
    footer = "To get detailed summaries, visit the channel or use /digest\n— Auto-generated"
    parts.append(footer)
    # Telegram supports markdown; we'll send as MarkdownV2 safe text later
    return "\n".join(parts)

def send_message(text):
    """
    Send the message to the chat. If TELEGRAM_CHAT_ID is not set, send to the bot owner (use getUpdates to find chat id).
    """
    try:
        if TELEGRAM_CHAT_ID:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        else:
            # Send to the bot owner — best practice: use a configured chat id in production
            # We'll attempt to send to the bot's own chat with bot.get_me() - this won't deliver to a human
            me = bot.get_me()
            logging.info("Sending digest to bot (not a user). Set TELEGRAM_CHAT_ID to send to a channel or chat.")
            bot.send_message(chat_id=me.id, text=text)
    except Exception as e:
        logging.exception("Failed to send Telegram message: %s", e)

def main():
    sources = load_sources(SOURCES_FILE)
    if not sources:
        logging.warning("No sources found in %s — add URLs to scrape.", SOURCES_FILE)
    else:
        logging.info("Processing %d sources...", len(sources))
        for url in sources:
            try:
                logging.info("Processing: %s", url)
                # process_and_index comes from run_pipeline; it indexes and creates a summary in Chroma
                process_and_index(url)
                # small sleep to avoid hammering sources / rate limits
                time.sleep(2)
            except Exception as e:
                logging.exception("Error while processing %s : %s", url, e)

    # collect the latest summaries and send
    items = collect_today_summaries(limit=MAX_SUMMARIES_TO_SEND)
    if not items:
        logging.info("No summaries to send. Exiting.")
        return
    message = assemble_message(items)
    # Telegram messages have a 4096 char limit; truncate if needed
    if len(message) > 3800:
        message = message[:3800] + "\n\n...[truncated]"
    send_message(message)
    logging.info("Digest sent.")

if __name__ == "__main__":
    main()
# delivery/telegram_bot.py
"""
Telegram Bot for UPSC News Digest
Sends daily summaries to the configured channel/chat.

Usage:
    python delivery/telegram_bot.py
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(override=True)  # override=True to ensure .env takes precedence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Telegram configuration
TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Output directory for summaries
OUTPUT_DIR = Path(__file__).parent.parent / "prompting" / "output"


def get_latest_summary_file():
    """Find the most recent summary file."""
    summary_files = list(OUTPUT_DIR.glob("yes_article_summaries_*.txt"))
    if not summary_files:
        return None
    return max(summary_files, key=lambda f: f.stat().st_mtime)


def get_latest_titles_file():
    """Find the most recent titles file."""
    title_files = list(OUTPUT_DIR.glob("yes_article_titles_*.txt"))
    if not title_files:
        return None
    return max(title_files, key=lambda f: f.stat().st_mtime)


def parse_summaries(file_path):
    """Parse summaries from the output file into individual messages."""
    summaries = []
    current_summary = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_summary = False
    for line in lines:
        # Skip header
        if line.startswith("UPSC-RELEVANT ARTICLE SUMMARIES") or line.startswith("="*20):
            continue
        
        # New article starts with a number
        if line.strip() and line.strip()[0].isdigit() and ". " in line[:5]:
            if current_summary:
                summaries.append('\n'.join(current_summary))
            current_summary = [line.rstrip()]
            in_summary = True
        elif in_summary:
            if line.strip() == "-" * 80:
                # End of this summary
                if current_summary:
                    summaries.append('\n'.join(current_summary))
                current_summary = []
                in_summary = False
            else:
                current_summary.append(line.rstrip())
    
    # Don't forget the last one
    if current_summary:
        summaries.append('\n'.join(current_summary))
    
    return summaries


def format_summary_for_telegram(summary_text):
    """Format a single summary for Telegram (handle markdown escaping)."""
    # Remove the article number prefix for cleaner display
    lines = summary_text.strip().split('\n')
    if not lines:
        return None
    
    # Extract title from first line
    first_line = lines[0]
    if ". " in first_line[:5]:
        first_line = first_line.split(". ", 1)[1] if ". " in first_line else first_line
    
    # Skip source/URL lines, get to the summary
    formatted_lines = []
    skip_metadata = True
    for line in lines[1:]:
        if line.strip().startswith("Summary:"):
            skip_metadata = False
            continue
        if skip_metadata and (line.strip().startswith("Source:") or line.strip().startswith("URL:")):
            continue
        if not skip_metadata:
            formatted_lines.append(line)
    
    if not formatted_lines:
        return None
    
    return '\n'.join(formatted_lines).strip()


async def send_to_telegram(token, chat_id, messages, delay_between=2):
    """Send messages to Telegram channel using httpx (async)."""
    import httpx
    
    base_url = f"https://api.telegram.org/bot{token}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # First send a header message
        header = f"🗞️ *UPSC Daily Digest — {datetime.now().strftime('%d %B %Y')}*\n\n" \
                 f"Today's UPSC-relevant articles from The Hindu & Indian Express:\n" \
                 f"━━━━━━━━━━━━━━━━━━━━"
        
        try:
            response = await client.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": header,
                    "parse_mode": "Markdown"
                }
            )
            response.raise_for_status()
            logger.info("Sent header message")
        except Exception as e:
            logger.error(f"Failed to send header: {e}")
        
        await asyncio.sleep(delay_between)
        
        # Send each summary
        sent_count = 0
        for i, msg in enumerate(messages, 1):
            if not msg or len(msg.strip()) < 50:
                continue
            
            # Telegram has a 4096 char limit
            if len(msg) > 4000:
                msg = msg[:4000] + "\n\n...[truncated]"
            
            try:
                response = await client.post(
                    f"{base_url}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": msg,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    }
                )
                if response.status_code == 200:
                    sent_count += 1
                    logger.info(f"Sent summary {i}/{len(messages)}")
                else:
                    # Try without markdown if it fails
                    response = await client.post(
                        f"{base_url}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": msg,
                            "disable_web_page_preview": True
                        }
                    )
                    if response.status_code == 200:
                        sent_count += 1
                        logger.info(f"Sent summary {i}/{len(messages)} (plain text)")
                    else:
                        logger.error(f"Failed to send summary {i}: {response.text}")
            except Exception as e:
                logger.error(f"Error sending summary {i}: {e}")
            
            await asyncio.sleep(delay_between)
        
        # Send footer
        footer = f"\n━━━━━━━━━━━━━━━━━━━━\n" \
                 f"📚 Total summaries: {sent_count}\n" \
                 f"🔗 Sources: The Hindu, Indian Express\n" \
                 f"📅 {datetime.now().strftime('%d %B %Y')}\n\n" \
                 f"#UPSC #CurrentAffairs #DailyDigest"
        
        try:
            await client.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": footer,
                    "parse_mode": "Markdown"
                }
            )
            logger.info("Sent footer message")
        except Exception as e:
            logger.error(f"Failed to send footer: {e}")
        
        return sent_count


async def send_digest_to_channel():
    """Main function to send today's digest to the Telegram channel."""
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN not set in .env")
        return False
    
    if not CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID not set in .env")
        return False
    
    # Find the latest summary file
    summary_file = get_latest_summary_file()
    if not summary_file:
        logger.error(f"No summary files found in {OUTPUT_DIR}")
        return False
    
    logger.info(f"Using summary file: {summary_file}")
    
    # Parse summaries
    summaries = parse_summaries(summary_file)
    if not summaries:
        logger.error("No summaries found in file")
        return False
    
    logger.info(f"Found {len(summaries)} summaries")
    
    # Format for Telegram
    formatted_messages = []
    for summary in summaries:
        formatted = format_summary_for_telegram(summary)
        if formatted:
            formatted_messages.append(formatted)
    
    logger.info(f"Formatted {len(formatted_messages)} messages for Telegram")
    
    # Send to Telegram
    sent = await send_to_telegram(TOKEN, CHAT_ID, formatted_messages)
    logger.info(f"Successfully sent {sent} summaries to Telegram channel {CHAT_ID}")
    
    return True


def main():
    """Entry point."""
    asyncio.run(send_digest_to_channel())


if __name__ == "__main__":
    main()
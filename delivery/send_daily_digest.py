# delivery/send_daily_digest.py
"""
Complete UPSC News Digest Pipeline:
1. Scrape today's articles from The Hindu and Indian Express
2. Classify articles for UPSC relevance
3. Generate summaries for relevant articles
4. Send summaries to Telegram channel

Usage:
  python delivery/send_daily_digest.py

Environment variables (set in .env):
  TELEGRAM_TOKEN - Telegram bot token
  TELEGRAM_CHAT_ID - Channel/chat ID (e.g., @daily_upsc_bot)
"""
import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)  # override=True to ensure .env takes precedence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
OUTPUT_DIR = PROJECT_ROOT / "prompting" / "output"
MAX_SUMMARIES_PER_MESSAGE = 1  # Send each summary as a separate message
MESSAGE_DELAY = 2  # Seconds between messages to avoid rate limiting


def scrape_todays_articles():
    """Scrape today's articles from Hindu and Indian Express."""
    from prompting.newspaper_scraper import IndianExpressScraper, TheHinduScraper
    
    all_articles = []
    
    # Scrape Indian Express
    logger.info("Scraping Indian Express...")
    try:
        ie_scraper = IndianExpressScraper()
        ie_articles = ie_scraper.scrape_todays_paper()
        logger.info(f"Scraped {len(ie_articles)} articles from Indian Express")
        all_articles.extend(ie_articles)
    except Exception as e:
        logger.error(f"Error scraping Indian Express: {e}")
    
    # Scrape The Hindu
    logger.info("Scraping The Hindu...")
    try:
        hindu_scraper = TheHinduScraper()
        hindu_articles = hindu_scraper.scrape_todays_paper()
        logger.info(f"Scraped {len(hindu_articles)} articles from The Hindu")
        all_articles.extend(hindu_articles)
    except Exception as e:
        logger.error(f"Error scraping The Hindu: {e}")
    
    return all_articles


def classify_articles(articles):
    """Classify articles for UPSC relevance."""
    from prompting.classifier import UPSCClassifier
    
    classifier = UPSCClassifier()
    yes_articles = []
    
    for i, article in enumerate(articles, 1):
        if i % 50 == 0:
            logger.info(f"Classified {i}/{len(articles)} articles")
        
        try:
            result = classifier.classify(article.get('title', ''), article.get('content', ''))
            if result and result.get('classification') == 'YES':
                yes_articles.append(article)
        except Exception as e:
            logger.error(f"Error classifying article: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return yes_articles


def generate_summaries(articles):
    """Generate summaries for UPSC-relevant articles."""
    from prompting.summarizer import UPSCSummarizer
    
    summarizer = UPSCSummarizer()
    summaries = []
    
    for i, article in enumerate(articles, 1):
        logger.info(f"Summarizing {i}/{len(articles)}: {article.get('title', '')[:50]}...")
        
        content = article.get('content', '')
        if len(content) < 200:
            logger.warning(f"Content too short for: {article.get('title', '')}")
            continue
        
        try:
            summary = summarizer.summarize(
                article.get('title', ''),
                content,
                article.get('source', ''),
                article.get('section', '')
            )
            if summary:
                summaries.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'section': article.get('section', ''),
                    'summary': summary
                })
        except Exception as e:
            logger.error(f"Error summarizing article: {e}")
        
        time.sleep(1)  # Rate limiting
    
    return summaries


def save_results(yes_articles, summaries):
    """Save results to output files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save YES article titles
    titles_file = OUTPUT_DIR / f"yes_article_titles_{timestamp}.txt"
    with open(titles_file, 'w', encoding='utf-8') as f:
        f.write("UPSC-RELEVANT ARTICLE TITLES\n")
        f.write("=" * 50 + "\n\n")
        for i, article in enumerate(yes_articles, 1):
            f.write(f"{i}. {article.get('title', '')}\n")
            f.write(f"   Source: {article.get('source', '')} | Section: {article.get('section', '')}\n")
            f.write(f"   URL: {article.get('url', '')}\n\n")
    
    logger.info(f"Saved titles to: {titles_file}")
    
    # Save summaries
    summaries_file = OUTPUT_DIR / f"yes_article_summaries_{timestamp}.txt"
    with open(summaries_file, 'w', encoding='utf-8') as f:
        f.write("UPSC-RELEVANT ARTICLE SUMMARIES\n")
        f.write("=" * 50 + "\n\n")
        for i, item in enumerate(summaries, 1):
            f.write(f"{i}. {item['title']}\n")
            f.write(f"   Source: {item['source']} | Section: {item['section']}\n")
            f.write(f"   URL: {item['url']}\n")
            f.write(f"   Summary:\n{item['summary']}\n")
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Saved summaries to: {summaries_file}")
    
    return titles_file, summaries_file


async def send_to_telegram(summaries):
    """Send summaries to Telegram channel."""
    import httpx
    
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set")
        return 0
    
    if not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID not set")
        return 0
    
    base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
    sent_count = 0
    
    # First verify the bot can access the chat
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test bot connectivity
        try:
            test_response = await client.get(f"{base_url}/getMe")
            if test_response.status_code == 200:
                bot_info = test_response.json()
                logger.info(f"Bot verified: @{bot_info.get('result', {}).get('username', 'unknown')}")
            else:
                logger.error(f"Bot verification failed: {test_response.text}")
                return 0
        except Exception as e:
            logger.error(f"Failed to verify bot: {e}")
            return 0
        
        # Send header
        header = (
            f"🗞️ *UPSC Daily Digest*\n"
            f"📅 {datetime.now().strftime('%d %B %Y')}\n\n"
            f"Today's UPSC-relevant articles from The Hindu & Indian Express\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        
        try:
            await client.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": header,
                    "parse_mode": "Markdown"
                }
            )
            logger.info("Sent header message")
        except Exception as e:
            logger.error(f"Failed to send header: {e}")
        
        await asyncio.sleep(MESSAGE_DELAY)
        
        # Send each summary
        for i, item in enumerate(summaries, 1):
            summary_text = item.get('summary', '')
            if not summary_text or len(summary_text) < 50:
                continue
            
            # Truncate if too long
            if len(summary_text) > 4000:
                summary_text = summary_text[:4000] + "\n\n...[truncated]"
            
            try:
                # Try with Markdown first
                response = await client.post(
                    f"{base_url}/sendMessage",
                    json={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": summary_text,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    }
                )
                
                if response.status_code == 200:
                    sent_count += 1
                    logger.info(f"Sent summary {i}/{len(summaries)}")
                else:
                    # Fallback to plain text
                    response = await client.post(
                        f"{base_url}/sendMessage",
                        json={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "text": summary_text,
                            "disable_web_page_preview": True
                        }
                    )
                    if response.status_code == 200:
                        sent_count += 1
                        logger.info(f"Sent summary {i}/{len(summaries)} (plain text)")
                    else:
                        logger.error(f"Failed to send summary {i}: {response.text}")
            except Exception as e:
                logger.error(f"Error sending summary {i}: {e}")
            
            await asyncio.sleep(MESSAGE_DELAY)
        
        # Send footer
        footer = (
            f"\n━━━━━━━━━━━━━━━━━━━━\n"
            f"📚 *Total summaries: {sent_count}*\n"
            f"🔗 Sources: The Hindu, Indian Express\n\n"
            f"#UPSC #CurrentAffairs #DailyDigest #Mains"
        )
        
        try:
            await client.post(
                f"{base_url}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": footer,
                    "parse_mode": "Markdown"
                }
            )
            logger.info("Sent footer message")
        except Exception as e:
            logger.error(f"Failed to send footer: {e}")
    
    return sent_count


async def send_existing_summaries():
    """Send already-generated summaries to Telegram."""
    # Find the latest summary file
    summary_files = list(OUTPUT_DIR.glob("yes_article_summaries_*.txt"))
    if not summary_files:
        logger.error("No summary files found")
        return 0
    
    latest_file = max(summary_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using summary file: {latest_file}")
    
    # Parse summaries from file
    summaries = []
    with open(latest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator
    parts = content.split('-' * 80)
    for part in parts:
        if 'Summary:' in part:
            # Extract the summary portion
            summary_start = part.find('Summary:')
            if summary_start != -1:
                summary_text = part[summary_start + 8:].strip()
                if summary_text:
                    summaries.append({'summary': summary_text})
    
    logger.info(f"Found {len(summaries)} summaries to send")
    return await send_to_telegram(summaries)


async def run_full_pipeline():
    """Run the complete pipeline: scrape, classify, summarize, and send."""
    logger.info("="*60)
    logger.info("UPSC NEWS DIGEST PIPELINE")
    logger.info("="*60)
    
    # Step 1: Scrape articles
    logger.info("\n[Step 1/4] Scraping today's articles...")
    articles = scrape_todays_articles()
    logger.info(f"Total articles scraped: {len(articles)}")
    
    if not articles:
        logger.error("No articles scraped. Exiting.")
        return
    
    # Step 2: Classify articles
    logger.info("\n[Step 2/4] Classifying articles for UPSC relevance...")
    yes_articles = classify_articles(articles)
    logger.info(f"UPSC-relevant articles: {len(yes_articles)}")
    
    if not yes_articles:
        logger.warning("No UPSC-relevant articles found.")
        return
    
    # Step 3: Generate summaries
    logger.info("\n[Step 3/4] Generating summaries...")
    summaries = generate_summaries(yes_articles)
    logger.info(f"Summaries generated: {len(summaries)}")
    
    # Save results
    save_results(yes_articles, summaries)
    
    # Step 4: Send to Telegram
    logger.info("\n[Step 4/4] Sending to Telegram...")
    sent_count = await send_to_telegram(summaries)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Articles scraped: {len(articles)}")
    logger.info(f"UPSC-relevant: {len(yes_articles)}")
    logger.info(f"Summaries generated: {len(summaries)}")
    logger.info(f"Sent to Telegram: {sent_count}")
    logger.info("="*60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UPSC News Digest Pipeline")
    parser.add_argument(
        "--send-only",
        action="store_true",
        help="Only send existing summaries to Telegram (skip scraping/classification)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (scrape, classify, summarize, send)"
    )
    
    args = parser.parse_args()
    
    if args.send_only:
        logger.info("Sending existing summaries to Telegram...")
        asyncio.run(send_existing_summaries())
    elif args.full:
        asyncio.run(run_full_pipeline())
    else:
        # Default: send existing summaries
        logger.info("Use --full for complete pipeline, or --send-only to send existing summaries")
        logger.info("Defaulting to --send-only mode...")
        asyncio.run(send_existing_summaries())


if __name__ == "__main__":
    main()
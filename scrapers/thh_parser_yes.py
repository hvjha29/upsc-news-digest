"""
Scrape articles from The Hindu sections (YES URLs) and write to a single CSV with schema:
id,year,paper,question_no,question_text,word_limit,marks,topic_hint,source_url
"""

import re
import time
import csv
import sys
import signal
import json
import os
import logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('thh_parser_yes.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
USER_AGENT = "Mozilla/5.0 (compatible; THHParserYes/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}
OUT_CSV = "data/thh_articles_yes.csv"  # Different output file
CHECKPOINT_FILE = "data/thh_checkpoint_yes.json"  # Different checkpoint file
SEEN_URLS_FILE = "data/thh_seen_urls_yes.json"  # Different seen URLs file
REQUEST_DELAY = 2.0   # seconds between requests (polite crawling)
MAX_RETRIES = 3       # maximum number of retries for failed requests
REQUEST_TIMEOUT = 20  # timeout for requests in seconds
MAX_ARTICLES_PER_SECTION = 100  # limit articles per section to 100
CSV_WRITE_INTERVAL = 50   # write partial CSV every N articles

# Regex patterns to identify article links on The Hindu
ARTICLE_RE = re.compile(r"https?://[^/]+/article.*")  # general article pattern
SPECIFIC_ARTICLE_RE = re.compile(r"https?://[^/]+/.*?/article.*")  # more specific pattern matching /section/article/...

# Global variable to track if shutdown was requested
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...", file=sys.stderr)
    shutdown_requested = True

def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination

def fetch(url, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    """Fetch a URL with retries and error handling."""
    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching URL: {url}")
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            logger.debug(f"Successfully fetched URL: {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"fetch error {url} (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                time.sleep(REQUEST_DELAY * 2)  # Wait longer between retries
            else:
                logger.error(f"Failed to fetch {url} after {retries} retries")
                return None
    return None

def parse_section_page(html, base_url):
    """Return set of article URLs found on a section page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Collect article links from the page
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absolute_url = urljoin(base_url, href)

        # Match The Hindu article URLs (they typically follow /section/article/ pattern)
        if SPECIFIC_ARTICLE_RE.match(absolute_url):
            links.add(absolute_url)
        elif ARTICLE_RE.match(absolute_url) and "article" in absolute_url.lower():
            links.add(absolute_url)

    logger.debug(f"Parsed section page and found {len(links)} potential article links")
    return links

def parse_article(html, url):
    """Extract title, publish_date (ISO), tags list, and content text from an article page."""
    logger.debug(f"Parsing article: {url}")
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        # Try alternative title selectors
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        if not title:
            # Remove " - The Hindu" suffix if present
            title = re.sub(r'\s*-\s*The Hindu$', '', title)
            logger.warning(f"No title found for article: {url}")

    # Publish date: look for time tag or meta properties
    date = ""
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        date = time_tag.get("datetime").strip()
    else:
        # meta property for published time
        meta_pub = soup.find("meta", {"property":"article:published_time"})
        if meta_pub and meta_pub.get("content"):
            date = meta_pub["content"].strip()
        else:
            # Alternative: look for meta name="publishdate" or similar
            meta_pub_alt = soup.find("meta", {"name": re.compile(r"publish", re.I)})
            if meta_pub_alt and meta_pub_alt.get("content"):
                date = meta_pub_alt["content"].strip()
            else:
                # Fallback: look for classes containing "date" or "time"
                date_el = soup.find(class_=re.compile(r"date|time|publish", re.I))
                if date_el:
                    date_str = date_el.get_text(strip=True)
                    # Try to extract date parts - The Hindu typically has dates in readable formats
                    date_match = re.search(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+\w+\s+\d{4})\b', date_str)
                    if date_match:
                        date = date_match.group(1)

    if not date:
        logger.warning(f"No publish date found for article: {url}")

    # Tags: look for rel="tag" or class containing "tags"
    tags = []
    for tag_a in soup.find_all("a", rel="tag"):
        tags.append(tag_a.get_text(strip=True))
    if not tags:
        # Look for tag containers
        tag_containers = soup.find_all(class_=re.compile(r"tag|category", re.I))
        for container in tag_containers:
            for tag_a in container.find_all("a"):
                tag_text = tag_a.get_text(strip=True)
                if tag_text and tag_text not in tags:
                    tags.append(tag_text)

    # Article body text extraction
    body_text = ""
    # The Hindu typically uses article body selectors like itemprop="articleBody"
    body = soup.find(attrs={"itemprop": "articleBody"})
    if not body:
        # Alternative selectors for The Hindu
        body = soup.find("div", class_=re.compile(r"article|content|body|article-content", re.I))
        if not body:
            body = soup.find("div", attrs={"id": re.compile(r"content|article", re.I)})

    if body:
        # Extract text but try to preserve readability
        paragraphs = body.find_all("p")
        if paragraphs:
            body_text = " ".join([p.get_text(strip=True) for p in paragraphs])
        else:
            body_text = body.get_text(separator=" ", strip=True)

    # Also try to find article summary from meta description
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc and meta_desc.get("content"):
        summary = meta_desc["content"].strip()
        if len(summary) > len(body_text):
            body_text = summary

    logger.debug(f"Successfully parsed article: {url} - Title: {title[:50]}...")

    return {
        "title": title,
        "date": date,
        "tags": tags,
        "body_text": body_text
    }

def save_checkpoint(articles, seen_urls, current_section_idx):
    """Save progress to checkpoint file."""
    logger.info(f"Saving checkpoint with {len(articles)} articles and {len(seen_urls)} seen URLs")
    checkpoint_data = {
        "articles": articles,
        "seen_urls": list(seen_urls),
        "current_section_idx": current_section_idx
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info("Checkpoint saved successfully")

def load_checkpoint():
    """Load progress from checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        logger.info("Loading checkpoint data...")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        logger.info(f"Loaded checkpoint data with {len(checkpoint_data.get('articles', []))} articles")
        return (
            checkpoint_data.get("articles", []),
            set(checkpoint_data.get("seen_urls", [])),
            checkpoint_data.get("current_section_idx", 0)
        )
    logger.info("No checkpoint file found, starting from scratch")
    return [], set(), 0

def save_seen_urls(seen_urls):
    """Save seen URLs to file."""
    logger.info(f"Saving {len(seen_urls)} seen URLs to file")
    with open(SEEN_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_urls), f, indent=2)
    logger.info("Seen URLs saved successfully")

def load_seen_urls():
    """Load seen URLs from file if it exists."""
    if os.path.exists(SEEN_URLS_FILE):
        logger.info("Loading seen URLs from file...")
        with open(SEEN_URLS_FILE, "r", encoding="utf-8") as f:
            urls = json.load(f)
        logger.info(f"Loaded {len(urls)} seen URLs from file")
        return set(urls)
    logger.info("No seen URLs file found")
    return set()

def write_articles_to_csv(articles, filename=OUT_CSV, append=False):
    """Write articles to CSV file."""
    logger.info(f"Writing {len(articles)} articles to CSV file: {filename}")
    write_header = not os.path.exists(filename) or not append

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id","year","paper","question_no","question_text","word_limit","marks","topic_hint","source_url"])

        for article in articles:
            # Extract year from date if available
            year = ""
            if article["date"]:
                year_match = re.search(r"(\d{4})", article["date"])
                if year_match:
                    year = year_match.group(1)

            # Join tags with semicolons
            tags_str = ";".join(article["tags"]) if article["tags"] else ""

            writer.writerow([
                article["id"],
                year,  # year extracted from date
                "The Hindu",  # paper name
                article["id"],  # question_no (using same as id)
                article["title"] + " " + article["body_text"][:2000],  # question_text (title + partial content)
                "",  # word_limit (not applicable for news)
                "",  # marks (not applicable for news)
                tags_str,  # topic_hint (from tags)
                article["source_url"]  # source_url
            ])

    logger.info(f"Successfully wrote {len(articles)} articles to CSV")

def scrape_sections(urls_file="data/YES_url.md", max_articles_per_section=MAX_ARTICLES_PER_SECTION):  # Changed default filename
    global shutdown_requested
    setup_signal_handlers()

    # Load existing progress and seen URLs
    articles, seen_urls, start_section_idx = load_checkpoint()
    initial_seen_urls = load_seen_urls()
    seen_urls.update(initial_seen_urls)

    # Get the list of URLs to scrape from the file
    with open(urls_file, "r") as f:
        all_urls = [line.strip() for line in f if line.strip().startswith("https://www.thehindu.com")]

    starting_articles_count = len(articles)
    logger.info(f"Starting scrape from section index {start_section_idx}. Found {len(seen_urls)} previously seen URLs and {len(articles)} previously scraped articles.")

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Track IDs sequentially
    next_article_id = len(articles) + 1

    # Process each section URL
    for idx, section_url in enumerate(all_urls):
        if shutdown_requested:
            logger.info("Shutdown requested. Saving progress before exit...")
            save_checkpoint(articles, seen_urls, idx)
            save_seen_urls(seen_urls)
            break

        # Skip sections we've already processed if resuming from checkpoint
        if idx < start_section_idx:
            continue

        logger.info(f"Processing section {idx + 1}/{len(all_urls)}: {section_url}")

        # Fetch the section page
        html = fetch(section_url)
        if not html:
            logger.error(f"Unable to fetch section page {section_url} — skipping.")
            continue

        # Parse links from this section page
        links = parse_section_page(html, section_url)
        new_links = [link for link in links if link not in seen_urls]

        logger.info(f"Found {len(links)} total links from section page; {len(new_links)} new.")

        # Limit the number of articles per section
        links_to_process = new_links[:max_articles_per_section]
        logger.info(f"Processing {len(links_to_process)} articles from this section (limited to {max_articles_per_section})")

        # Process each article link
        for link_idx, article_url in enumerate(tqdm(links_to_process, desc=f"Articles from {section_url.split('/')[-2] or section_url.split('/')[-1].split('.')[0]}")):
            if shutdown_requested:
                logger.info("Shutdown requested during article processing. Saving progress before exit...")
                save_checkpoint(articles, seen_urls, idx)
                save_seen_urls(seen_urls)
                return  # Exit early to handle shutdown

            # Fetch the article page
            article_html = fetch(article_url)
            if not article_html:
                logger.warning(f"Failed to fetch article: {article_url}")
                continue

            # Parse the article content
            article_data = parse_article(article_html, article_url)

            if article_data["title"] and len(article_data["body_text"]) > 50:  # Only save if it's actually an article with content
                # Add to seen URLs
                seen_urls.add(article_url)

                # Create article record following the required schema
                article_record = {
                    "id": next_article_id,
                    "date": article_data["date"],
                    "tags": article_data["tags"],
                    "title": article_data["title"],
                    "body_text": article_data["body_text"],
                    "source_url": article_url
                }

                articles.append(article_record)
                next_article_id += 1

                # Write partial CSV every N articles to prevent data loss
                if len(articles) % CSV_WRITE_INTERVAL == 0:
                    logger.info(f"Writing partial CSV with {CSV_WRITE_INTERVAL} articles...")
                    start_idx = len(articles) - CSV_WRITE_INTERVAL
                    write_articles_to_csv(articles[start_idx:], OUT_CSV, append=True)

                    # Save checkpoint every N articles
                    if len(articles) % (CSV_WRITE_INTERVAL * 2) == 0:
                        logger.info(f"Saving checkpoint at article {len(articles)}...")
                        save_checkpoint(articles, seen_urls, idx)

            # Small delay before next request
            time.sleep(REQUEST_DELAY)

        # Small delay before next section
        time.sleep(REQUEST_DELAY * 2)

    # Final write of any remaining articles
    remaining_count = len(articles) % CSV_WRITE_INTERVAL
    if remaining_count > 0:
        start_idx = len(articles) - remaining_count
        logger.info(f"Writing final partial CSV with {remaining_count} articles...")
        write_articles_to_csv(articles[start_idx:], OUT_CSV, append=True)

    # Save final progress
    save_seen_urls(seen_urls)
    logger.info(f"Scraping completed successfully. Scraped {len(articles) - starting_articles_count} new articles across {len(all_urls)} sections.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape The Hindu articles from YES sections')
    parser.add_argument('--urls-file', type=str, default='data/YES_url.md',  # Changed default filename
                        help='File containing YES URLs to scrape (default: data/YES_url.md)')
    parser.add_argument('--max-per-section', type=int, default=MAX_ARTICLES_PER_SECTION,
                        help=f'Maximum articles to scrape per section (default: {MAX_ARTICLES_PER_SECTION})')

    args = parser.parse_args()

    scrape_sections(urls_file=args.urls_file, max_articles_per_section=args.max_per_section)
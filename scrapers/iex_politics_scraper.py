"""
Scrape all articles from https://indianexpress.com/section/politics/
and write CSV with columns:
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
        logging.FileHandler('iex_politics_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

BASE_SECTION = "https://indianexpress.com/section/politics/"
USER_AGENT = "Mozilla/0.0 (compatible; MyScraper/1.0; +https://example.com/bot)"
HEADERS = {"User-Agent": USER_AGENT}
OUT_CSV = "data/iex_politics.csv"
CHECKPOINT_FILE = "data/iex_politics_checkpoint.json"
SEEN_URLS_FILE = "data/iex_politics_seen_urls.json"
REQUEST_DELAY = 2.0   # seconds between requests (increased for polite crawling)
MAX_EMPTY_PAGES = 3   # stop if N consecutive pages contain no new links (safety)
DEFAULT_MAX_PAGES = 1000  # default page limit
CHECKPOINT_INTERVAL = 10  # save checkpoint every N pages
CSV_WRITE_INTERVAL = 50   # write partial CSV every N articles
MAX_RETRIES = 3   # maximum number of retries for failed requests
REQUEST_TIMEOUT = 20  # timeout for requests in seconds

# Regex to identify probable article links (Indian Express Politics articles have /article/india/politics/ or subcategories in URL)
ARTICLE_RE = re.compile(r"https?://[^/]+/article/india/politics/.+")  # absolute links for main politics articles
SUBCATEGORY_ARTICLE_RE = re.compile(r"https?://[^/]+/article/india/politics/[^/]+/.+")  # articles in politics subcategories
REL_ARTICLE_RE = re.compile(r"/article/india/politics/.+")           # relative links

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
    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching URL: {url}")
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            logger.debug(f"Successfully fetched URL: {url}")
            return r.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"fetch error {url} (attempt {attempt + 1}/{retries + 1}): {e}")
            if attempt < retries:
                time.sleep(REQUEST_DELAY * 2)  # Wait longer between retries
            else:
                logger.error(f"Failed to fetch {url} after {retries} retries")
                return None
    return None

def parse_section_page(html, base_url):
    """Return set of absolute article URLs found on a section/archive page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    # Collect every <a href> that looks like an article permalink using regex heuristics.
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if ARTICLE_RE.match(href) or SUBCATEGORY_ARTICLE_RE.match(href):
            links.add(href)
        elif REL_ARTICLE_RE.match(href):
            links.add(urljoin(base_url, href))

    logger.debug(f"Parsed section page and found {len(links)} potential article links")
    return links

def parse_article(html, url):
    """Extract title, publish_date (ISO), tags list, and word count (if possible)."""
    logger.debug(f"Parsing article: {url}")
    soup = BeautifulSoup(html, "html.parser")
    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""
    if not title:
        logger.warning(f"No title found for article: {url}")
    # Publish date: look for time tag or meta property
    date = ""
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        date = time_tag.get("datetime").strip()
    else:
        # meta property
        meta_pub = soup.find("meta", {"property":"article:published_time"})
        if meta_pub and meta_pub.get("content"):
            date = meta_pub["content"].strip()
        else:
            # fallback: try .published or .date classes
            el = soup.find(class_=re.compile(r"(date|published|entry-date)"))
            if el:
                date = el.get_text(strip=True)

    if not date:
        logger.warning(f"No publish date found for article: {url}")

    # Tags: look for rel="tag" anchors or class 'tags' elements
    tags = []
    for tag_a in soup.find_all("a", rel="tag"):
        tags.append(tag_a.get_text(strip=True))
    if not tags:
        # fallback: find container with 'tags' in class
        tag_cont = soup.find(class_=re.compile(r"tag|tags", re.I))
        if tag_cont:
            for ta in tag_cont.find_all("a"):
                tags.append(ta.get_text(strip=True))

    # Article body word count attempt
    # Indian Express often uses itemprop="articleBody" or div with class containing 'article' or 'content'
    body_text = ""
    body = soup.find(attrs={"itemprop":"articleBody"})
    if not body:
        body = soup.find("div", class_=re.compile(r"article|content|entry-content", re.I))
    if body:
        body_text = body.get_text(separator=" ", strip=True)
    word_count = len(body_text.split()) if body_text else 0

    logger.debug(f"Successfully parsed article: {url} - Title: {title[:50]}...")
    return {
        "title": title,
        "date": date,
        "tags": tags,
        "word_count": word_count,
        "body_text": body_text
    }

def save_checkpoint(page, articles, seen_urls):
    """Save progress to checkpoint file."""
    logger.info(f"Saving checkpoint at page {page}")
    checkpoint_data = {
        "current_page": page,
        "articles": articles,
        "seen_urls": list(seen_urls)
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Checkpoint saved at page {page}")

def load_checkpoint():
    """Load progress from checkpoint file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        logger.info("Loading checkpoint data...")
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        logger.info(f"Loaded checkpoint data from page {checkpoint_data.get('current_page', 1)} with {len(checkpoint_data.get('articles', []))} articles")
        return (
            checkpoint_data.get("current_page", 1),
            checkpoint_data.get("articles", []),
            set(checkpoint_data.get("seen_urls", []))
        )
    logger.info("No checkpoint file found, starting from scratch")
    return 1, [], set()

def save_seen_urls(seen_urls):
    """Save seen URLs to file."""
    logger.info(f"Saving {len(seen_urls)} seen URLs to file")
    with open(SEEN_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_urls), f, indent=2)
    logger.info(f"Seen URLs saved successfully")

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

def write_partial_csv(articles, filename=OUT_CSV, append=True):
    """Write articles to CSV file (append mode for partial writes)."""
    logger.info(f"Writing {len(articles)} articles to CSV file: {filename}")
    write_header = not os.path.exists(filename) or not append
    with open(filename, "a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["id","year","paper","question_no","question_text","word_limit","marks","topic_hint","source_url"])
        for row in articles:
            writer.writerow([
                row["id"],
                row["year"],
                row["paper"],
                row["question_no"],
                row["title"],
                row["word_limit"],
                row["marks"],
                row["topic_hint"],
                row["source_url"]
            ])
    logger.info(f"Successfully wrote {len(articles)} articles to CSV")

def section_page_url(page_num):
    if page_num == 1:
        return BASE_SECTION
    return urljoin(BASE_SECTION, f"page/{page_num}/")

def scrape(max_pages=DEFAULT_MAX_PAGES, resume_from_page=1):
    global shutdown_requested
    setup_signal_handlers()

    # Load existing progress and seen URLs
    checkpoint_page, articles, seen_urls = load_checkpoint()
    initial_seen_urls = load_seen_urls()
    seen_urls.update(initial_seen_urls)

    starting_page = resume_from_page if resume_from_page > 1 else checkpoint_page
    logger.info(f"Starting scrape from page {starting_page}. Found {len(seen_urls)} previously seen URLs.")

    page = starting_page
    empty_pages = 0
    articles_count_at_start = len(articles)
    last_checkpoint_page = starting_page - 1

    # If user wants to limit pages during testing, pass max_pages.
    while True:
        if shutdown_requested:
            logger.info("Shutdown requested. Saving progress before exit...")
            save_checkpoint(page, articles, seen_urls)
            save_seen_urls(seen_urls)
            break

        if max_pages and (page - starting_page + 1) > max_pages:
            logger.info(f"Reached maximum pages limit ({max_pages}). Stopping.")
            break
        section_url = section_page_url(page)
        logger.info(f"Fetching section page: {section_url}")
        html = fetch(section_url)
        if not html:
            logger.error(f"Unable to fetch section page {section_url} — stopping.")
            break
        links = parse_section_page(html, BASE_SECTION)
        new_links = [l for l in links if l not in seen_urls]
        logger.info(f"Found {len(links)} links; {len(new_links)} new.")
        if not new_links:
            empty_pages += 1
        else:
            empty_pages = 0

        if empty_pages >= MAX_EMPTY_PAGES:
            logger.info("No new links on multiple pages — assuming end of archive. Stopping.")
            break

        logger.info(f"Processing {len(new_links)} new articles on page {page}")
        for article_url in tqdm(sorted(new_links), desc=f"Articles on page {page}"):
            if shutdown_requested:
                logger.info("Shutdown requested. Saving progress before exit...")
                save_checkpoint(page, articles, seen_urls)
                save_seen_urls(seen_urls)
                return  # Exit early to handle shutdown

            time.sleep(REQUEST_DELAY)
            art_html = fetch(article_url)
            if not art_html:
                logger.warning(f"Failed to fetch article: {article_url}")
                continue
            art = parse_article(art_html, article_url)
            seen_urls.add(article_url)

            # Calculate the ID for this new article to be sequential
            article_id = len(articles) + articles_count_at_start + 1

            # Map to CSV fields required by user
            # id and question_no assigned later (sequential)
            year = ""
            if art["date"]:
                # attempt to extract year with regex
                m = re.search(r"(\d{4})", art["date"])
                if m:
                    year = m.group(1)

            topic_hint = ";".join(art["tags"]) if art["tags"] else ""
            article_data = {
                "id": article_id,
                "year": year,
                "paper": "Indian Express - Politics",
                "question_no": article_id,  # question_no == id (change if you want per-year numbering)
                "title": art["title"],
                "word_limit": "",           # word_limit (not applicable for news)
                "marks": "",                # marks (not applicable)
                "topic_hint": topic_hint,
                "source_url": article_url,
                "word_count": art["word_count"],
                "raw_date": art["date"]
            }

            articles.append(article_data)

            # Write partial CSV every N articles to prevent data loss
            if len(articles) % CSV_WRITE_INTERVAL == 0:
                logger.info(f"Writing partial CSV with {len(articles)} articles...")
                # Only write newly added articles to avoid duplicates
                new_articles_start = len(articles) - CSV_WRITE_INTERVAL
                write_partial_csv(articles[new_articles_start:], append=True)

        # Save checkpoint every N pages
        if page % CHECKPOINT_INTERVAL == 0:
            logger.info(f"Saving checkpoint at page {page}...")
            save_checkpoint(page, articles, seen_urls)
            save_seen_urls(seen_urls)
            last_checkpoint_page = page

        page += 1
        # small delay before next section page
        time.sleep(REQUEST_DELAY)

    # Final save of progress
    logger.info(f"Finalizing scrape. Writing {len(articles)} rows to {OUT_CSV}")
    # Write any remaining articles since last partial write
    if len(articles) % CSV_WRITE_INTERVAL != 0:
        remaining_count = len(articles) % CSV_WRITE_INTERVAL
        if remaining_count > 0:
            start_idx = len(articles) - remaining_count
            write_partial_csv(articles[start_idx:], append=True)

    save_seen_urls(seen_urls)

    logger.info("Scraping completed successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scrape Indian Express Politics articles')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES,
                        help=f'Maximum number of pages to scrape (default: {DEFAULT_MAX_PAGES})')
    parser.add_argument('--resume-from', type=int, default=1,
                        help='Page number to resume scraping from (default: 1)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset progress and start from scratch')

    args = parser.parse_args()

    if args.reset:
        print("Resetting progress...")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        if os.path.exists(SEEN_URLS_FILE):
            os.remove(SEEN_URLS_FILE)

    # Example: scrape first 50 pages for testing:
    # scrape(max_pages=50)
    # To do full scrape leave max_pages=None (but be patient; site has many pages).
    scrape(max_pages=args.max_pages, resume_from_page=args.resume_from)
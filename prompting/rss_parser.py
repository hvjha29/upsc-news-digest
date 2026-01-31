"""
RSS Feed Parser for Indian Express and other news sources.
Fetches articles from RSS feeds and extracts title, link, description, and content.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_TIMEOUT = 20


@dataclass
class Article:
    """Represents a news article."""
    title: str
    link: str
    description: str
    content: str
    published: Optional[str] = None
    source: str = "indian_express"
    
    def get_text_for_classification(self) -> str:
        """Return combined text for classification."""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.description:
            parts.append(f"Summary: {self.description}")
        if self.content:
            parts.append(f"Content: {self.content}")
        return "\n\n".join(parts)


def fetch_url(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
    """Fetch URL content with retry logic."""
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < 2:
                time.sleep(2)
    return None


def extract_article_content(url: str) -> str:
    """Extract main article content from a news page."""
    html = fetch_url(url)
    if not html:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
        tag.decompose()
    
    content = ""
    
    # Indian Express specific selectors
    article_body = soup.find('div', class_='full-details')
    if article_body:
        paragraphs = article_body.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs)
    
    # Fallback: try common article containers
    if not content:
        for selector in ['article', '.article-body', '.story-content', '.article-content', '#article-body']:
            container = soup.select_one(selector)
            if container:
                paragraphs = container.find_all('p')
                content = ' '.join(p.get_text(strip=True) for p in paragraphs)
                if content:
                    break
    
    # Last fallback: get all paragraphs
    if not content:
        paragraphs = soup.find_all('p')
        content = ' '.join(p.get_text(strip=True) for p in paragraphs[:20])  # Limit to first 20 paragraphs
    
    return content.strip()


def parse_rss_feed(feed_url: str, fetch_content: bool = True, delay: float = 1.0) -> List[Article]:
    """
    Parse an RSS feed and return list of Article objects.
    
    Args:
        feed_url: URL of the RSS feed
        fetch_content: Whether to fetch full article content (slower but more accurate)
        delay: Delay between requests when fetching content
    
    Returns:
        List of Article objects
    """
    logger.info(f"Parsing RSS feed: {feed_url}")
    
    feed = feedparser.parse(feed_url)
    
    if feed.bozo:
        logger.warning(f"Feed parsing warning: {feed.bozo_exception}")
    
    articles = []
    total = len(feed.entries)
    
    logger.info(f"Found {total} entries in feed")
    
    for i, entry in enumerate(feed.entries):
        title = entry.get('title', '').strip()
        link = entry.get('link', '').strip()
        
        # Get description/summary
        description = ""
        if 'summary' in entry:
            # Clean HTML from summary
            soup = BeautifulSoup(entry.summary, 'html.parser')
            description = soup.get_text(strip=True)
        elif 'description' in entry:
            soup = BeautifulSoup(entry.description, 'html.parser')
            description = soup.get_text(strip=True)
        
        # Get published date
        published = entry.get('published', entry.get('updated', ''))
        
        # Fetch full content if requested
        content = ""
        if fetch_content and link:
            logger.info(f"[{i+1}/{total}] Fetching content for: {title[:50]}...")
            content = extract_article_content(link)
            time.sleep(delay)  # Be polite
        
        article = Article(
            title=title,
            link=link,
            description=description,
            content=content,
            published=published
        )
        articles.append(article)
    
    logger.info(f"Successfully parsed {len(articles)} articles")
    return articles


def get_feed_titles_only(feed_url: str) -> List[Dict[str, str]]:
    """
    Quick function to just get titles and links from RSS feed.
    Useful for preview without fetching full content.
    """
    feed = feedparser.parse(feed_url)
    
    results = []
    for entry in feed.entries:
        results.append({
            'title': entry.get('title', '').strip(),
            'link': entry.get('link', '').strip(),
            'published': entry.get('published', entry.get('updated', '')),
            'summary': BeautifulSoup(entry.get('summary', ''), 'html.parser').get_text(strip=True)[:200]
        })
    
    return results


if __name__ == "__main__":
    # Test the parser
    feed_url = "https://indianexpress.com/rss/"
    
    print("=" * 60)
    print("Fetching RSS Feed Titles (Quick Preview)")
    print("=" * 60)
    
    titles = get_feed_titles_only(feed_url)
    for i, item in enumerate(titles[:10], 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Link: {item['link']}")
        print(f"   Summary: {item['summary'][:100]}...")

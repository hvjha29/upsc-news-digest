"""
Newspaper Scraper for Indian Express and The Hindu Today's Paper ONLY.
Fetches articles exclusively from:
- https://indianexpress.com/todays-paper/
- https://www.thehindu.com/todays-paper/YYYY-MM-DD/th_chennai/
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Headers to mimic browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


@dataclass
class Article:
    """Represents a news article."""
    title: str
    link: str
    source: str  # 'indian_express' or 'the_hindu'
    section: str  # 'editorial', 'opinion', 'explained', 'national', etc.
    summary: str = ""
    content: str = ""
    author: str = ""
    
    def get_text_for_classification(self) -> str:
        """Get text for classification."""
        text = f"Title: {self.title}"
        if self.summary:
            text += f"\nSummary: {self.summary}"
        if self.content:
            text += f"\nContent: {self.content[:1500]}"
        return text


class IndianExpressScraper:
    """Scraper for Indian Express Archive (more robust than todays-paper)."""
    
    BASE_URL = "https://indianexpress.com"
    MAX_PAGES = 20  # Safety limit for pagination
    
    def __init__(self, date: Optional[datetime] = None):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.date = date or datetime.now()
    
    def _get_archive_url(self, page: int = 1) -> str:
        """Get archive URL with pagination: /archive/yyyy/mm/dd/ or /archive/yyyy/mm/dd/page/N/"""
        date_str = self.date.strftime('%Y/%m/%d')
        if page == 1:
            return f"https://indianexpress.com/archive/{date_str}/"
        else:
            return f"https://indianexpress.com/archive/{date_str}/page/{page}/"
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def scrape_todays_paper(self) -> List[Article]:
        """Scrape all pages from today's archive."""
        all_articles = []
        seen_urls = set()
        
        logger.info(f"Fetching Indian Express Archive for {self.date.strftime('%Y-%m-%d')}")
        
        for page in range(1, self.MAX_PAGES + 1):
            url = self._get_archive_url(page)
            logger.info(f"  Scraping page {page}: {url}")
            soup = self._get_page(url)
            if not soup:
                break
            
            page_articles = []
            new_articles_count = 0
            
            # Find all article links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                # Only include article links from indianexpress.com
                if '/article/' in href and 'indianexpress.com' in href:
                    if href not in seen_urls:
                        seen_urls.add(href)
                        title = link.get_text(strip=True)
                        if title and len(title) > 20:  # Filter out short/navigation links
                            section = self._extract_section(href)
                            page_articles.append(Article(
                                title=title,
                                link=href,
                                source='indian_express',
                                section=section
                            ))
                            new_articles_count += 1
            
            all_articles.extend(page_articles)
            logger.info(f"    Found {new_articles_count} new articles")
            
            # Stop if no new articles found (we've seen all pages)
            if new_articles_count == 0:
                logger.info(f"  No new articles on page {page}, stopping pagination")
                break
        
        logger.info(f"Found {len(all_articles)} unique articles from Indian Express Archive")
        return all_articles
    
    def _extract_section(self, url: str) -> str:
        """Extract section from URL."""
        if '/opinion/' in url or '/editorials/' in url or '/columns/' in url:
            return 'opinion'
        elif '/explained/' in url:
            return 'explained'
        elif '/political-pulse/' in url:
            return 'political-pulse'
        elif '/legal-news/' in url:
            return 'legal-news'
        elif '/india/' in url:
            return 'india'
        elif '/business/' in url:
            return 'business'
        elif '/world/' in url:
            return 'world'
        elif '/cities/' in url:
            return 'cities'
        return 'general'
    
    def fetch_article_content(self, article: Article, delay: float = 1.0) -> str:
        """Fetch full content of an article."""
        time.sleep(delay)
        soup = self._get_page(article.link)
        if not soup:
            return ""
        
        content_parts = []
        
        # Try different content containers
        content_div = soup.find('div', class_='full-details') or \
                      soup.find('div', class_='story_details') or \
                      soup.find('article')
        
        if content_div:
            # Get all paragraphs
            for p in content_div.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 30:
                    content_parts.append(text)
        
        # Get summary/description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            article.summary = meta_desc.get('content', '')
        
        article.content = '\n\n'.join(content_parts)
        return article.content
    
    def get_all_relevant_articles(self) -> List[Article]:
        """Get all articles from today's paper only."""
        return self.scrape_todays_paper()


class TheHinduScraper:
    """Scraper for The Hindu Today's Paper - ALL sections including city editions."""
    
    BASE_URL = "https://www.thehindu.com"
    
    # All sections of The Hindu today's paper (including city editions)
    SECTIONS = [
        # City editions (have most articles)
        'th_chennai',      # Chennai local news
        'th_mumbai',       # Mumbai edition
        'th_hyderabad',    # Hyderabad edition
        'th_kolkata',      # Kolkata edition
        # Main sections (may have duplicates across editions)
        'th_national',     # National news
        'th_international', # International news
        'th_editorial',    # Editorials
        'th_opinion',      # Opinion pieces
        'th_business',     # Business news
    ]
    
    # Additional opinion/editorial pages to scrape (top banner articles only)
    OPINION_SECTIONS = [
        'https://www.thehindu.com/opinion/',
        'https://www.thehindu.com/opinion/editorial/',
        'https://www.thehindu.com/opinion/op-ed/',
        'https://www.thehindu.com/opinion/lead/',
    ]
    
    def __init__(self, date: Optional[datetime] = None):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.date = date or datetime.now()
    
    def _get_section_url(self, section: str) -> str:
        """Get URL for a specific section."""
        date_str = self.date.strftime("%Y-%m-%d")
        return f"https://www.thehindu.com/todays-paper/{date_str}/{section}/"
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a page."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _scrape_section(self, section: str) -> List[Article]:
        """Scrape a single section of today's paper."""
        articles = []
        url = self._get_section_url(section)
        date_str = self.date.strftime("%Y-%m-%d")
        
        soup = self._get_page(url)
        if not soup:
            return articles
        
        # Find all article links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            full_url = href if href.startswith('http') else f"https://www.thehindu.com{href}"
            
            # Include articles from today's paper format OR regular thehindu.com articles
            is_todays_paper_article = f'/todays-paper/{date_str}/' in full_url and '.ece' in full_url
            is_hindu_article = 'thehindu.com' in full_url and '.ece' in full_url and '/article' in full_url
            
            # Exclude sportstar, frontline, businessline (separate publications)
            is_main_hindu = 'www.thehindu.com' in full_url
            
            if (is_todays_paper_article or is_hindu_article) and is_main_hindu:
                title = link.get_text(strip=True)
                if title and len(title) > 15:
                    articles.append(Article(
                        title=title,
                        link=full_url,
                        source='the_hindu',
                        section=section.replace('th_', '')  # e.g., 'th_editorial' -> 'editorial'
                    ))
        
        return articles
    
    def _scrape_opinion_banner(self, url: str) -> List[Article]:
        """Scrape top banner articles from opinion/editorial pages (NOT 'More stories')."""
        articles = []
        soup = self._get_page(url)
        if not soup:
            return articles
        
        # Find the "more stories" section and only get articles BEFORE it
        more_stories_marker = soup.find(string=lambda t: t and 'more stories' in t.lower() if t else False)
        
        # Get all article links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            
            # If we've passed the "more stories" section, stop
            if more_stories_marker:
                # Check if this link comes after the marker
                link_position = str(soup).find(str(link))
                marker_position = str(soup).find(str(more_stories_marker))
                if link_position > marker_position:
                    continue
            
            # Only include opinion articles from thehindu.com
            if '/opinion/' in href and '.ece' in href and 'thehindu.com' in href:
                full_url = href if href.startswith('http') else f"https://www.thehindu.com{href}"
                title = link.get_text(strip=True)
                if title and len(title) > 20:
                    # Determine section from URL
                    section = 'opinion'
                    if '/editorial/' in href:
                        section = 'editorial'
                    elif '/lead/' in href:
                        section = 'lead'
                    elif '/op-ed/' in href:
                        section = 'op-ed'
                    
                    articles.append(Article(
                        title=title,
                        link=full_url,
                        source='the_hindu',
                        section=section
                    ))
        
        return articles
    
    def scrape_todays_paper(self) -> List[Article]:
        """Scrape ALL sections of today's paper plus opinion banner articles."""
        all_articles = []
        date_str = self.date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching The Hindu Today's Paper (all sections) for {date_str}")
        
        for section in self.SECTIONS:
            logger.info(f"  Scraping section: {section}")
            section_articles = self._scrape_section(section)
            all_articles.extend(section_articles)
            logger.info(f"    Found {len(section_articles)} articles")
        
        # Also scrape opinion/editorial banner articles (today's featured)
        logger.info("Scraping opinion/editorial banner articles...")
        for opinion_url in self.OPINION_SECTIONS:
            logger.info(f"  Scraping: {opinion_url}")
            banner_articles = self._scrape_opinion_banner(opinion_url)
            all_articles.extend(banner_articles)
            logger.info(f"    Found {len(banner_articles)} banner articles")
        
        # Deduplicate by link
        seen = set()
        unique_articles = []
        for article in all_articles:
            if article.link not in seen:
                seen.add(article.link)
                unique_articles.append(article)
        
        logger.info(f"Found {len(unique_articles)} unique articles from The Hindu")
        return unique_articles
    
    def _extract_section(self, url: str) -> str:
        """Extract section from URL."""
        if '/opinion/' in url or '/editorial/' in url:
            return 'opinion'
        elif '/lead/' in url:
            return 'lead'
        elif '/national/' in url:
            return 'national'
        elif '/international/' in url:
            return 'international'
        elif '/business/' in url:
            return 'business'
        elif '/sci-tech/' in url:
            return 'science'
        elif '/sport/' in url:
            return 'sports'
        return 'general'
    
    def fetch_article_content(self, article: Article, delay: float = 1.0) -> str:
        """Fetch full content of an article."""
        time.sleep(delay)
        soup = self._get_page(article.link)
        if not soup:
            return ""
        
        content_parts = []
        
        # Try different content containers
        content_div = soup.find('div', class_='articlebodycontent') or \
                      soup.find('article') or \
                      soup.find('div', class_='article-body')
        
        if content_div:
            for p in content_div.find_all('p'):
                text = p.get_text(strip=True)
                if text and len(text) > 30:
                    content_parts.append(text)
        
        # Get summary
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            article.summary = meta_desc.get('content', '')
        
        article.content = '\n\n'.join(content_parts)
        return article.content
    
    def get_all_relevant_articles(self) -> List[Article]:
        """Get all articles from today's paper only."""
        return self.scrape_todays_paper()


def scrape_all_newspapers(date: Optional[datetime] = None) -> List[Article]:
    """
    Scrape articles from today's paper ONLY from both newspapers.
    
    Args:
        date: Optional date to scrape for (defaults to today)
    """
    all_articles = []
    target_date = date or datetime.now()
    
    logger.info(f"Scraping Today's Paper for: {target_date.strftime('%Y-%m-%d')}")
    
    # Indian Express Today's Paper
    logger.info("="*50)
    logger.info("Scraping Indian Express Today's Paper...")
    ie_scraper = IndianExpressScraper()
    ie_articles = ie_scraper.get_all_relevant_articles()
    all_articles.extend(ie_articles)
    
    # The Hindu Today's Paper (Chennai edition)
    logger.info("="*50)
    logger.info("Scraping The Hindu Today's Paper (Chennai)...")
    th_scraper = TheHinduScraper(date=target_date)
    th_articles = th_scraper.get_all_relevant_articles()
    all_articles.extend(th_articles)
    
    logger.info("="*50)
    logger.info(f"Total articles from Today's Paper: {len(all_articles)}")
    logger.info(f"  - Indian Express: {len(ie_articles)}")
    logger.info(f"  - The Hindu: {len(th_articles)}")
    
    return all_articles


if __name__ == "__main__":
    articles = scrape_all_newspapers()
    
    print("\n" + "="*70)
    print("TODAY'S PAPER ARTICLES")
    print("="*70)
    
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. [{article.source}] [{article.section}]")
        print(f"   {article.title}")
        print(f"   {article.link}")

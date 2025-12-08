"""
Scrape articles from https://compass.rauias.com for all of November 2025
and write CSV with columns:
id,year,paper,question_no,question_text,word_limit,marks,topic_hint,source_url

This scraper will:
1. Iterate through all dates in November 2025
2. Scrape content from each date's page
3. Extract articles from each page
4. Write to rausias_yes.csv
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
from datetime import date, timedelta
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rauias_scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://compass.rauias.com/"
START_DATE = date(2025, 11, 1)
END_DATE = date(2025, 11, 30)
USER_AGENT = "Mozilla/5.0 (compatible; RauIASBot/1.0; +https://example.com/bot)"

# Request session with headers
session = requests.Session()
session.headers.update({
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
})

def get_date_range(start_date, end_date):
    """
    Generate all dates in the given range
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)

def get_articles_from_date_page(date_obj):
    """
    Get all articles from a specific date page
    URL format: https://compass.rauias.com/YYYY/MM/DD/
    """
    date_url = f"{BASE_URL}{date_obj.year:04d}/{date_obj.month:02d}/{date_obj.day:02d}/"
    logger.info(f"Fetching articles from {date_url}")
    
    try:
        response = session.get(date_url, timeout=30)
        
        # If page doesn't exist, return empty list
        if response.status_code == 404:
            logger.info(f"Page does not exist: {date_url}")
            return []
        
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Look for article containers - common selectors for blog posts
        article_selectors = [
            'article',  # Standard article tag
            '.post',  # Common post class
            '.entry',  # Common entry class
            '.blog-post',  # Blog post class
            '.article-item',  # Article item class
            '.news-item',  # News item class
            '.entry-content',  # Entry content
        ]
        
        articles = []
        
        # Try to find articles using various selectors
        found = False
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    title_elem = element.select_one('h1, h2, h3, .entry-title, .post-title')
                    content_elem = element.select_one('.entry-content, .post-content, .content, .article-content, .post-body, p')
                    
                    title = ""
                    content = ""
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    
                    if content_elem:
                        content = content_elem.get_text(separator=' ', strip=True)
                    
                    # If we have meaningful content, add it
                    if content and len(content) > 50:  # At least 50 characters
                        # If title is empty, try to extract from the main page title
                        if not title:
                            title_tag = soup.find('title')
                            if title_tag:
                                title = title_tag.get_text().strip()
                        
                        # If title is still empty, extract from URL or use a default
                        if not title:
                            title = f"Article from {date_obj.strftime('%Y-%m-%d')}"
                        
                        full_text = f"{title}. {content}" if title and content else content
                        
                        articles.append({
                            'title': title,
                            'content': content,
                            'full_text': full_text,
                            'date': date_obj.strftime('%Y-%m-%d'),
                            'url': date_url,
                            'topic_hint': 'RAU IAS;Editorial Analysis;Current Affairs'
                        })
                        found = True
                break
        
        # If no articles found with structured selectors,
        # try to get content more generally
        if not found:
            # Look for any paragraphs that might be article content
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
                if content:
                    title_tag = soup.find('title')
                    title = title_tag.get_text().strip() if title_tag else f"Article from {date_obj.strftime('%Y-%m-%d')}"
                    
                    articles.append({
                        'title': title,
                        'content': content,
                        'full_text': f"{title}. {content}",
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'url': date_url,
                        'topic_hint': 'RAU IAS;Editorial Analysis;Current Affairs'
                    })
        
        logger.info(f"Found {len(articles)} articles on {date_url}")
        return articles
        
    except requests.RequestException as e:
        if e.response and e.response.status_code == 404:
            logger.info(f"Page does not exist: {date_url}")
        else:
            logger.error(f"Error fetching {date_url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error parsing {date_url}: {e}")
        return []


def scrape_november_articles():
    """
    Scrape articles from all dates in November 2025
    """
    logger.info(f"Starting to scrape articles from {START_DATE} to {END_DATE}")
    
    all_articles = []
    
    for current_date in tqdm(list(get_date_range(START_DATE, END_DATE)), desc="Scraping dates"):
        articles = get_articles_from_date_page(current_date)
        all_articles.extend(articles)
        
        # Be respectful to the server
        time.sleep(1)
    
    logger.info(f"Total articles scraped: {len(all_articles)}")
    return all_articles


def create_rauias_csv(articles_data, output_file='data/rausias_yes.csv'):
    """
    Create CSV file with the scraped articles
    """
    logger.info(f"Creating CSV file: {output_file}")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'year', 'paper', 'question_no', 'question_text', 'word_limit', 'marks', 'topic_hint', 'source_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, article in enumerate(tqdm(articles_data, desc="Writing to CSV")):
            row = {
                'id': str(uuid.uuid4()),
                'year': article.get('date', '').split('-')[0] if article.get('date') else '2025',
                'paper': 'RAU IAS Editorial Analysis',
                'question_no': i + 1,
                'question_text': article.get('full_text', '')[:5000],  # Limit text length
                'word_limit': '',
                'marks': '',
                'topic_hint': article.get('topic_hint', ''),
                'source_url': article.get('url', '')
            }
            
            # Clean the question_text to remove excessive whitespace and newlines
            row['question_text'] = " ".join(row['question_text'].split())
            
            writer.writerow(row)
    
    logger.info(f"Successfully created {output_file} with {len(articles_data)} articles")


def main():
    """
    Main function to scrape RAU IAS articles and create CSV
    """
    logger.info("Starting RAU IAS scraper for November 2025")
    
    # Scrape articles from all dates in November
    articles = scrape_november_articles()
    
    if not articles:
        logger.warning("No articles found. Creating an empty CSV with headers.")
        # Create an empty file with headers
        os.makedirs('data', exist_ok=True)
        with open('data/rausias_yes.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'year', 'paper', 'question_no', 'question_text', 'word_limit', 'marks', 'topic_hint', 'source_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        logger.info("Created empty CSV file: data/rausias_yes.csv")
        return
    
    # Create the CSV file
    create_rauias_csv(articles)
    
    logger.info(f"Scraping completed. Processed {len(articles)} articles.")


if __name__ == "__main__":
    main()
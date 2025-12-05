import requests
from bs4 import BeautifulSoup
import time
import csv
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path

def get_page_content(url):
    """
    Fetch and parse the content of a webpage
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_article_links(base_url):
    """
    Extract all individual article links from a Vajiram & IAS articles page
    """
    soup = get_page_content(base_url)
    if not soup:
        return []
    
    article_links = []
    processed_links = set()
    
    # Find all links on the page
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link['href']
        
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        
        # Check if this is a likely article URL
        if (('/articles/' in full_url or 
             '/article' in full_url or
             full_url.startswith('https://vajiramias.com/article')) and
            not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.json']) and
            'vajiramias.com' in full_url and
            full_url != base_url and  # Not the same page
            full_url not in processed_links):
            
            # Additional check: make sure it's not a general site navigation link
            if not any(skip in full_url for skip in ['/about', '/contact', '/privacy', '/terms', '/login', '/register', '/search', '/category', '/tag', '/author']):
                
                article_links.append(full_url)
                processed_links.add(full_url)
    
    # Look for article links in specific containers that commonly contain article lists
    article_containers = soup.find_all(['div', 'section', 'ul', 'ol'], class_=re.compile(r'article|post|list|archive|news|blog|content', re.I))
    
    for container in article_containers:
        for link in container.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if (('/articles/' in full_url or 
                 '/article' in full_url or
                 full_url.startswith('https://vajiramias.com/article')) and
                full_url not in processed_links and
                full_url != base_url):
                
                if not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.json']):
                    
                    article_links.append(full_url)
                    processed_links.add(full_url)
    
    return list(set(article_links))  # Remove duplicates

def extract_article_content(article_url):
    """
    Extract the main content from a specific article page
    """
    soup = get_page_content(article_url)
    if not soup:
        return None
    
    # Try different selectors to find the main content
    content_selectors = [
        'article',  # HTML5 article tag
        '.article-content',  # Common class names
        '.article__content',
        '.content', 
        '.post-content',
        '.entry-content',
        '.main-content',
        '[class*="article"]',
        '[class*="content"]',
        '.single-post-content',
        '#content',
        'main',
        '.post-body',
        '.post-inner',
        '.entry-content',
        '.story-content',
        '.article-body'
    ]
    
    article_text = ""
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            for element in elements:
                # Remove script and style elements
                for script in element(["script", "style"]):
                    script.decompose()
                
                text = element.get_text(separator=' ', strip=True)
                if len(text) > len(article_text):  # Get the longest text
                    article_text = text
    
    # If no content found with selectors, try to get all paragraphs within main content areas
    if not article_text:
        # Look for main content areas
        main_selectors = ['main', '.main', '.container', '.content-area', '.post']
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    # Get all paragraphs within the main content
                    paragraphs = element.find_all('p')
                    text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    if len(text) > len(article_text):
                        article_text = text
    
    # If still no content found, try to get all paragraphs
    if not article_text:
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    
    # Clean the text
    article_text = re.sub(r'\s+', ' ', article_text).strip()
    
    # Get title if available
    title = ""
    title_element = soup.find('h1') or soup.find('title')
    if title_element:
        title = title_element.get_text().strip()
    
    # If the title is just "Vajiram & IAS", use the first heading
    if title and ('vajiram' in title.lower() and 'ias' in title.lower() and len(title.split()) <= 4):
        for tag in ['h1', 'h2', 'h3']:
            heading = soup.find(tag)
            if heading and heading.get_text().strip():
                title = heading.get_text().strip()
                break
    
    return {
        'url': article_url,
        'title': title,
        'content': article_text
    }

def extract_year_from_url(url):
    """
    Extract year from the URL
    """
    # Extract year from URL pattern like /articles/2025/11/
    match = re.search(r'/articles/(\d{4})/', url)
    if match:
        return match.group(1)
    return "2025"  # Default to 2025 if not found

def create_csv_row(article_data, article_url, index):
    """
    Create a row in the format: id,year,paper,question_no,question_text,word_limit,marks,topic_hint,source_url
    """
    year = extract_year_from_url(article_url)
    
    # Extract topic hint from the URL path or article content
    topic_hint = "Mains Articles"
    path_parts = urlparse(article_url).path.split('/')
    # Take the last meaningful part of the URL as topic hint if available
    if len(path_parts) > 2:
        potential_hint = path_parts[-2] if path_parts[-1] == '' else path_parts[-1]
        if len(potential_hint) > 2:  # Avoid short fragments
            topic_hint = potential_hint.replace('-', ' ').title()
    
    # Calculate word count
    word_count = len(article_data['content'].split()) if article_data['content'] else 0
    
    # Create a unique ID
    id_value = f"VJ{year}{index:04d}"
    
    return {
        'id': id_value,
        'year': year,
        'paper': "Vajiram IAS Mains Articles",  # Generic paper name
        'question_no': "",  # Not applicable for articles
        'question_text': article_data['content'][:5000],  # Limit to 5000 chars to avoid CSV issues
        'word_limit': word_count,
        'marks': "",  # Not applicable for articles
        'topic_hint': topic_hint,
        'source_url': article_url
    }

def save_articles_to_csv_format(all_articles, output_file):
    """
    Save articles to a CSV file in the specified format
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'year', 'paper', 'question_no', 'question_text', 'word_limit', 'marks', 'topic_hint', 'source_url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, (article_url, article_data) in enumerate(all_articles):
            csv_row = create_csv_row(article_data, article_url, i+1)
            writer.writerow(csv_row)

def main():
    # Read URLs from the markdown file
    urls_file = 'data/YES_url_vajiram.md'
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f.readlines() if line.strip() and line.startswith('https://')]
    
    print(f"Found {len(urls)} URLs to process")
    
    all_articles = []
    
    for base_url in urls:
        print(f"Processing: {base_url}")
        
        # Extract all article links from the page
        article_links = extract_article_links(base_url)
        print(f"Found {len(article_links)} potential article links for {base_url}")
        
        for i, article_url in enumerate(article_links):
            print(f"  Processing article {i+1}/{len(article_links)}: {article_url}")
            
            # Extract content from the article
            article_data = extract_article_content(article_url)
            
            if article_data and article_data['content'] and len(article_data['content']) > 50:  # At least 50 chars
                all_articles.append((article_url, article_data))
                print(f"    Saved article: {article_data['title'][:50]}... ({len(article_data['content'])} chars)")
            else:
                print(f"    No significant content found for: {article_url}")
            
            # Be respectful to the server
            time.sleep(0.5)
    
    # Save results in CSV format
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    csv_output = output_dir / 'vajiram_articles_formatted.csv'
    
    save_articles_to_csv_format(all_articles, csv_output)
    
    print(f"\nScraping completed!")
    print(f"Total articles scraped: {len(all_articles)}")
    print(f"CSV output saved to: {csv_output}")
    
    # Print sample of what was scraped
    if all_articles:
        print("\nSample of scraped articles:")
        for i, (url, article) in enumerate(all_articles[:3]):
            print(f"{i+1}. Title: {article['title'][:60]}...")
            print(f"   URL: {url}")
            print(f"   Content length: {len(article['content'])} characters")
            print()

if __name__ == "__main__":
    main()
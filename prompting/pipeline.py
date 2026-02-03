#!/usr/bin/env python3
"""
Main pipeline for fetching and classifying UPSC-relevant news from Indian Express RSS.
"""

import os
import json
import csv
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

try:
    from .rss_parser import parse_rss_feed, get_feed_titles_only, Article
    from .classifier import UPSCClassifier, find_working_model
    from .config import RSS_FEEDS, OUTPUT_DIR, DATA_DIR, REQUEST_DELAY, DEFAULT_MODEL
except ImportError:  # allows running as a standalone script
    from rss_parser import parse_rss_feed, get_feed_titles_only, Article
    from classifier import UPSCClassifier, find_working_model
    from config import RSS_FEEDS, OUTPUT_DIR, DATA_DIR, REQUEST_DELAY, DEFAULT_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dirs():
    """Ensure output directories exist."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    Path(DATA_DIR).mkdir(exist_ok=True)


def save_results_csv(articles: List[Dict], filepath: str):
    """Save classified articles to CSV."""
    if not articles:
        return
    
    fieldnames = ['title', 'link', 'published', 'classification', 'summary']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            writer.writerow({
                'title': article.get('title', ''),
                'link': article.get('link', ''),
                'published': article.get('published', ''),
                'classification': article.get('classification', ''),
                'summary': article.get('summary', '')[:500]  # Truncate summary
            })
    
    logger.info(f"Saved {len(articles)} articles to {filepath}")


def save_results_json(articles: List[Dict], filepath: str):
    """Save classified articles to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} articles to {filepath}")


def save_results_txt(classified_articles: List[Dict], filepath: str):
    """Save classification results (article title, reasoning, YES/NO) to a txt file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("UPSC NEWS CLASSIFICATION RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        yes_count = sum(1 for a in classified_articles if a.get('classification') == 'YES')
        no_count = len(classified_articles) - yes_count
        
        f.write(f"Total Articles: {len(classified_articles)}\n")
        f.write(f"UPSC Relevant (YES): {yes_count}\n")
        f.write(f"Not Relevant (NO): {no_count}\n")
        f.write(f"Relevance Rate: {yes_count/len(classified_articles)*100:.1f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS (Article | Reasoning | Classification)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, article in enumerate(classified_articles, 1):
            classification = article.get('classification', 'N/A')
            title = article.get('title', 'No Title')
            reasoning = article.get('reasoning', 'No reasoning provided')
            
            marker = "✅ YES" if classification == "YES" else "❌ NO"
            f.write(f"{i}. {title}\n")
            f.write(f"   Reasoning: {reasoning}\n")
            f.write(f"   Classification: {marker}\n")
            f.write("-" * 80 + "\n\n")
        
        # Summary section for YES articles only
        f.write("\n" + "=" * 80 + "\n")
        f.write("UPSC RELEVANT ARTICLES ONLY (YES)\n")
        f.write("=" * 80 + "\n\n")
        
        yes_articles = [a for a in classified_articles if a.get('classification') == 'YES']
        for i, article in enumerate(yes_articles, 1):
            f.write(f"{i}. {article.get('title', 'No Title')}\n")
            f.write(f"   Reasoning: {article.get('reasoning', '')}\n\n")
    
    logger.info(f"Saved classification results to {filepath}")


def run_quick_preview(feed_url: str):
    """Quick preview of RSS feed without classification."""
    print("\n" + "=" * 70)
    print("📰 RSS FEED PREVIEW (No Classification)")
    print("=" * 70)
    
    titles = get_feed_titles_only(feed_url)
    
    for i, item in enumerate(titles, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   🔗 {item['link']}")
        if item['summary']:
            print(f"   📝 {item['summary'][:150]}...")
    
    print(f"\n📊 Total articles in feed: {len(titles)}")
    return titles


def run_classification_pipeline(
    feed_url: str,
    model: str = None,
    fetch_content: bool = True,
    save_all: bool = False
):
    """
    Main pipeline to fetch articles and classify them.
    
    Args:
        feed_url: RSS feed URL
        model: Model to use for classification
        fetch_content: Whether to fetch full article content
        save_all: Save all articles or just UPSC-relevant ones
    """
    ensure_dirs()
    
    # Find working model if not specified
    if model is None:
        logger.info("Finding a working model...")
        model = find_working_model()
        if not model:
            logger.error("No working model found!")
            return
    
    logger.info(f"Using model: {model}")
    classifier = UPSCClassifier(model=model)
    
    # Parse RSS feed
    print("\n" + "=" * 70)
    print("🔄 FETCHING ARTICLES FROM RSS FEED")
    print("=" * 70)
    
    articles = parse_rss_feed(feed_url, fetch_content=fetch_content, delay=REQUEST_DELAY)
    
    if not articles:
        logger.error("No articles found!")
        return
    
    # Classify articles
    print("\n" + "=" * 70)
    print("🤖 CLASSIFYING ARTICLES FOR UPSC RELEVANCE")
    print("=" * 70)
    
    classified_articles = []
    upsc_relevant = []
    
    for i, article in enumerate(articles, 1):
        text = article.get_text_for_classification()
        
        logger.info(f"[{i}/{len(articles)}] Classifying: {article.title[:50]}...")
        
        classification, reasoning = classifier.classify(text)
        
        article_data = {
            'title': article.title,
            'link': article.link,
            'published': article.published,
            'summary': article.description,
            'content': article.content[:1000] if article.content else '',
            'classification': classification,
            'reasoning': reasoning
        }
        
        classified_articles.append(article_data)
        
        if classification == "YES":
            upsc_relevant.append(article_data)
            print(f"   ✅ YES - {reasoning[:60]}..." if reasoning and len(reasoning) > 60 else f"   ✅ YES - {reasoning}")
        else:
            print(f"   ❌ NO - {reasoning[:60]}..." if reasoning and len(reasoning) > 60 else f"   ❌ NO - {reasoning}")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the prompting folder path for saving results
    prompting_dir = Path(__file__).parent
    
    # Save classification results to txt file in prompting folder
    results_txt = prompting_dir / f"classification_results_{timestamp}.txt"
    save_results_txt(classified_articles, str(results_txt))
    
    # Save results to data directory
    if save_all:
        all_csv = os.path.join(DATA_DIR, f"all_articles_{timestamp}.csv")
        all_json = os.path.join(DATA_DIR, f"all_articles_{timestamp}.json")
        save_results_csv(classified_articles, all_csv)
        save_results_json(classified_articles, all_json)
    
    # Always save UPSC-relevant articles
    upsc_csv = os.path.join(DATA_DIR, f"upsc_relevant_{timestamp}.csv")
    upsc_json = os.path.join(DATA_DIR, f"upsc_relevant_{timestamp}.json")
    save_results_csv(upsc_relevant, upsc_csv)
    save_results_json(upsc_relevant, upsc_json)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total articles processed: {len(articles)}")
    print(f"UPSC Relevant (YES):      {len(upsc_relevant)}")
    print(f"Not Relevant (NO):        {len(classified_articles) - len(upsc_relevant)}")
    print(f"Relevance Rate:           {len(upsc_relevant)/len(articles)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("✅ UPSC RELEVANT ARTICLES")
    print("=" * 70)
    
    for i, article in enumerate(upsc_relevant, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   🔗 {article['link']}")
        if article['summary']:
            print(f"   📝 {article['summary'][:150]}...")
    
    return upsc_relevant


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and classify UPSC-relevant news from RSS feeds"
    )
    parser.add_argument(
        "--feed", "-f",
        default="https://indianexpress.com/feed/",
        help="RSS feed URL to parse"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Quick preview without classification"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model to use for classification"
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="Don't fetch full article content (faster but less accurate)"
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all articles, not just UPSC-relevant ones"
    )
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connection and find working model"
    )
    
    args = parser.parse_args()
    
    if args.test_api:
        print("\n" + "=" * 70)
        print("🔌 TESTING API CONNECTION")
        print("=" * 70)
        working_model = find_working_model()
        if working_model:
            print(f"\n✅ Working model found: {working_model}")
        else:
            print("\n❌ No working model found!")
        return
    
    if args.preview:
        run_quick_preview(args.feed)
    else:
        run_classification_pipeline(
            feed_url=args.feed,
            model=args.model,
            fetch_content=not args.no_content,
            save_all=args.save_all
        )


if __name__ == "__main__":
    main()

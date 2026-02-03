"""
Full Classification and Summarization Pipeline for UPSC News Digest.
Processes all scraped articles and saves YES articles and summaries to separate files.
"""

import json
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "output"

try:
    from .newspaper_scraper import Article
    from .classifier import UPSCClassifier
    from .summarizer import UPSCSummarizer
except ImportError:  # allows running as a standalone script
    from newspaper_scraper import Article
    from classifier import UPSCClassifier
    from summarizer import UPSCSummarizer
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_articles_from_json():
    """Load articles from the JSON files created by scraping."""
    articles = []

    # Load IE articles
    ie_file = OUTPUT_DIR / "ie_articles_list.json"
    if ie_file.exists():
        with open(ie_file, 'r', encoding='utf-8') as f:
            ie_data = json.load(f)
            for item in ie_data['articles']:
                articles.append(Article(
                    title=item['title'],
                    link=item['url'],
                    source='indian_express',
                    section=item['section']
                ))
        logger.info(f'Loaded {len(ie_data["articles"])} IE articles')

    # Load Hindu articles
    hindu_file = OUTPUT_DIR / "hindu_articles_list.json"
    if hindu_file.exists():
        with open(hindu_file, 'r', encoding='utf-8') as f:
            hindu_data = json.load(f)
            for item in hindu_data['articles']:
                articles.append(Article(
                    title=item['title'],
                    link=item['url'],
                    source='the_hindu',
                    section=item['section']
                ))
        logger.info(f'Loaded {len(hindu_data["articles"])} Hindu articles')

    logger.info(f'Total articles loaded: {len(articles)}')
    return articles

def main():
    """Main pipeline execution."""
    # Initialize components
    classifier = UPSCClassifier(model='DeepSeek-V3.2')
    summarizer = UPSCSummarizer(model='DeepSeek-V3.2')

    # Load articles
    all_articles = load_articles_from_json()

    # Process ALL articles
    logger.info(f'Processing ALL {len(all_articles)} articles...')

    # Classify articles
    logger.info('Starting classification...')
    yes_articles = []
    classification_results = []

    for i, article in enumerate(all_articles):
        logger.info(f'Classifying {i+1}/{len(all_articles)}: {article.title[:50]}...')

        text = article.get_text_for_classification()
        classification, reasoning = classifier.classify(text)

        classification_results.append({
            'title': article.title,
            'link': article.link,
            'source': article.source,
            'section': article.section,
            'classification': classification,
            'reasoning': reasoning
        })

        if classification == 'YES':
            yes_articles.append(article)

        time.sleep(0.5)

    logger.info(f'Found {len(yes_articles)} UPSC-relevant articles')

    # Save YES article titles
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    yes_titles_file = OUTPUT_DIR / f"yes_article_titles_{timestamp}.txt"
    with open(yes_titles_file, 'w', encoding='utf-8') as f:
        f.write('UPSC-RELEVANT ARTICLE TITLES\n')
        f.write('=' * 50 + '\n\n')
        for i, article in enumerate(yes_articles, 1):
            f.write(f'{i}. {article.title}\n')
            f.write(f'   Source: {article.source} | Section: {article.section}\n')
            f.write(f'   URL: {article.link}\n\n')

    logger.info(f'Saved YES article titles to: {yes_titles_file}')

    # Summarize YES articles
    logger.info('Starting summarization...')
    summaries = []

    for i, article in enumerate(yes_articles):
        logger.info(f'Summarizing {i+1}/{len(yes_articles)}: {article.title[:50]}...')
        summary = summarizer.summarize_article(article)
        if summary:
            summaries.append({
                'title': article.title,
                'link': article.link,
                'source': article.source,
                'section': article.section,
                'summary': summary
            })
        time.sleep(1)

    # Save summaries
    summaries_file = OUTPUT_DIR / f"yes_article_summaries_{timestamp}.txt"
    with open(summaries_file, 'w', encoding='utf-8') as f:
        f.write('UPSC-RELEVANT ARTICLE SUMMARIES\n')
        f.write('=' * 50 + '\n\n')
        for i, summary_data in enumerate(summaries, 1):
            f.write(f'{i}. {summary_data["title"]}\n')
            f.write(f'   Source: {summary_data["source"]} | Section: {summary_data["section"]}\n')
            f.write(f'   URL: {summary_data["link"]}\n')
            f.write(f'   Summary:\n{summary_data["summary"]}\n')
            f.write('-' * 80 + '\n\n')

    logger.info(f'Saved YES article summaries to: {summaries_file}')

    print('\n' + '=' * 70)
    print('FULL CLASSIFICATION & SUMMARIZATION COMPLETE')
    print('=' * 70)
    print(f'Total articles processed: {len(all_articles)}')
    print(f'UPSC-relevant articles: {len(yes_articles)}')
    print(f'Summaries generated: {len(summaries)}')
    print()
    print('Files created:')
    print(f'  - {yes_titles_file}')
    print(f'  - {summaries_file}')

if __name__ == "__main__":
    main()
"""
UPSC Daily Digest Pipeline.
Scrapes newspapers → Classifies → Summarizes → Outputs for Telegram.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .newspaper_scraper import scrape_all_newspapers, IndianExpressScraper, TheHinduScraper, Article
    from .classifier import UPSCClassifier
    from .summarizer import UPSCSummarizer, format_for_telegram
except ImportError:  # allows running as a standalone script
    from newspaper_scraper import scrape_all_newspapers, IndianExpressScraper, TheHinduScraper, Article
    from classifier import UPSCClassifier
    from summarizer import UPSCSummarizer, format_for_telegram

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UPSCDigestPipeline:
    """Complete pipeline for UPSC daily digest."""
    
    def __init__(self, model: str = "DeepSeek-V3.2"):
        self.model = model
        self.classifier = UPSCClassifier(model=model)
        self.summarizer = UPSCSummarizer(model=model)
        self.ie_scraper = IndianExpressScraper()
        self.th_scraper = TheHinduScraper()
        self.output_dir = Path(__file__).parent / "output"
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self, fetch_content: bool = True, max_articles: int = 50) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            fetch_content: Whether to fetch full article content
            max_articles: Maximum articles to process
            
        Returns:
            Dictionary with results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%d %B %Y")
        
        logger.info("="*70)
        logger.info("UPSC DAILY DIGEST PIPELINE")
        logger.info(f"Date: {date_str}")
        logger.info("="*70)
        
        results = {
            "date": date_str,
            "timestamp": timestamp,
            "total_scraped": 0,
            "total_relevant": 0,
            "articles": [],
            "summaries": []
        }
        
        # Step 1: Scrape articles
        logger.info("\n📰 STEP 1: Scraping newspapers...")
        all_articles = scrape_all_newspapers()
        results["total_scraped"] = len(all_articles)
        logger.info(f"Total articles scraped: {len(all_articles)}")
        
        if not all_articles:
            logger.error("No articles scraped. Exiting.")
            return results
        
        # Limit articles if needed
        articles_to_process = all_articles[:max_articles]
        
        # Step 2: Fetch content for articles (if needed)
        if fetch_content:
            logger.info("\n📖 STEP 2: Fetching article content...")
            for i, article in enumerate(articles_to_process):
                logger.info(f"Fetching {i+1}/{len(articles_to_process)}: {article.title[:50]}...")
                if article.source == 'indian_express':
                    self.ie_scraper.fetch_article_content(article, delay=0.5)
                else:
                    self.th_scraper.fetch_article_content(article, delay=0.5)
        
        # Step 3: Classify articles
        logger.info("\n🔍 STEP 3: Classifying articles for UPSC relevance...")
        relevant_articles = []
        classification_results = []
        
        for i, article in enumerate(articles_to_process):
            logger.info(f"Classifying {i+1}/{len(articles_to_process)}: {article.title[:50]}...")
            
            # Get text for classification
            text = article.get_text_for_classification()
            
            # Classify
            classification, reasoning = self.classifier.classify(text)
            
            classification_results.append({
                "title": article.title,
                "link": article.link,
                "source": article.source,
                "section": article.section,
                "classification": classification,
                "reasoning": reasoning
            })
            
            if classification == "YES":
                relevant_articles.append(article)
                results["articles"].append({
                    "title": article.title,
                    "link": article.link,
                    "source": article.source,
                    "section": article.section,
                    "reasoning": reasoning
                })
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        
        results["total_relevant"] = len(relevant_articles)
        logger.info(f"\n✅ Found {len(relevant_articles)} UPSC-relevant articles")
        
        # Save classification results
        classification_file = self.output_dir / f"classification_{timestamp}.txt"
        self._save_classification_results(classification_results, classification_file)
        
        # Step 4: Summarize relevant articles
        logger.info("\n📝 STEP 4: Summarizing relevant articles...")
        summaries = []
        
        for i, article in enumerate(relevant_articles):
            logger.info(f"Summarizing {i+1}/{len(relevant_articles)}: {article.title[:50]}...")
            summary = self.summarizer.summarize_article(article)
            if summary:
                summaries.append((article.title, summary))
                results["summaries"].append({
                    "title": article.title,
                    "link": article.link,
                    "summary": summary
                })
            time.sleep(1)  # Delay between API calls
        
        # Step 5: Format for Telegram
        logger.info("\n📱 STEP 5: Formatting for Telegram...")
        telegram_digest = format_for_telegram(summaries, date_str)
        
        # Save outputs
        telegram_file = self.output_dir / f"telegram_digest_{timestamp}.txt"
        with open(telegram_file, 'w') as f:
            f.write(telegram_digest)
        logger.info(f"Saved Telegram digest to: {telegram_file}")
        
        # Save JSON results
        json_file = self.output_dir / f"digest_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON results to: {json_file}")
        
        # Print summary
        self._print_summary(results, telegram_digest)
        
        return results
    
    def _save_classification_results(self, results: list, filepath: Path):
        """Save classification results to file."""
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UPSC ARTICLE CLASSIFICATION RESULTS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")
            
            yes_count = sum(1 for r in results if r['classification'] == 'YES')
            no_count = len(results) - yes_count
            
            f.write(f"Total Articles: {len(results)}\n")
            f.write(f"UPSC Relevant (YES): {yes_count}\n")
            f.write(f"Not Relevant (NO): {no_count}\n")
            f.write("\n" + "="*70 + "\n\n")
            
            for i, r in enumerate(results, 1):
                f.write(f"Article {i}:\n")
                f.write(f"Title: {r['title']}\n")
                f.write(f"Link: {r['link']}\n")
                f.write(f"Source: {r['source']} | Section: {r['section']}\n")
                f.write(f"Classification: {r['classification']}\n")
                f.write(f"Reasoning: {r['reasoning']}\n")
                f.write("-"*50 + "\n\n")
        
        logger.info(f"Saved classification results to: {filepath}")
    
    def _print_summary(self, results: dict, telegram_digest: str):
        """Print final summary."""
        print("\n" + "="*70)
        print("📊 PIPELINE SUMMARY")
        print("="*70)
        print(f"📅 Date: {results['date']}")
        print(f"📰 Total Articles Scraped: {results['total_scraped']}")
        print(f"✅ UPSC Relevant Articles: {results['total_relevant']}")
        print(f"📝 Summaries Generated: {len(results['summaries'])}")
        print("="*70)
        
        print("\n📱 TELEGRAM DIGEST PREVIEW (First 2000 chars):")
        print("-"*70)
        print(telegram_digest[:2000])
        if len(telegram_digest) > 2000:
            print("\n... [truncated] ...")
        print("-"*70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="UPSC Daily Digest Pipeline")
    parser.add_argument("--max-articles", type=int, default=30,
                        help="Maximum articles to process (default: 30)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Don't fetch full article content")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.2",
                        help="Model to use for classification and summarization")
    
    args = parser.parse_args()
    
    pipeline = UPSCDigestPipeline(model=args.model)
    results = pipeline.run(
        fetch_content=not args.no_fetch,
        max_articles=args.max_articles
    )
    
    return results


if __name__ == "__main__":
    main()

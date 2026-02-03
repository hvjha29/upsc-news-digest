"""
Article Summarizer for UPSC using DeepSeek-V3.2.
Creates Mains-Ready Flashcards for Telegram.
"""

from openai import OpenAI
import logging
from typing import Optional
from pathlib import Path

try:
    from .config import API_KEY, BASE_URL
except ImportError:  # allows running as a standalone script
    from config import API_KEY, BASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UPSCSummarizer:
    """Summarizes articles into UPSC Mains-Ready Flashcards."""
    
    def __init__(self, model: str = "DeepSeek-V3.2"):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
        self.model = model
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load the summarization prompt template."""
        prompt_path = Path(__file__).parent / "summarisation_prompt.txt"
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("summarisation_prompt.txt not found!")
            return """You are a UPSC expert. Summarize the article for UPSC aspirants.
Include:
- Why in news
- Key takeaways
- Mains relevant points
- Keywords and constitutional links"""
    
    def summarize(self, title: str, content: str) -> str:
        """
        Summarize an article for UPSC Mains preparation.
        
        Args:
            title: Article title
            content: Article content/text
            
        Returns:
            Formatted summary for Telegram
        """
        if not content or len(content) < 100:
            logger.warning(f"Content too short for article: {title}")
            return ""
        
        user_prompt = f"""
Article Title: {title}

Article Content:
{content[:4000]}

Please create a UPSC Mains-Ready Flashcard for this article.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt_template},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Summarized: {title[:50]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing '{title}': {e}")
            return ""
    
    def summarize_article(self, article) -> str:
        """
        Summarize an Article object.
        
        Args:
            article: Article dataclass with title, content, summary
            
        Returns:
            Formatted summary
        """
        content = article.content if article.content else article.summary
        if not content:
            content = article.title
        return self.summarize(article.title, content)


def format_for_telegram(summaries: list, date_str: str = "") -> str:
    """
    Format multiple summaries into a Telegram digest.
    
    Args:
        summaries: List of (title, summary) tuples
        date_str: Date string for the digest
        
    Returns:
        Complete Telegram message
    """
    header = f"""
📰 *UPSC Daily News Digest*
📅 {date_str}
━━━━━━━━━━━━━━━━━━━━━━━

"""
    
    content_parts = [header]
    
    for i, (title, summary) in enumerate(summaries, 1):
        if summary:
            content_parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━\n")
            content_parts.append(summary)
            content_parts.append("\n\n")
    
    footer = """
━━━━━━━━━━━━━━━━━━━━━━━
📱 *UPSC News Digest* | Stay ahead in your preparation
🔔 Daily updates at 8 AM
"""
    
    content_parts.append(footer)
    return "".join(content_parts)


if __name__ == "__main__":
    # Test summarizer
    summarizer = UPSCSummarizer()
    
    test_content = """
    The Supreme Court on Tuesday upheld the constitutional validity of the GST compensation cess, 
    ruling that states do not have a constitutional right to receive compensation from the Centre 
    beyond the five-year transition period. The bench, headed by Justice B.V. Nagarathna, said the 
    compensation mechanism under the GST framework was a policy decision and not a constitutional 
    mandate. The court emphasized that fiscal federalism requires cooperation between the Centre 
    and states, and that the GST Council's recommendations are not binding but have persuasive value.
    The judgment noted that Article 279A establishes the GST Council as a constitutional body 
    aimed at harmonizing indirect taxation across the country.
    """
    
    summary = summarizer.summarize(
        title="SC upholds validity of GST compensation cess",
        content=test_content
    )
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(summary)

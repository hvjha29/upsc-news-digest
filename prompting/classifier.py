"""
UPSC Relevance Classifier using LLM API.
Classifies news articles as YES/NO based on UPSC CSE relevance.
"""

import time
import logging
from typing import Optional, Tuple
from pathlib import Path

# Allow running as a standalone script: `python prompting/classifier.py`
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from prompting.config import API_KEY, BASE_URL, DEFAULT_MODEL, AVAILABLE_MODELS, MAX_TEXT_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UPSCClassifier:
    """Classifier for UPSC-relevant news articles."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model = model
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        template_path = Path(__file__).parent / "content_auditor_prompt.txt"
        with open(template_path, 'r') as f:
            return f.read()
    
    def _truncate_text(self, text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
        """Truncate text to max length while preserving meaning."""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at a sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:
            return truncated[:last_period + 1]
        return truncated + "..."
    
    def classify(self, text: str, retries: int = 3) -> Tuple[str, Optional[str]]:
        """
        Classify text as YES or NO for UPSC relevance.
        
        Args:
            text: The article text to classify
            retries: Number of retries on failure
            
        Returns:
            Tuple of (classification, reasoning)
            classification is "YES", "NO", or "ERROR"
            reasoning is the model's reasoning for the classification
        """
        truncated_text = self._truncate_text(text)
        prompt = self.prompt_template.replace("{text}", truncated_text)
        
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert UPSC CSE Content Auditor. Follow the criteria and output format exactly as specified."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,  # Zero temperature for deterministic output
                    max_tokens=150,   # Allow enough tokens for reasoning + answer
                )
                
                raw_response = response.choices[0].message.content.strip()
                
                # Parse the response to extract reasoning and answer
                reasoning = ""
                classification = "NO"  # Default
                
                # Try to extract reasoning
                if "Reasoning:" in raw_response:
                    parts = raw_response.split("Answer:")
                    if len(parts) >= 2:
                        reasoning = parts[0].replace("Reasoning:", "").strip()
                        answer_part = parts[1].strip().upper()
                    else:
                        reasoning = raw_response.replace("Reasoning:", "").strip()
                        answer_part = raw_response.upper()
                else:
                    reasoning = raw_response
                    answer_part = raw_response.upper()
                
                # Extract YES or NO from answer part
                if "YES" in answer_part and "NO" not in answer_part:
                    classification = "YES"
                elif "NO" in answer_part:
                    classification = "NO"
                elif "YES" in answer_part and "NO" in answer_part:
                    # Both present - check which comes last
                    yes_pos = answer_part.rfind("YES")
                    no_pos = answer_part.rfind("NO")
                    classification = "YES" if yes_pos > no_pos else "NO"
                
                return classification, reasoning
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return "ERROR", None
    
    def classify_batch(self, texts: list, delay: float = 1.0) -> list:
        """
        Classify multiple texts with delay between calls.
        
        Args:
            texts: List of text strings to classify
            delay: Delay in seconds between API calls
            
        Returns:
            List of (classification, raw_response) tuples
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Classifying {i+1}/{len(texts)}...")
            result = self.classify(text)
            results.append(result)
            if i < len(texts) - 1:
                time.sleep(delay)
        return results
    
    def test_connection(self) -> bool:
        """Test if the API connection works."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Say hello"}
                ],
                max_tokens=10,
            )
            logger.info(f"API connection successful. Model: {self.model}")
            logger.info(f"Test response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return False


def find_working_model() -> Optional[str]:
    """Find a working model from the available models list."""
    for model in AVAILABLE_MODELS:
        logger.info(f"Testing model: {model}")
        classifier = UPSCClassifier(model=model)
        if classifier.test_connection():
            return model
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing UPSC Classifier")
    print("=" * 60)
    
    # Find a working model
    print("\nFinding a working model...")
    working_model = find_working_model()
    
    if not working_model:
        print("No working model found!")
        exit(1)
    
    print(f"\nUsing model: {working_model}")
    
    classifier = UPSCClassifier(model=working_model)
    
    # Test with sample texts
    test_texts = [
        """Title: Union Cabinet approves National Green Hydrogen Mission
        The Union Cabinet has approved the National Green Hydrogen Mission with an outlay of Rs 19,744 crore. 
        The mission aims to make India a global hub for production, utilization and export of Green Hydrogen 
        and its derivatives. It will help India meet its climate commitments.""",
        
        """Title: Virat Kohli scores century in IPL match
        Star cricketer Virat Kohli scored a brilliant century in yesterday's IPL match against 
        Mumbai Indians. The crowd went wild as he hit six sixes in the final over.""",
        
        """Title: Supreme Court upholds reservation in promotions
        The Supreme Court has upheld the validity of reservation in promotions for SC/ST candidates 
        in government jobs. The court ruled that the policy does not violate the basic structure 
        of the Constitution.""",
    ]
    
    print("\n" + "=" * 60)
    print("Classification Results")
    print("=" * 60)
    
    for text in test_texts:
        title = text.split('\n')[0].replace('Title:', '').strip()
        result, raw = classifier.classify(text)
        print(f"\n📰 {title[:60]}...")
        print(f"   Classification: {result}")

import re

# preprocessing/cleaner.py

def simple_clean(text: str) -> str:
    """Clean text by removing whitespace, footers, and other common issues."""
    if not text:
        return ""
    
    # Remove repeated whitespace
    s = re.sub(r"\s+", " ", text).strip()
    
    # Remove small editorial footers
    s = re.sub(r"Read more: .*", "", s)
    s = re.sub(r"Click here: .*", "", s)
    
    # Remove URLs
    s = re.sub(r"https?://\S+", "", s)
    
    # Remove email addresses
    s = re.sub(r"\S+@\S+", "", s)
    
    # Remove HTML tags
    s = re.sub(r"<[^>]+>", "", s)
    
    # Remove extra punctuation at the end
    s = re.sub(r"([.!?])\1+", r"\1", s)
    
    # Remove special characters but keep basic punctuation
    s = re.sub(r"[^\w\s.!?,-]", "", s)
    
    # Final whitespace cleanup
    s = re.sub(r"\s+", " ", s).strip()
    
    return s
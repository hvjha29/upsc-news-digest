"""
Root-level configuration for the UPSC News Digest pipeline.
Values are read from environment variables when available (for CI),
with local defaults for development.
"""
import os

# API Configuration - read from env for CI, fallback for local dev
API_KEY = os.getenv("DEEPSEEK_API_KEY", "LsWy_vOUjn26mHhVI9i-e92d")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://cloud.olakrutrim.com/v1/")

# Available models (in order of preference for classification)
AVAILABLE_MODELS = [
    "DeepSeek-V3.2",           # Good reasoning - primary choice
    "Llama-3.3-70B-Instruct",  # Strong instruction following
    "gpt-oss-120b",            # Large model
    "gpt-oss-120b-at",         # Alternate
    "nemotron-ultra",          # NVIDIA model
]

# Default model to use
DEFAULT_MODEL = "DeepSeek-V3.2"

# Classification settings
MAX_TEXT_LENGTH = 4000  # Max characters to send for classification

# RSS Feed URLs (kept for compatibility with prompting.config users)
RSS_FEEDS = {
    "indian_express_main": "https://indianexpress.com/feed/",
    "indian_express_explained": "https://indianexpress.com/section/explained/feed/",
    "indian_express_india": "https://indianexpress.com/section/india/feed/",
    "indian_express_political_pulse": "https://indianexpress.com/section/political-pulse/feed/",
}

# Output settings
OUTPUT_DIR = "output"
DATA_DIR = "data"

# Request settings
REQUEST_DELAY = 1.0   # seconds between API calls
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BATCH_SIZE = 10       # Number of articles to process before saving checkpoint

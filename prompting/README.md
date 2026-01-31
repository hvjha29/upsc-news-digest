# UPSC News Digest - Prompting Pipeline

This folder contains scripts for fetching and classifying UPSC-relevant news articles using LLM-based prompting (no fine-tuning required).

## Overview

The pipeline:
1. Fetches articles from RSS feeds (Indian Express, etc.)
2. Extracts title, summary, and full content
3. Classifies each article using LLM API (Krutrim Cloud)
4. Saves UPSC-relevant articles for further processing

## Files

- `config.py` - Configuration (API keys, models, settings)
- `prompt_template.txt` - Classification prompt template
- `rss_parser.py` - RSS feed parser and content extractor
- `classifier.py` - LLM-based UPSC relevance classifier
- `pipeline.py` - Main pipeline script

## Usage

### Quick Preview (No Classification)
```bash
python pipeline.py --preview
```

### Test API Connection
```bash
python pipeline.py --test-api
```

### Full Classification Pipeline
```bash
# Default: Indian Express main RSS
python pipeline.py

# Custom RSS feed
python pipeline.py --feed "https://indianexpress.com/section/explained/feed/"

# Faster mode (no full content fetch)
python pipeline.py --no-content

# Save all articles (not just relevant ones)
python pipeline.py --save-all

# Specify model
python pipeline.py --model "Qwen3-32B"
```

## Output

Results are saved in the `data/` directory:
- `upsc_relevant_YYYYMMDD_HHMMSS.csv` - UPSC-relevant articles
- `upsc_relevant_YYYYMMDD_HHMMSS.json` - Same in JSON format
- `all_articles_YYYYMMDD_HHMMSS.*` - All articles (with `--save-all`)

## API Configuration

Using Krutrim Cloud API with the following models:
- Qwen3-32B (default)
- Llama-3.3-70B-Instruct
- DeepSeek-V3.2
- gpt-oss-120b
- nemotron-ultra

## Classification Criteria

Articles are marked **YES** (relevant) if they relate to:
- Government policies, laws, schemes, regulations
- Constitutional matters, Supreme Court judgments
- Economy: budgets, RBI reports, GDP, taxation
- International relations affecting India
- Science & Tech with policy relevance

Articles are marked **NO** if they relate to:
- Local city news without national relevance
- Crimes, accidents, lifestyle, entertainment, sports
- Stock market movements, company news
- International news unrelated to India

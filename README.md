# UPSC News Digest

Small prototype to ingest news / gov PDFs, summarize for UPSC candidates, and deliver via website / Telegram.

## Quickstart (local)

1. Clone:
   ```bash
   git clone git@github.com:YOUR_USERNAME/upsc-news-digest.git
   cd upsc-news-digest
```

## Classification Script

The repository includes a script `classify_csv_hf.py` that uses the Hugging Face API to classify articles in the CSV file as YES/NO based on relevance. The script uses the DeepSeek-V3.2 model.

To use the classification script:

1. Get a Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens)
2. Set the environment variable:
   ```bash
   export HF_API_TOKEN=your_token_here
   ```
3. Run the script:
   ```bash
   python classify_csv_hf.py
   ```

The script will create a new file `data/iex_explained_classified.csv` with an additional column for the classification results. You can customize the classification prompt by modifying the `prompt_template` variable in the script.
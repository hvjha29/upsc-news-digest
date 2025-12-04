#!/usr/bin/env python3
"""
Create a simple relevance dataset from PYQs (labeled as YES) and news articles (labeled as NO).

This script creates a relevance dataset in the format:
{"id": "de1343a4-501e-4863-a86e-a02ea0fd8e9f", "text": "Though not very useful from the point of view...", "label": "YES", "source": "https://pwonlyias.com/mains-solved-papers-by-year/gs-paper-2013/"}

Label everything in data/pyqs_pwonly.csv as YES and everything in data/iex_entertainment.csv, 
data/iex_lifestyle.csv, data/iex_political_pulse.csv, data/iex_politics.csv, 
data/iex_sports.csv, and data/thh_articles.csv as NO.

Usage:
    python training/create_simple_relevance_dataset.py --out data/relevance_dataset.jsonl
"""

import argparse
import csv
import json
import os
import uuid
from pathlib import Path


def load_csv_data(file_path):
    """Load data from a CSV file and return a list of dictionaries."""
    if not Path(file_path).exists():
        print(f"Warning: {file_path} does not exist, skipping...")
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def create_relevance_dataset():
    """Create the relevance dataset with positive and negative examples."""
    # Define file paths
    pyq_file = "data/pyqs_pwonly.csv"
    news_files = [
        "data/iex_entertainment.csv",
        "data/iex_lifestyle.csv", 
        "data/iex_political_pulse.csv",
        "data/iex_politics.csv",
        "data/iex_sports.csv",
        "data/thh_articles.csv"
    ]
    
    dataset = []
    
    # Process PYQs (label as YES)
    print(f"Loading positive examples from {pyq_file}...")
    pyqs = load_csv_data(pyq_file)
    for row in pyqs:
        text = row.get("question_text", "").strip()
        source = row.get("source_url", "pyq_file")
        if text:
            dataset.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "label": "YES",
                "source": source
            })
    
    print(f"Added {len(pyqs)} positive examples (YES labels)")
    
    # Process news articles (label as NO)
    for news_file in news_files:
        print(f"Loading negative examples from {news_file}...")
        news_data = load_csv_data(news_file)
        for row in news_data:
            text = row.get("question_text", "").strip()
            source = row.get("source_url", news_file)
            if text:
                dataset.append({
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "label": "NO",
                    "source": source
                })
    
    total_news = sum(len(load_csv_data(f)) for f in news_files if Path(f).exists())
    print(f"Added {total_news} negative examples (NO labels)")
    
    return dataset


def save_dataset(dataset, output_path):
    """Save the dataset as JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved to {output_path} with {len(dataset)} total examples")


def main():
    parser = argparse.ArgumentParser(description="Create a simple relevance dataset")
    parser.add_argument("--out", default="data/relevance_dataset.jsonl", 
                        help="Output file path for the relevance dataset")
    
    args = parser.parse_args()
    
    dataset = create_relevance_dataset()
    save_dataset(dataset, args.out)


if __name__ == "__main__":
    main()
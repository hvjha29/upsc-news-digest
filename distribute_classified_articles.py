#!/usr/bin/env python3
"""
Distribute articles from classified CSV files to JSONL files based on their classification.
Articles with classification 'YES' go to relevance_dataset_yes.jsonl and those with 'NO'
go to relevance_dataset_no.jsonl
"""

import csv
import json
import os
from datetime import datetime
import uuid


def load_csv_to_dict_list(file_path: str) -> list:
    """Load a CSV file and return a list of dictionaries, each representing a row."""
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    return rows


def write_dict_to_jsonl(data: dict, file_path: str):
    """Write a single dictionary to a JSONL file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data) + '\n')


def process_csv_files(csv_files: list):
    """Process each CSV file and distribute articles based on classification."""
    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping...")
            continue

        print(f"Processing {file_path}...")
        rows = load_csv_to_dict_list(file_path)

        for row in rows:
            # Get classification from the classification column (last column)
            classification = row.get('classification', '').upper()
            text = row.get('question_text', '')

            # Create a unique ID for this entry
            entry_id = str(uuid.uuid4())[:12]

            # Prepare the entry
            entry = {
                'id': entry_id,
                'text': text,
                'label': classification,
                'source': file_path,
                'year': row.get('year', ''),
                'paper': row.get('paper', ''),
                'topic_hint': row.get('topic_hint', '')
            }

            # Write to appropriate file based on classification
            if classification == 'YES':
                write_dict_to_jsonl(entry, 'data/relevance_dataset_yes.jsonl')
            elif classification == 'NO':
                write_dict_to_jsonl(entry, 'data/relevance_dataset_no.jsonl')
            else:
                print(f"Unknown classification '{classification}' for row ID {row.get('id', 'unknown')}, skipping...")

        print(f"Finished processing {file_path}")


def main():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    # Clear existing content from the target files
    with open('data/relevance_dataset_yes.jsonl', 'w') as f:
        pass  # Just open and close to clear the file
    with open('data/relevance_dataset_no.jsonl', 'w') as f:
        pass  # Just open and close to clear the file

    # List of files to process
    csv_files = [
        'data/iex_explained_classified.csv',
        'data/thh_articles_yes_classified.csv'
    ]

    # Process the files
    process_csv_files(csv_files)

    print("Distribution completed!")


if __name__ == '__main__':
    main()
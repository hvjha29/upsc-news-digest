import csv
import json
import uuid
from pathlib import Path

def append_vajiram_to_relevance_dataset():
    # Process only the Vajiram articles CSV file and append to existing JSONL
    csv_file = "data/vajiram_articles_formatted.csv"
    output_file = "data/relevance_dataset_yes.jsonl"

    if not Path(csv_file).exists():
        print(f"Error: {csv_file} does not exist")
        return

    # Open in append mode to add to existing file
    with open(output_file, 'a', encoding='utf-8') as outfile:
        with open(csv_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                # Create a record with "YES" label for relevance
                record = {
                    "id": str(uuid.uuid4()),
                    "text": row.get('question_text', '') or row.get('topic_hint', ''),
                    "label": "YES",  # These are positive examples
                    "source": f"file:///{Path(csv_file).absolute()}"
                }

                # Write the record to the output file
                outfile.write(json.dumps(record) + '\n')

    print(f"Successfully appended Vajiram articles to {output_file}")

if __name__ == "__main__":
    append_vajiram_to_relevance_dataset()
import csv
import json
import uuid
from pathlib import Path

def create_relevance_dataset_yes():
    # List of CSV files to process, now including the Vajiram articles file
    csv_files = [
        "data/Environment_PT730_v2_notes.csv",
        "data/Geography_PT_730_notes.csv",
        "data/History_PT730_v2_notes.csv",
        "data/International_Relations_PT730_notes.csv",
        "data/upsc_polity_notes.csv",
        "data/Science_and_Technology_v5_1_notes.csv",
        "data/vajiram_articles_formatted.csv"  # Adding the Vajiram articles file
    ]

    output_file = "data/relevance_dataset_yes.jsonl"

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for csv_file in csv_files:
            if not Path(csv_file).exists():
                print(f"Warning: {csv_file} does not exist, skipping...")
                continue

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

    print(f"Successfully created {output_file} with positive examples from all CSV files")

if __name__ == "__main__":
    create_relevance_dataset_yes()
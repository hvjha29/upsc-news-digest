import csv
import json
import uuid
from pathlib import Path

def create_relevance_dataset_yes_append():
    # List of additional CSV files to process
    csv_files = [
        "data/rausias_yes.csv",
        "data/visionias_yes.csv"
    ]

    output_file = "data/relevance_dataset_yes.jsonl"

    with open(output_file, 'a', encoding='utf-8') as outfile:  # Open in append mode
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

    print(f"Successfully appended data to {output_file} from all CSV files")

if __name__ == "__main__":
    create_relevance_dataset_yes_append()
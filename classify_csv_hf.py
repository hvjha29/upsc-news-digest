import csv
import os
from huggingface_hub import InferenceApi
import time
from typing import List, Dict, Optional
import random


def classify_text_with_llm(text: str, inference_api: InferenceApi, prompt_template: str) -> str:
    """
    Classify text using an LLM hosted on Hugging Face.

    Args:
        text: The text to classify
        inference_api: The Hugging Face Inference API object
        prompt_template: Template for the prompt

    Returns:
        Classification result (YES/NO)
    """
    # Format the prompt with the text
    prompt = prompt_template.format(text=text)

    # Call the LLM
    response = inference_api(inputs=prompt)

    # Extract classification from response
    # Assuming the response contains text that we need to parse for YES/NO
    response_text = response[0]['generated_text'] if isinstance(response, list) else str(response)

    # Simple parsing logic - adjust based on the actual model output
    if 'YES' in response_text.upper():
        return 'YES'
    elif 'NO' in response_text.upper():
        return 'NO'
    else:
        # Return the most prominent answer or a default
        response_lower = response_text.lower()
        if 'yes' in response_lower:
            return 'YES'
        elif 'no' in response_lower:
            return 'NO'
        else:
            # Return the first occurrence of yes/no-like terms
            words = response_lower.split()
            for word in words:
                if word.startswith('yes'):
                    return 'YES'
                elif word.startswith('no'):
                    return 'NO'
            # Default to NO if uncertain
            print(f"Uncertain classification for: {text[:100]}...")
            return 'NO'


def classify_text_test_mode(text: str, prompt_template: str) -> str:
    """
    Simulate text classification without calling the LLM API.

    Args:
        text: The text to classify
        prompt_template: Template for the prompt (for reference)

    Returns:
        Simulated classification result (YES/NO)
    """
    # Simple heuristic for testing: classify based on keywords
    text_lower = text.lower()

    # Keywords that might indicate policy/governance topics
    policy_keywords = [
        'government', 'policy', 'politics', 'election', 'minister', 'law', 'legislation',
        'parliament', 'vote', 'voter', 'migration', 'urbanisation', 'migration',
        'international', 'relations', 'foreign', 'diplomacy', 'trade', 'economics',
        'finance', 'budget', 'court', 'legal', 'rights', 'social', 'welfare'
    ]

    # Keywords that might indicate non-policy topics
    non_policy_keywords = [
        'science', 'technology', 'health', 'medicine', 'biology', 'neuroscience',
        'vaccine', 'bacteria', 'tuberculosis', 'sports', 'music', 'entertainment',
        'art', 'culture', 'artificial intelligence', 'ai', 'sports', 'cricket',
        'football', 'movie', 'film', 'actor', 'science', 'research'
    ]

    # Count occurrences of policy vs non-policy keywords
    policy_score = sum(1 for keyword in policy_keywords if keyword in text_lower)
    non_policy_score = sum(1 for keyword in non_policy_keywords if keyword in text_lower)

    # If more policy-related keywords, classify as YES
    if policy_score > non_policy_score:
        return 'YES'
    elif non_policy_score > policy_score:
        return 'NO'
    else:
        # If equal or unclear, make a random choice to simulate uncertainty
        return random.choice(['YES', 'NO'])


def classify_csv(input_file: str, output_file: str, api_token: Optional[str] = None,
                 prompt_template: str = "Classify the following text as YES or NO based on relevance: {text}\n\nAnswer:",
                 test_mode: bool = False) -> None:
    """
    Classify a CSV file using an LLM.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        api_token: Hugging Face API token (optional)
        prompt_template: Template for the prompt to send to the LLM
        test_mode: If True, use test mode instead of calling the actual API
    """
    # Initialize the Hugging Face Inference API if not in test mode
    if not test_mode:
        model_id = "deepseek-ai/DeepSeek-V3.2"
        inference_api = InferenceApi(model_id, token=api_token)
    else:
        inference_api = None  # Not used in test mode

    # Read the input CSV file
    rows = []
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append(row)

    # Prepare output with a new column for classification
    fieldnames = reader.fieldnames + ['classification']

    # Open the output file
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each row
        for i, row in enumerate(rows):
            print(f"Processing row {i+1}/{len(rows)}...")

            # Get the text to classify (question_text column)
            text_to_classify = row.get('question_text', '')

            if text_to_classify.strip():  # Only classify non-empty text
                try:
                    if test_mode:
                        # Use test mode function
                        classification = classify_text_test_mode(text_to_classify, prompt_template)
                    else:
                        # Classify the text using the LLM
                        classification = classify_text_with_llm(text_to_classify, inference_api, prompt_template)

                    # Add classification to the row
                    row['classification'] = classification
                except Exception as e:
                    print(f"Error processing row {i+1}: {str(e)}")
                    row['classification'] = 'ERROR'

                    # Add a small delay to avoid rate limiting
                    if not test_mode:
                        time.sleep(1)
            else:
                row['classification'] = 'EMPTY'

            # Write the row to the output file
            writer.writerow(row)

    print(f"Classification completed. Output saved to {output_file}")


def load_prompt_template(file_path: str = 'prompt_template.txt') -> str:
    """
    Load the prompt template from an external file.

    Args:
        file_path: Path to the prompt template file

    Returns:
        The content of the prompt template file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Prompt template file {file_path} not found. Using default template.")
        return "Classify the following text as YES or NO based on relevance: {text}\n\nAnswer:"


def main():
    # File paths
    input_file = 'data/iex_explained.csv'
    output_file = 'data/iex_explained_classified.csv'

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found!")
        return

    # Get API token from environment variable or user input
    api_token = os.getenv('HF_API_TOKEN')
    if not api_token:
        print("HF_API_TOKEN not found. Running in test mode...")

        # Load prompt template from external file
        prompt_template = load_prompt_template()

        classify_csv(input_file, output_file, api_token=None, prompt_template=prompt_template, test_mode=True)
        return

    # Load prompt template from external file
    prompt_template = load_prompt_template()

    # Run the classification in normal mode
    classify_csv(input_file, output_file, api_token, prompt_template)


if __name__ == "__main__":
    main()
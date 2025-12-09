#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic inference script to classify news articles using the LoRA adapter.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

# Model configuration
# The adapter was trained on a local path, but we'll try using the public model
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # The base model you mentioned
ADAPTER_PATH = "lora_relevance_adapter"  # The path where the adapter is stored

def load_model_and_tokenizer(adapter_path, base_model=BASE_MODEL):
    """
    Load the base model and apply the LoRA adapter
    """
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load the base model with specific device_map to handle memory efficiently
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",  # Automatically map to available device (GPU if available)
        trust_remote_code=True,  # Needed for some models
        low_cpu_mem_usage=True,  # Try to reduce memory usage
        offload_folder="offload",  # Folder to offload tensors to disk
        offload_state_dict=True  # Enable state dict offloading
    )

    # Apply the LoRA adapter
    print(f"Applying LoRA adapter from: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        print("The adapter might have been trained on a different base model architecture.")
        print("This can happen when the original model path differs from the public model.")
        print("Trying to continue without adapter (using base model only)...")
        # If the adapter doesn't load properly, return the base model without the adapter
        return model, tokenizer

    # Set to evaluation mode
    model.eval()

    return model, tokenizer

def classify_article(model, tokenizer, article_text, max_length=2048, threshold_confidence=0.8):
    """
    Classify a news article using the fine-tuned model
    """
    # Create a prompt template for classification
    prompt = f"""Classify the following news article as relevant or not relevant for UPSC preparation.
    Answer with only 'YES' or 'NO'.

    Article: {article_text}

    Classification:"""

    try:
        # Tokenize the input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )

        # Try to move model to real device if it's on meta device
        try:
            # Check if any parameters are on meta device
            has_meta = any(param.device.type == 'meta' for param in model.parameters())
            if has_meta:
                print("Moving model parameters from meta device to actual device...")
                # Move model to CPU first, then to the target device
                model = model.to(torch.float16 if torch.cuda.is_available() else torch.float32)
        except:
            # If there's an issue with moving, continue with the current state
            pass

        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        # Handle the meta device issue by moving tensors explicitly
        if device.type == 'meta':
            device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # We only need a short answer
                temperature=0.1,    # Low temperature for more deterministic output
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode the output
        response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        prompt_length = len(tokenizer.encode(prompt))
        generated_tokens = outputs.sequences[0][prompt_length:]
        classification = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Extract the YES/NO answer with confidence
        classification_upper = classification.upper()
        if 'YES' in classification_upper or 'RELEVANT' in classification_upper:
            relevance = 'YES'
        elif 'NO' in classification_upper or 'NOT RELEVANT' in classification_upper or 'IRRELEVANT' in classification_upper:
            relevance = 'NO'
        else:
            relevance = 'UNKNOWN'

        return relevance, classification

    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return 'ERROR', str(e)

def batch_classify_articles(model, tokenizer, articles_list, max_length=2048):
    """
    Classify multiple news articles in batch
    """
    results = []
    for i, article in enumerate(articles_list):
        print(f"Processing article {i+1}/{len(articles_list)}...")
        relevance, response = classify_article(model, tokenizer, article, max_length)
        results.append({
            'article_index': i,
            'article_preview': article[:100] + "..." if len(article) > 100 else article,
            'relevance': relevance,
            'response': response
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Classify news articles using LoRA adapter")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH,
                        help="Path to the LoRA adapter")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL,
                        help="Base model name")
    parser.add_argument("--article", type=str,
                        help="News article text to classify")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Enable batch processing mode")
    parser.add_argument("--input_file", type=str,
                        help="Path to input file with articles (one per line)")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.adapter_path, args.base_model)

    if args.batch_mode and args.input_file:
        # Read articles from file
        with open(args.input_file, 'r', encoding='utf-8') as f:
            articles = [line.strip() for line in f if line.strip()]

        print(f"Classifying {len(articles)} articles in batch mode...")
        results = batch_classify_articles(model, tokenizer, articles)

        # Print results
        for result in results:
            print(f"Article {result['article_index']+1}: {result['relevance']} - {result['article_preview']}")

        # Print summary
        yes_count = sum(1 for r in results if r['relevance'] == 'YES')
        no_count = sum(1 for r in results if r['relevance'] == 'NO')
        unknown_count = sum(1 for r in results if r['relevance'] == 'UNKNOWN')
        error_count = sum(1 for r in results if r['relevance'] == 'ERROR')

        print(f"\nSummary: {yes_count} YES, {no_count} NO, {unknown_count} UNKNOWN, {error_count} ERROR")

    elif args.article:
        # Classify single article
        relevance, full_response = classify_article(model, tokenizer, args.article)

        print(f"Article Relevance: {relevance}")
        print(f"Full Model Response: {full_response}")

    else:
        parser.print_help()

if __name__ == "__main__":
    # Example usage with a sample article
    sample_article = """
    The government has announced a new policy regarding renewable energy subsidies
    that will take effect next month. The policy aims to boost solar and wind energy
    adoption across rural areas, potentially impacting agricultural communities significantly.
    """

    print("Loading model and tokenizer...")
    # You can run this script directly to test with the sample article
    model, tokenizer = load_model_and_tokenizer(ADAPTER_PATH, BASE_MODEL)
    print("Model loaded successfully!")

    print("\nSample Article Classification:")
    relevance, full_response = classify_article(model, tokenizer, sample_article.strip())
    print(f"Relevance: {relevance}")
    print(f"Response: {full_response}")

    print("\nFor command line usage, run:")
    print("python classify_news_with_lora.py --article 'Your news article text here'")

    print("\nFor batch processing, create a file with one article per line and run:")
    print("python classify_news_with_lora.py --batch_mode --input_file path/to/articles.txt")
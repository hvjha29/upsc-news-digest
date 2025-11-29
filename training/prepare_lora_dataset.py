# training/prepare_lora_dataset.py
"""
Convert gen_pairs_distilled.jsonl to a HuggingFace dataset object on disk (data/train.jsonl)
Produce tokenized dataset with 'input_ids' and 'labels' with prompt masking.
"""
import json, argparse, os
from transformers import AutoTokenizer
from datasets import Dataset

DEFAULT_MODEL = "tiiuae/falcon-7b-instruct"  # choose base tokenizer model (must match your base)
MAX_LEN = 512

def build_prompt(example_input, target_output):
    # simple prompt template — you can change to a more formal instruction template
    prompt = f"Instruction: Given the following context, produce a UPSC mains question similar in style to the example.\n\nContext:\n{example_input}\n\nExample PYQ:\n{target_output}\n\nQuestion:"
    return prompt

def main(infile="data/gen_pairs_distilled.jsonl", out_jsonl="data/train_lora.jsonl", tokenizer_name=DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    records = []
    with open(infile, encoding="utf8") as fin, open(out_jsonl, "w", encoding="utf8") as fout:
        for line in fin:
            j=json.loads(line)
            context=j.get("context","")
            target=j.get("target", j.get("pyq_text",""))
            prompt = build_prompt(context, target)
            # For causal LM training, input = prompt + " " + target
            full = prompt + " " + target
            # truncate if necessary
            # write as jsonl lines for Dataset
            row={"text": full, "prompt_len": len(prompt)}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("Wrote tokenization-ready jsonl:", out_jsonl)
    # optionally, you can load it with datasets and tokenize there in training script

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--infile", default="data/gen_pairs_distilled.jsonl")
    p.add_argument("--out", default="data/train_lora.jsonl")
    p.add_argument("--tokenizer", default=DEFAULT_MODEL)
    args=p.parse_args()
    main(args.infile, args.out, args.tokenizer)

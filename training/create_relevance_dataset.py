# training/create_relevance_dataset.py
"""
Create a relevance dataset from your PYQs (positives) and generated negatives.

Output: data/relevance_dataset.jsonl
Each line: {"id": "...", "text": "...", "label": "YES"/"NO", "source": "..."}

Negative sampling strategy (tries in order):
  1) If Chroma collection "news_chunks" exists: sample chunks that have low cosine similarity to PYQs.
  2) Else: fetch random Wikipedia summaries via REST API.
  3) Else: generate synthetic negatives by shuffling / perturbing PYQ text.

Usage:
  python training/create_relevance_dataset.py \
    --pyq_csv data/pyqs_pwonly.csv \
    --out data/relevance_dataset.jsonl \
    --neg_per_pos 3

Requirements:
  pip install sentence-transformers requests python-dotenv chromadb
  (chromadb optional; if not present, script falls back to wikipedia or synthetic negatives)
"""
import argparse
import csv
import json
import os
import uuid
import random
import time
from typing import List

# try imports
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# chroma helper from your repo if present
try:
    from index.chroma_client import get_client, get_or_create_collection
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

# simple vector math
import numpy as np
import requests

WIKI_RANDOM_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/random/summary"

DEFAULT_PYQ_CSV = "data/pyqs_pwonly.csv"
OUT_DEFAULT = "data/relevance_dataset.jsonl"
EMBED_MODEL = "all-mpnet-base-v2"

def load_pyqs(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("question_text"):
                out.append(r)
    return out

def embed_texts(model, texts: List[str]):
    if model is None:
        raise RuntimeError("No embedding model available")
    return model.encode(texts, show_progress_bar=False)

def sample_negatives_from_chroma(pyq_texts: List[str], per_pos: int, embed_model):
    """
    Query Chroma 'news_chunks' and find chunks with low similarity to pyq embeddings.
    Returns a list of negative texts.
    """
    negs = []
    try:
        client = get_client()
        col = get_or_create_collection(client, "news_chunks")
    except Exception as e:
        print("[chroma] not available or collection missing:", e)
        return negs

    # fetch all chunk docs (careful: for big DB this may be heavy; we'll fetch med sample)
    try:
        all_docs = col.get()
        docs = all_docs.get("documents", []) or []
        # if result is nested list, flatten
        if docs and isinstance(docs[0], list):
            # docs is list per query, flatten
            docs_flat = [d for sub in docs for d in sub]
        else:
            docs_flat = docs
    except Exception:
        docs_flat = []

    if not docs_flat:
        return []

    # compute embeddings for docs using embed_model
    try:
        doc_embs = embed_texts(embed_model, docs_flat)
    except Exception as e:
        print("[embed] failed to embed docs:", e)
        return []

    # compute pyq embeddings
    pyq_embs = embed_texts(embed_model, pyq_texts)

    # For each pyq choose low-sim docs as negatives
    for i, q_emb in enumerate(pyq_embs):
        sims = np.dot(doc_embs, q_emb) / (np.linalg.norm(doc_embs, axis=1) * (np.linalg.norm(q_emb) + 1e-12))
        # choose candidates with smallest similarity
        order = np.argsort(sims)[: max(50, per_pos*5)]  # pick from bottom 50
        chosen_idx = list(order[:per_pos])
        for idx in chosen_idx:
            negs.append({"text": docs_flat[idx], "source": "chroma_news_chunk"})
    return negs

def fetch_wikipedia_random(n: int):
    out = []
    for _ in range(n):
        try:
            r = requests.get(WIKI_RANDOM_SUMMARY, timeout=10)
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract") or data.get("title") or ""
                if extract:
                    out.append({"text": extract, "source": "wikipedia_random"})
            time.sleep(0.3)
        except Exception as e:
            # stop trying if wikipedia is unreachable
            print("[wiki] error:", e)
            break
    return out

def synthetic_negative_from_pyq(pyq_text: str):
    # simple perturbations: shuffle sentences, drop keywords, append random tokens
    parts = [p.strip() for p in pyq_text.split(".") if p.strip()]
    random.shuffle(parts)
    shuffled = ". ".join(parts)
    words = shuffled.split()
    # drop some words
    if len(words) > 8:
        for _ in range(min(3, len(words)//10)):
            del words[random.randrange(len(words))]
    perturbed = " ".join(words)
    # append some unrelated phrase
    perturbed += " " + random.choice([
        "This text is about gardening and home decor.",
        "An unrelated passage about cooking techniques and recipes.",
        "Notes on a fictional travelogue about a small town."
    ])
    return {"text": perturbed, "source": "synthetic"}

def main(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pyqs = load_pyqs(args.pyq_csv)
    print(f"[INFO] Loaded {len(pyqs)} PYQs from {args.pyq_csv}")

    # create embedding model if available
    embed_model = None
    if SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print("[WARN] failed to load embedding model:", e)
            embed_model = None
    else:
        print("[WARN] sentence-transformers not installed; some negative strategies will be skipped.")

    positives = []
    for p in pyqs:
        qtext = p.get("question_text") or p.get("pyq_text") or ""
        if qtext.strip():
            positives.append({"text": qtext.strip(), "source": p.get("source_url", "pyq_file")})

    # build dataset
    out_f = open(args.out, "w", encoding="utf8")
    total_pos = len(positives)
    negs_needed = total_pos * args.neg_per_pos
    negatives = []

    # Strategy 1: sample from CHROMA if available and we have embeddings
    if HAS_CHROMA and embed_model is not None:
        print("[INFO] Trying Chroma-based negative sampling")
        pyq_texts = [p["text"] for p in positives]
        try:
            chroma_negs = sample_negatives_from_chroma(pyq_texts, args.neg_per_pos, embed_model)
            print(f"[INFO] Got {len(chroma_negs)} negatives from Chroma")
            negatives.extend(chroma_negs)
        except Exception as e:
            print("[WARN] Chroma sampling failed:", e)

    # Strategy 2: Wikipedia random extracts
    if len(negatives) < negs_needed:
        to_fetch = negs_needed - len(negatives)
        print(f"[INFO] Fetching {to_fetch} random Wikipedia summaries as negatives")
        wiki_negs = fetch_wikipedia_random(to_fetch)
        print(f"[INFO] Got {len(wiki_negs)} wiki negatives")
        negatives.extend(wiki_negs)

    # Strategy 3: synthetic negatives from PYQs (safe fallback)
    if len(negatives) < negs_needed:
        print("[INFO] Generating synthetic negatives from PYQs")
        while len(negatives) < negs_needed:
            src = random.choice(positives)
            negatives.append(synthetic_negative_from_pyq(src["text"]))

    # Truncate negatives to exact amount
    negatives = negatives[:negs_needed]

    # Shuffle positives and negatives together and write labels
    # We'll write POSITIVE label as "YES" and NEGATIVE as "NO"
    # To avoid ordering bias, interleave examples a bit
    pos_idx = 0
    neg_idx = 0
    written = 0
    # write all positives first or interleaved—here we write interleaved for mixing
    while pos_idx < len(positives) or neg_idx < len(negatives):
        if pos_idx < len(positives):
            rec = positives[pos_idx]
            out_f.write(json.dumps({"id": str(uuid.uuid4()), "text": rec["text"], "label": "YES", "source": rec.get("source", "pyq")}, ensure_ascii=False) + "\n")
            pos_idx += 1
            written += 1
        # write up to neg_per_pos negatives per positive
        for _ in range(args.neg_per_pos):
            if neg_idx < len(negatives):
                rec = negatives[neg_idx]
                out_f.write(json.dumps({"id": str(uuid.uuid4()), "text": rec["text"], "label": "NO", "source": rec.get("source", "")}, ensure_ascii=False) + "\n")
                neg_idx += 1
                written += 1
    out_f.close()
    print(f"[DONE] Wrote {written} examples to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pyq_csv", default=DEFAULT_PYQ_CSV)
    p.add_argument("--out", default=OUT_DEFAULT)
    p.add_argument("--neg_per_pos", type=int, default=2, help="Number of negatives per positive example")
    args = p.parse_args()
    main(args)

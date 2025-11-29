#!/usr/bin/env python3
"""
run_pipeline.py
A minimal, clear smoke-run that scrapes one article, runs lightweight preprocessing,
calls the (stubbed) embedder, produces a tiny summary, and writes an HTML report.
"""

from __future__ import annotations

import warnings

# Suppress harmless LibreSSL warning on macOS (urllib3 v2 + LibreSSL compatibility)
# See: https://github.com/urllib3/urllib3/issues/3020
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

from ingest.scraper_news import fetch_article
from preprocessing.cleaner import simple_clean
from preprocessing.chunker import chunk_text
from embeddings.embedder import embed_text, embed_texts
from utils.output_writer import write_html_report

# indexing + summarization
from index.chroma_client import get_client, get_or_create_collection
from utils.rag.summarizer import is_relevant_article, call_openai_summarizer
import uuid


# ---- logging ----
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---- small typed result container ----
@dataclass
class Result:
    title: str
    cleaned_text: str
    embedding_length: int
    summary_text: str
    meta: Dict


# ---- core pipeline ----
def run_pipeline(url: str) -> Result:
    """
    Run the minimal pipeline for a single article URL.

    Returns:
        Result: contains title, cleaned text, embedding size and generated summary text.
    """
    # 1) fetch
    text, meta = fetch_article(url)
    title = meta.get("title", "Untitled")
    logger.info("Fetched article: %s", title)

    # 2) clean / preprocess
    cleaned = simple_clean(text)
    logger.debug("Cleaned length: %d", len(cleaned))

    # 3) embedding (stubbed)
    # embed_text returns numpy array; get the shape
    vectors = embed_text(cleaned[:2000])
    emb_len = vectors.shape[0] if hasattr(vectors, 'shape') else len(vectors)
    logger.info("Embedding vector length: %d", emb_len)

    # 4) summary (placeholder — swap in real summarizer)
    summary_lines = [
        "1) 3-line gist: (placeholder) This is a short gist of the article.",
        "2) 6 bullet facts: (placeholder) - fact1\n- fact2\n- fact3\n- fact4\n- fact5\n- fact6",
        "3) 2 PYQ-style prompts: (placeholder) Q1; Q2"
    ]
    summary_text = "\n\n".join(summary_lines)

    return Result(
        title=title,
        cleaned_text=cleaned,
        embedding_length=emb_len,
        summary_text=summary_text,
        meta=meta,
    )


def orchestrator(
    url: str,
    collection_name: str = "news",
    max_tokens: int = 500,
    overlap: int = 50,
):
    """Higher-level orchestration: run pipeline, chunk, index, and summarize.

    Steps:
    - Run the basic `run_pipeline` to fetch, clean and embed a short slice.
    - Chunk the cleaned text into overlapping passages.
    - Embed passages and add to a Chroma collection (best-effort).
    - Call the summarizer (OpenAI-backed if API key is present) on the chunks.

    Returns a tuple `(result: Result, summary_text: str, indexed: bool)`.
    """
    result = run_pipeline(url)

    # Heuristic relevance check (optional)
    relevant, tag_info = is_relevant_article(result.cleaned_text)
    logger.info("Relevance: %s; tags: %s", relevant, tag_info)

    # Chunk the cleaned text (token-accurate when tiktoken is installed)
    chunks = chunk_text(result.cleaned_text, max_tokens=max_tokens, overlap=overlap)
    logger.info("Created %d chunks (approx).", len(chunks))

    # Embed all chunks (may be slow for many chunks)
    # embed_texts returns numpy array; convert to list for Chroma
    try:
        embeddings_np = embed_texts(chunks)
        embeddings = embeddings_np.tolist() if hasattr(embeddings_np, 'tolist') else embeddings_np
    except Exception:
        # fallback: embed slices one-by-one
        embeddings = []
        for c in chunks:
            emb = embed_text(c)
            embeddings.append(emb.tolist() if hasattr(emb, 'tolist') else emb)

    # Index into Chroma (best-effort)
    indexed = False
    try:
        client = get_client()
        col = get_or_create_collection(client, collection_name)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"title": result.title, "url": result.meta.get("url"), "chunk_index": i} for i in range(len(chunks))]

        # Chroma collection.add expects embeddings as list[list[float]] and documents/metadatas
        col.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
        indexed = True
        logger.info("Indexed %d chunks into collection '%s'", len(chunks), collection_name)
    except Exception as exc:
        logger.warning("Indexing into Chroma failed: %s", exc)

    # Sanitize meta (datetime -> isoformat) before sending to summarizer
    safe_meta = {}
    for k, v in (result.meta or {}).items():
        try:
            if hasattr(v, "isoformat"):
                safe_meta[k] = v.isoformat()
            else:
                safe_meta[k] = v
        except Exception:
            safe_meta[k] = str(v)

    # Summarize (OpenAI-backed if key present, otherwise local fallback)
    try:
        summary = call_openai_summarizer(chunks, safe_meta)
    except Exception as exc:
        logger.warning("Summarizer failed; falling back to placeholder: %s", exc)
        summary = result.summary_text

    return result, summary, indexed


# ---- CLI / runner ----
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a minimal news-to-summary pipeline.")
    p.add_argument("--sample", action="store_true", help="Use built-in sample URL")
    p.add_argument(
        "--url",
        type=str,
        default="",
        help="Article URL to ingest. If omitted and --sample is not provided, you'll be prompted.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample:
        url = "https://indianexpress.com/article/explained/explained-economics/c-grade-imf-valuation-india-national-account-statistics-10392483"
    elif args.url:
        url = args.url
    else:
        url = input("Enter article URL: ").strip()

    if not url:
        logger.error("No URL provided. Exiting.")
        return

    try:
        result, summary, indexed = orchestrator(url)
    except Exception as exc:  # keep broad for the smoke-run; narrow later
        logger.exception("Pipeline failed: %s", exc)
        return

    # Build a tidy HTML body: title + summary + meta + indexing info
    html_body = (
        f"<h2>{result.title}</h2>\n"
        f"<h3>Summary</h3>\n"
        f"<pre>{summary}</pre>\n"
        f"<h3>Meta</h3>\n"
        f"<pre>{result.meta}</pre>\n"
        f"<h3>Notes</h3>\n"
        f"<p>Embedding length: {result.embedding_length}</p>\n"
        f"<p>Indexed to collection: {indexed}</p>\n"
    )
    report_path = write_html_report(html_body, title=f"UPSC Digest - {result.title}")

    logger.info("Report saved to: %s", report_path)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

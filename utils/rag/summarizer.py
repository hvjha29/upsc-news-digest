# rag/summarizer.py
import os
import openai
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# model used for embeddings (same as embedder)
_EMBED_MODEL = None
def embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _EMBED_MODEL

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

SYLLABUS_SEEDS = {
    # minimal seeds, extend this list with more syllabus examples per subject
    "Polity": ["constitution", "parliament", "preamble", "fundamental rights", "civic"],
    "Economy": ["rbi", "inflation", "gdp", "fiscal", "monetary policy", "budget"],
    "Environment": ["climate", "biodiversity", "forest", "conservation", "pollution"],
    "International Relations": ["united nations", "bilateral", "sanctions", "treaty", "foreign policy"],
    "Science & Tech": ["space", "ai", "nuclear", "innovation", "research"],
}

def build_seed_embeddings():
    model = embed_model()
    seed_embeddings = {}
    for tag, seed_texts in SYLLABUS_SEEDS.items():
        concat = " . ".join(seed_texts)
        seed_embeddings[tag] = model.encode([concat])[0]
    return seed_embeddings

SEED_EMBS = build_seed_embeddings()

def is_relevant_article(text: str, threshold=0.45) -> (bool, dict):
    """
    Heuristic relevance: compute article embedding and cosine similarity to syllabus seeds.
    Returns (is_relevant, tag_scores)
    """
    model = embed_model()
    emb = model.encode([text])[0].reshape(1, -1)
    tags = {}
    for tag, s_emb in SEED_EMBS.items():
        sim = cosine_similarity(emb, s_emb.reshape(1, -1))[0][0]
        tags[tag] = float(sim)
    # decide relevance if any tag score > threshold
    best_tag = max(tags, key=lambda k: tags[k])
    best_score = tags[best_tag]
    return (best_score >= threshold, {"best_tag": best_tag, "best_score": best_score, "scores": tags})

def call_openai_summarizer(chunks: List[str], source_meta: dict) -> str:
    """
    Use OpenAI chat completion to get a UPSC-style summary.
    """
    if not OPENAI_KEY:
        # fallback simple summary when API key not present
        return simple_local_summary(chunks, source_meta)

    # Compose context: include up to N chunks (trim if too long)
    max_chunks = 6
    context = "\n\n---\n\n".join(chunks[:max_chunks])

    system = (
        "You are an expert assistant that summarizes news and government documents specifically "
        "for UPSC Civil Services Examination candidates. Output must be factual, concise, and exam-focused."
    )
    user = (
        f"Context (retrieved passages):\n{context}\n\n"
        "Produce the following in JSON (only valid JSON):\n"
        "1) gist: three-line concise gist suitable for prelim revision.\n"
        "2) facts: list of up to 6 short bullet facts, each with a date or source if present.\n"
        "3) pyq_questions: list of 2 probable PYQ-style questions (short-answer or MCQ), labelled with difficulty: prelim/mains.\n"
        "4) tags: list of likely syllabus tags from the following: Polity, Economy, Environment, International Relations, Science & Tech, Misc.\n\n"
        f"Also include source metadata: {source_meta}.\n\nReturn only a JSON object."
    )

    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # change model if you prefer
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.0,
        max_tokens=800,
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # try to sanitize and return text
    return text

def simple_local_summary(chunks: List[str], source_meta: dict):
    # fallback: create a basic extractive summary + tags
    gist = " ".join([c[:300].rsplit(".", 1)[0] + "." for c in chunks[:2]])
    facts = []
    ct = 0
    for c in chunks[:6]:
        sentence = c.split(".")[0].strip()
        if sentence:
            facts.append(sentence)
            ct += 1
            if ct >= 6:
                break
    tags = [max(SEED_EMBS.keys(), key=lambda k: SEED_EMBS[k].dot(embed_model().encode([chunks[0]])[0]) )]
    out = {
        "gist": gist,
        "facts": facts,
        "pyq_questions": [
            {"q": "What is the key issue mentioned in the article?", "difficulty": "prelim"},
            {"q": "Explain in 100-200 words the implications of the article for policy.", "difficulty": "mains"}
        ],
        "tags": tags,
        "source_meta": source_meta
    }
    import json
    return json.dumps(out, ensure_ascii=False, indent=2)

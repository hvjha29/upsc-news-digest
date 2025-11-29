# training/prepare_generator_pairs.py
"""
Create context -> PYQ training pairs (data/gen_pairs.jsonl).

Workflow:
 - For each PYQ in data/pyqs_pwonly.csv:
     - Retrieve top-K chunks from Chroma (news_chunks)
     - For each chunk create a pair where context=chunk_text and target=pyq_text
 - Optionally dedupe and write JSONL.

Output: data/gen_pairs.jsonl
"""
import csv, json, uuid, argparse, os, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path so we can import from sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))
from index.chroma_client import get_client, get_or_create_collection

DEFAULT_PYQ_CSV = "data/pyqs_pwonly.csv"
OUT = "data/gen_pairs.jsonl"
TOP_K = 6
SBERT_NAME = "all-mpnet-base-v2"

def load_pyqs(path):
    out=[]
    with open(path, encoding="utf8") as f:
        r=csv.DictReader(f)
        for row in r:
            out.append(row)
    return out

def get_chunks_for_pyq(pyq_text, model, top_k=TOP_K):
    client = get_client()
    col = get_or_create_collection(client, "news_chunks")
    q_emb = model.encode([pyq_text])[0].tolist()
    try:
        res = col.query(query_embeddings=[q_emb], n_results=top_k)
        chunks = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
    except Exception:
        chunks = []
        metas = []
    pairs=[]
    for i,ch in enumerate(chunks):
        pairs.append({"context": ch, "meta": metas[i] if i < len(metas) else {}})
    return pairs

def main(pyq_csv=DEFAULT_PYQ_CSV, out=OUT):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model = SentenceTransformer(SBERT_NAME)
    pyqs = load_pyqs(pyq_csv)
    with open(out, "w", encoding="utf8") as fout:
        for pyq in pyqs:
            pyq_id = pyq.get("id")
            pyq_text = pyq.get("question_text")
            if not pyq_text:
                continue
            chunks = get_chunks_for_pyq(pyq_text, model, TOP_K)
            # if no chunks (index empty), still emit one sample with empty context
            if not chunks:
                rec = {"id": str(uuid.uuid4()), "pyq_id": pyq_id, "pyq_text": pyq_text, "context": "", "target": pyq_text}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            for ch in chunks:
                rec = {
                    "id": str(uuid.uuid4()),
                    "pyq_id": pyq_id,
                    "pyq_text": pyq_text,
                    "context": ch["context"],
                    "target": pyq_text,
                    "meta": ch.get("meta", {})
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote pairs to", out)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--pyq_csv", default=DEFAULT_PYQ_CSV)
    p.add_argument("--out", default=OUT)
    args=p.parse_args()
    main(args.pyq_csv, args.out)

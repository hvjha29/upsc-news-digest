# training/distill_targets_openai.py
"""
Read data/gen_pairs.jsonl, call OpenAI to generate a cleaned-up target for each pair,
write data/gen_pairs_distilled.jsonl with a 'target' field (string).
"""
import os, json, argparse, time
import openai
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"  # or gpt-4o/others you have access to
SLEEP=0.5

def distill_one(context, pyq_example):
    prompt = (
        "You are an expert UPSC question editor. Given an example PYQ and a background context, "
        "produce one concise mains-style question that could be asked in UPSC Mains. "
        "Keep it formal, exam-ready and single-line. "
        f"\n\nExample PYQ: {pyq_example}\n\nContext: {context}\n\nOutput:"
    )
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0,
        max_tokens=150
    )
    txt = resp["choices"][0]["message"]["content"].strip()
    return txt

def main(infile="data/gen_pairs.jsonl", outfile="data/gen_pairs_distilled.jsonl"):
    if not openai.api_key:
        raise SystemExit("Set OPENAI_API_KEY in env")
    fout = open(outfile, "w", encoding="utf8")
    with open(infile, encoding="utf8") as fin:
        for line in tqdm(fin):
            rec = json.loads(line)
            try:
                target = distill_one(rec.get("context",""), rec.get("pyq_text",""))
                rec["target"] = target
            except Exception as e:
                rec["target"] = rec.get("target", rec.get("pyq_text",""))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            time.sleep(SLEEP)
    fout.close()
    print("Wrote distilled pairs to", outfile)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--infile", default="data/gen_pairs.jsonl")
    p.add_argument("--outfile", default="data/gen_pairs_distilled.jsonl")
    args=p.parse_args()
    main(args.infile, args.outfile)

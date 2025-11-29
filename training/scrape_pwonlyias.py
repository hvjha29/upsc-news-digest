# training/scrape_pwonlyias.py
"""
Scrape PWOnlyIAS mains solved pages (GS papers by year)
and write a CSV of PYQs to data/pyqs_pwonly.csv

Usage:
  python training/scrape_pwonlyias.py --start 2013 --end 2025 --out data/pyqs_pwonly.csv

Notes:
 - The script is defensive because site HTML varies across years.
 - Inspect and lightly clean the CSV after running; topic hints are high-recall but may need minor fixes.
 - Respect the site terms of use. Use this data for your project and cite the source_url.
"""
import requests
from bs4 import BeautifulSoup
import re
import csv
import argparse
import uuid
import time

BASE = "https://pwonlyias.com/mains-solved-papers-by-year/gs-paper-{year}/"
OUT_DEFAULT = "data/pyqs_pwonly.csv"
USER_AGENT = "Mozilla/5.0 (compatible; UpscNewsDigestBot/1.0; +https://github.com/YOURNAME/upsc-news-digest)"

# heuristics: list of common UPSC topic tags to try to detect in stray text
COMMON_TOPICS = [
    "Art & culture", "Art & cultureIndian Society", "Geography", "Indian Society", "Polity",
    "Economy", "Science & technology", "Environment", "Internal security", "International Relation",
    "Social Justice", "Governance", "Disaster Management", "Ethics", "World History",
    "Ethics (Section A)", "Ethics (Section B)", "Ethics (Section C)"
]

QUESTION_SPLIT_RE = re.compile(r'(?:Que\.|Ques\.|Q\.)\s*\d+', flags=re.IGNORECASE)
# parentheses pattern for "(150 Words, 10 Marks)" or "(250 Words, 15 Marks)"
PAREN_RE = re.compile(r'\((?P<words>\d{2,4})\s*Words\s*,\s*(?P<marks>\d{1,3})\s*Marks\)', flags=re.IGNORECASE)
# simple question number extract
QNO_RE = re.compile(r'^(?:Que\.|Ques\.|Q\.)\s*(?P<num>\d+)\b', flags=re.IGNORECASE)

HEADERS = {"User-Agent": USER_AGENT}

def fetch_page(year, pause=0.5):
    url = BASE.format(year=year)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    time.sleep(pause)
    return resp.text, resp.url

def extract_article_soup(soup):
    # common container patterns; be permissive
    selectors = [
        {"name": "div", "attrs": {"class": "entry-content"}},
        {"name": "div", "attrs": {"class": "post-content"}},
        {"name": "article", "attrs": {}},
        {"name": "div", "attrs": {"id": "content"}},
        {"name": "div", "attrs": {"class": "content"}}
    ]
    for sel in selectors:
        el = soup.find(sel["name"], sel.get("attrs", {}))
        if el:
            return el
    return soup.body

def text_blocks_from_article(article):
    """
    Return a list of textual lines preserving order. We'll later split by 'Que.' tokens.
    """
    # get text with line breaks where block tags are present
    txt = article.get_text(separator="\n")
    # normalize newlines and collapse multiple blank lines
    lines = [ln.strip() for ln in txt.splitlines()]
    # drop empty strings but keep relative order
    lines = [ln for ln in lines if ln]
    return lines

def join_until_next_question(lines, start_idx):
    """
    Build a block starting at start_idx line which contains 'Que.N' and gather until next 'Que.' line or end.
    Return block_text, next_index.
    """
    block_lines = [lines[start_idx]]
    idx = start_idx + 1
    while idx < len(lines):
        if re.search(r'^\s*(?:Que\.|Ques\.|Q\.)\s*\d+', lines[idx], flags=re.IGNORECASE):
            break
        block_lines.append(lines[idx])
        idx += 1
    return " ".join(block_lines), idx

def parse_block(block_text):
    """
    Parse a question block and return dict fields:
    question_no, question_text, word_limit, marks, topic_hint
    """
    # Extract question number from start
    qno_m = QNO_RE.match(block_text)
    qno = qno_m.group("num") if qno_m else ""

    # Remove initial 'Que.X' prefix
    qtext = re.sub(r'^(?:Que\.|Ques\.|Q\.)\s*\d+\s*[:\.\-]?\s*', '', block_text, flags=re.IGNORECASE)

    # If there is a "Show Answer" marker in the block, remove it and split around it
    # Some pages might have "Show Answer" or "Show Answer" as a separate line; handle both
    parts = re.split(r'\bShow Answer\b', qtext, flags=re.IGNORECASE)
    before = parts[0].strip()
    after = parts[1].strip() if len(parts) > 1 else ""

    # If after contains topic tag(s), capture last short token(s)
    topic_hint = ""
    if after:
        # pick the last short chunk from after (often a single token like "Geography" or two tokens)
        # split by whitespace and punctuation and take last up to 4 words
        tokens = re.split(r'[\|\-\–\—\n,;]+', after)
        candidate = tokens[-1].strip() if tokens else after.strip()
        # sanitize candidate
        if 1 <= len(candidate) <= 80:
            topic_hint = candidate

    # attempt to find topic inside the 'before' part at end if not found
    if not topic_hint:
        # sometimes topic sits on same line after question; check trailing tokens in 'before'
        tail = before.split()[-6:]
        tail_s = " ".join(tail)
        for t in COMMON_TOPICS:
            if t.lower() in tail_s.lower():
                topic_hint = t
                # remove the matched suffix from question text
                before = re.sub(re.escape(t), '', before, flags=re.IGNORECASE).strip()
                break

    # Extract parentheses pattern for words/marks from 'before'
    words = ""
    marks = ""
    p = PAREN_RE.search(before)
    if p:
        words = p.group("words")
        marks = p.group("marks")
        # remove the parentheses phrase from question text
        before = before[:p.start()].strip() + " " + before[p.end():].strip()
        before = before.strip()

    # fallback: if parentheses not found but a trailing "(150 Words, 10 Marks)" style without spaces
    if not words:
        p2 = re.search(r'\((?P<inner>[^\)]+)\)$', before)
        if p2:
            inner = p2.group("inner")
            m2 = re.search(r'(\d{2,4})\s*Words.*?(\d{1,3})\s*Marks', inner, flags=re.IGNORECASE)
            if m2:
                words = m2.group(1); marks = m2.group(2)
                before = before[:p2.start()].strip()

    # final clean-up of question text — collapse whitespace
    question_text = re.sub(r'\s+', ' ', before).strip()

    # If question_text is empty, fallback to full block_text cleaned
    if not question_text:
        question_text = re.sub(r'\s+', ' ', block_text).strip()

    return {
        "question_no": qno,
        "question_text": question_text,
        "word_limit": words,
        "marks": marks,
        "topic_hint": topic_hint
    }

def parse_article_text(lines):
    """
    Given flattened lines from article, find sequences containing 'Que.' and parse them.
    Track which GS Paper each question belongs to based on explicit "GS Paper N" markers.
    """
    rows = []
    idx = 0
    current_paper = ""
    
    # Pattern to match "GS Paper 1", "GS Paper 2", etc. (used on pwonlyias.com)
    # Also matches "GS Paper I", "GS Paper II", etc. for other sites
    # Use word boundary \b or lookahead to avoid matching "GS Paper 2013"
    paper_re = re.compile(r'GS\s+Paper\s+([1-4]|I{1,3}|IV)(?:\s|$|\.)', flags=re.IGNORECASE)
    
    while idx < len(lines):
        ln = lines[idx].strip()
        
        # Check if this line contains a paper marker like "GS Paper 1"
        paper_match = paper_re.search(ln)
        if paper_match:
            paper_num_str = paper_match.group(1).upper()
            # Normalize to "GS Paper I/II/III/IV"
            if paper_num_str == "1":
                current_paper = "GS Paper I"
            elif paper_num_str == "2":
                current_paper = "GS Paper II"
            elif paper_num_str == "3":
                current_paper = "GS Paper III"
            elif paper_num_str == "4":
                current_paper = "GS Paper IV"
            elif paper_num_str == "I":
                current_paper = "GS Paper I"
            elif paper_num_str == "II":
                current_paper = "GS Paper II"
            elif paper_num_str == "III":
                current_paper = "GS Paper III"
            elif paper_num_str == "IV":
                current_paper = "GS Paper IV"
            else:
                current_paper = f"GS Paper {paper_num_str}"
            
            idx += 1
            continue

        # detect question start: lines beginning with Que. or Ques.
        if re.match(r'^(?:Que\.|Ques\.|Q\.)\s*\d+', ln, flags=re.IGNORECASE):
            block, next_idx = join_block_from_lines(lines, idx)
            parsed = parse_block(block)
            parsed["paper"] = current_paper
            rows.append(parsed)
            
            # Move to next_idx, but check for paper markers in between
            idx = next_idx
            # Backtrack to check for paper markers that might be between current question and next question
            # This handles the case where paper markers appear after "Show Answer" lines
            if idx > 0:
                for check_idx in range(idx - 3, idx):
                    if check_idx >= 0 and check_idx < len(lines):
                        check_ln = lines[check_idx].strip()
                        check_match = paper_re.search(check_ln)
                        if check_match:
                            # Found a paper marker - process it
                            paper_num_str = check_match.group(1).upper()
                            if paper_num_str == "1":
                                current_paper = "GS Paper I"
                            elif paper_num_str == "2":
                                current_paper = "GS Paper II"
                            elif paper_num_str == "3":
                                current_paper = "GS Paper III"
                            elif paper_num_str == "4":
                                current_paper = "GS Paper IV"
            continue

        # Sometimes questions are numbered inline like "Que.1" embedded; look for 'Que.' anywhere
        if 'Que.' in ln or 'Ques.' in ln or re.search(r'\bQ\.\s*\d+', ln):
            # split into subblocks by Que. tokens
            parts = re.split(r'(\b(?:Que\.|Ques\.|Q\.)\s*\d+)', ln, flags=re.IGNORECASE)
            # recombine: parts like [pre, marker, body, marker, body,...]
            combined = []
            i = 0
            while i < len(parts):
                piece = parts[i]
                if re.match(r'^(?:Que\.|Ques\.|Q\.)', piece or "", flags=re.IGNORECASE):
                    # start new block
                    # include marker with next piece if present
                    if i+1 < len(parts):
                        combined.append(piece + (parts[i+1] if parts[i+1] else ""))
                    i += 2
                else:
                    i += 1
            for block in combined:
                parsed = parse_block(block)
                parsed["paper"] = current_paper
                rows.append(parsed)
            
            idx += 1
            continue

        idx += 1
    
    return rows

def join_block_from_lines(lines, start_idx):
    """
    Similar to join_until_next_question but handles 'Show Answer' sometimes present
    """
    block_lines = [lines[start_idx]]
    idx = start_idx + 1
    while idx < len(lines):
        # stop if next line begins with Que.
        if re.match(r'^(?:Que\.|Ques\.|Q\.)\s*\d+', lines[idx], flags=re.IGNORECASE):
            break
        block_lines.append(lines[idx])
        idx += 1
    return " ".join(block_lines), idx

def run_scrape(start_year=2013, end_year=2025, out=OUT_DEFAULT, pause=0.6):
    rows_out = []
    for year in range(start_year, end_year + 1):
        try:
            html, final_url = fetch_page(year, pause=pause)
        except Exception as e:
            print(f"[WARN] Failed fetching year {year}: {e}")
            continue
        soup = BeautifulSoup(html, "html.parser")
        article = extract_article_soup(soup)
        lines = text_blocks_from_article(article)
        # parse lines into question rows
        parsed_rows = parse_article_text(lines)
        # annotate with year and source url
        for pr in parsed_rows:
            pr_out = {
                "id": str(uuid.uuid4()).replace("-", "")[:12],
                "year": year,
                "paper": pr.get("paper",""),
                "question_no": pr.get("question_no",""),
                "question_text": pr.get("question_text",""),
                "word_limit": pr.get("word_limit",""),
                "marks": pr.get("marks",""),
                "topic_hint": pr.get("topic_hint",""),
                "source_url": final_url
            }
            rows_out.append(pr_out)
        print(f"[INFO] Year {year}: extracted {len(parsed_rows)} questions from {final_url}")

    # write CSV
    fieldnames = ["id","year","paper","question_no","question_text","word_limit","marks","topic_hint","source_url"]
    # ensure output dir exists
    out_dir = out.rsplit("/",1)[0] if "/" in out else "."
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"[DONE] Wrote {len(rows_out)} rows to {out}")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2013)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--out", default=OUT_DEFAULT)
    args = p.parse_args()
    run_scrape(args.start, args.end, args.out)

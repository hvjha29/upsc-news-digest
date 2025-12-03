# ingest/scraper_news.py
from newspaper import Article
from typing import Tuple

def fetch_article(url: str) -> Tuple[str, dict]:
    a = Article(url)
    a.download()
    a.parse()
    text = a.text or ""
    meta = {
        "title": a.title,
        "authors": a.authors,
        "publish_date": a.publish_date,
        "url": url
    }
    return text, meta

if __name__ == "__main__":
    # quick test
    url = "https://indianexpress.com/article/explained/"
    t, m = fetch_article(url)
    print(m)
    print(t[:400])

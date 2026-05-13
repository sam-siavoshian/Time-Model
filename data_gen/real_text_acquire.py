"""Acquire real English text for Phase 3 mixed LM training.

Downloads ~10 public-domain books from Project Gutenberg, cleans the
header/footer, chunks the text into ~256-token segments, saves as JSONL.

Total corpus: ~5-15 MB raw text → ~1.5M tokens. Enough for Phase 3 mixing.

Project Gutenberg URLs use the /ebooks/<id>/<id>-0.txt format. All works
listed below are public domain (US copyright expired).

Usage:
  uv run python -m data_gen.real_text_acquire --out data/real_text/gutenberg.jsonl
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path
from time import sleep

import tiktoken
from tqdm import tqdm

from data_gen.schemas import RealTextChunk


# (gutenberg_id, title, author) — all public domain in the US
BOOKS = [
    (11,    "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (84,    "Frankenstein",                     "Mary Shelley"),
    (1342,  "Pride and Prejudice",              "Jane Austen"),
    (1661,  "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (2701,  "Moby Dick",                        "Herman Melville"),
    (98,    "A Tale of Two Cities",             "Charles Dickens"),
    (1080,  "A Modest Proposal",                "Jonathan Swift"),
    (174,   "The Picture of Dorian Gray",       "Oscar Wilde"),
    (215,   "The Call of the Wild",             "Jack London"),
    (158,   "Emma",                             "Jane Austen"),
]


def fetch_book(gid: int) -> str:
    """Try a few URL patterns for Project Gutenberg plain-text encodings."""
    candidates = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
    ]
    last_err = None
    for url in candidates:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IPCN-research-bot/0.1"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:                                # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"Could not fetch Gutenberg #{gid}: {last_err}")


HEADER_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)
FOOTER_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*", re.IGNORECASE | re.DOTALL)


def strip_gutenberg(text: str) -> str:
    """Remove Gutenberg license boilerplate header + footer."""
    m_start = HEADER_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = FOOTER_RE.search(text)
    if m_end:
        text = text[:m_end.start()]
    return text.strip()


def chunk_text(text: str, tokenizer, target_tokens: int = 256) -> list[str]:
    """Greedy chunking: pack consecutive paragraphs until ~target_tokens hit."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for para in paragraphs:
        ptoks = len(tokenizer.encode(para))
        if current_tokens + ptoks > target_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += ptoks
        if current_tokens >= target_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/real_text/gutenberg.jsonl")
    p.add_argument("--target-tokens", type=int, default=256)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    total_chunks = 0
    total_tokens = 0
    with open(out, "w") as f:
        for gid, title, author in tqdm(BOOKS, desc="books"):
            try:
                raw = fetch_book(gid)
            except Exception as e:                            # noqa: BLE001
                print(f"  skipped Gutenberg #{gid} ({title}): {e}")
                continue
            cleaned = strip_gutenberg(raw)
            chunks = chunk_text(cleaned, tokenizer, args.target_tokens)
            for chunk in chunks:
                tc = len(tokenizer.encode(chunk))
                obj = RealTextChunk(source="other", text=chunk, token_count=tc)
                f.write(obj.model_dump_json() + "\n")
                total_chunks += 1
                total_tokens += tc
            sleep(1)                                          # be polite to gutenberg.org

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {total_chunks} chunks ({total_tokens:,} tokens, {size_mb:.1f} MB) to {out}")


if __name__ == "__main__":
    main()

# data/prepare_chunks.py
"""Sentence-aware clinical text chunking.

Fixes (audit items 37/39): chunks never split a sentence, so negation cues and
qualifiers cannot be severed across chunk boundaries; `chunk_size > overlap`
is enforced to prevent an infinite loop; output is streamed to disk.
"""
import glob
import json
import os
import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+")

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """Pack whole sentences into word-bounded chunks; never split a sentence."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})")
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    n_words = 0
    for sent in sentences:
        w = len(sent.split())
        if current and n_words + w > chunk_size:
            chunks.append(" ".join(current))
            # keep trailing overlap sentences for continuity (whole sentences only)
            tail: list[str] = []
            tail_words = 0
            for s in reversed(current):
                sw = len(s.split())
                if tail_words + sw > overlap:
                    break
                tail.insert(0, s)
                tail_words += sw
            current, n_words = tail, tail_words
        current.append(sent)
        n_words += w
    if current:
        chunks.append(" ".join(current))
    return chunks

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    n_records = 0
    out_path = os.path.join("data", "processed", "chunks.jsonl")
    with open(out_path, "w", encoding="utf-8") as out:
        for path in glob.glob("data/raw/pmc/*.txt"):
            with open(path, encoding="utf-8") as f:
                text = clean_text(f.read())
            for i, chunk in enumerate(chunk_text(text)):
                out.write(json.dumps({
                    "doc_id": path, "chunk_id": i, "text": chunk,
                    "n_words": len(chunk.split()),
                }) + "\n")
                n_records += 1
    print(f"Wrote {n_records} sentence-safe chunks to {out_path}")

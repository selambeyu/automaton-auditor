"""PDF ingestion and RAG-lite query for DocAnalyst."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DocChunk:
    """A chunk of document text with optional page/section info."""

    text: str
    page: Optional[int] = None
    start: int = 0
    end: int = 0


def ingest_pdf(path: str | Path) -> List[DocChunk]:
    """
    Extract text from the PDF and chunk it (by page for simplicity).
    Uses PyMuPDF. Returns list of DocChunk for RAG-lite querying.
    """
    path = Path(path)
    if not path.exists() or not path.suffix.lower() == ".pdf":
        return []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    chunks: List[DocChunk] = []
    try:
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append(
                    DocChunk(text=text.strip(), page=i + 1, start=0, end=len(text))
                )
        doc.close()
    except Exception:
        return []
    return chunks


def query_document(
    chunks: List[DocChunk],
    query: str,
    max_chars: int = 4000,
    keywords: Optional[List[str]] = None,
) -> str:
    """
    RAG-lite: return relevant snippets from chunks for the query.
    If keywords are provided, prefer chunks that contain them; otherwise
    do simple substring/word overlap. Returns concatenated snippet string.
    """
    if not chunks:
        return ""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    if keywords:
        keyword_set = set(k.lower() for k in keywords)
    else:
        keyword_set = query_words

    scored: List[tuple[float, DocChunk]] = []
    for c in chunks:
        t = c.text.lower()
        score = 0.0
        for kw in keyword_set:
            if kw in t:
                score += 1.0
        for w in query_words:
            if len(w) > 2 and w in t:
                score += 0.5
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    out: List[str] = []
    total = 0
    for _, c in scored:
        if total >= max_chars:
            break
        if c.text:
            take = c.text[: max_chars - total]
            out.append(take)
            total += len(take)
    return "\n\n---\n\n".join(out) if out else ""

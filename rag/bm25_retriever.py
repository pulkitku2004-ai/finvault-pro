"""
BM25 retriever with persistent corpus.

build_bm25_index() is called at ingestion time and saves the corpus to disk.
bm25_retrieve() loads from disk on every call (fast — JSON + BM25Okapi rebuild
is ~10ms for a 60-chunk corpus).

If the corpus file is missing on first use, _bootstrap_from_chroma() pulls all
documents from the existing ChromaDB collection so no full rebuild is required.
"""

import json
import logging
import os
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "vector_db", "bm25_corpus.json"
)


# ---------------------------------------------------------------------------
# Ingestion-time: build and persist the corpus
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: List[Document]) -> None:
    """Persist chunk texts so bm25_retrieve() can search them. Call at ingestion."""
    os.makedirs(os.path.dirname(_CORPUS_PATH), exist_ok=True)
    corpus = [{"text": c.page_content, "metadata": c.metadata} for c in chunks]
    with open(_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f)
    logger.info("BM25 corpus saved: %d chunks → %s", len(corpus), _CORPUS_PATH)


# ---------------------------------------------------------------------------
# Query-time: load corpus (with auto-bootstrap fallback)
# ---------------------------------------------------------------------------

def _bootstrap_from_chroma() -> bool:
    """Pull all texts from ChromaDB and build the corpus file (one-time migration)."""
    try:
        from rag.vector_store import load_vector_db
        db = load_vector_db()
        if db is None:
            return False
        result = db._collection.get(include=["documents", "metadatas"])
        texts = result.get("documents") or []
        metas = result.get("metadatas") or [{} for _ in texts]
        if not texts:
            return False
        corpus = [{"text": t, "metadata": m} for t, m in zip(texts, metas)]
        os.makedirs(os.path.dirname(_CORPUS_PATH), exist_ok=True)
        with open(_CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(corpus, f)
        logger.info("BM25 corpus bootstrapped from ChromaDB: %d chunks", len(corpus))
        return True
    except Exception as e:
        logger.warning("Could not bootstrap BM25 from ChromaDB: %s", e)
        return False


def _load_corpus() -> List[dict]:
    if not os.path.exists(_CORPUS_PATH):
        logger.info("BM25 corpus missing — bootstrapping from ChromaDB...")
        _bootstrap_from_chroma()
    if not os.path.exists(_CORPUS_PATH):
        logger.warning("BM25 corpus unavailable; BM25 branch will be skipped")
        return []
    with open(_CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def bm25_retrieve(query: str, top_k: int = 10) -> List[Document]:
    """Return top_k Documents ranked by BM25 score against the persisted corpus."""
    corpus = _load_corpus()
    if not corpus:
        return []

    texts = [entry["text"] for entry in corpus]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return [
        Document(page_content=e["text"], metadata=e.get("metadata") or {})
        for e, _ in ranked[:top_k]
    ]


# ---------------------------------------------------------------------------
# Legacy helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def bm25_search(query: str, documents: List[str], top_k: int = 5) -> List[str]:
    tokenized = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

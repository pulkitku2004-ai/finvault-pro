
import logging
from typing import List, Optional
from langchain_core.documents import Document
from rag.vector_store import retrieve_docs as vector_retrieve

logger = logging.getLogger(__name__)

# Default retrieval parameters
DEFAULT_TOP_K = 10  # Increased from 5 for better context


def retrieve_docs(query: str, top_k: Optional[int] = None) -> List[Document]:
    """
    Retrieve documents from vector store.
    
    Args:
        query: Search query
        top_k: Number of results to return (default: 10)
        
    Returns:
        List of Document objects with metadata
    """
    if top_k is None:
        top_k = DEFAULT_TOP_K
    
    logger.info(f"Retrieving top {top_k} documents for: '{query}'")
    docs = vector_retrieve(query, top_k=top_k)
    
    if not docs:
        logger.warning(f"⚠️  No documents found for query: '{query}'")
    else:
        logger.info(f"✅ Retrieved {len(docs)} documents")
    
    return docs


def retrieve(query: str, top_k: Optional[int] = None) -> List[str]:
    """
    Retrieve document texts (string format).
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of document page_content strings
    """
    docs = retrieve_docs(query, top_k=top_k)
    return [doc.page_content for doc in docs]


def retrieve_with_scores(query: str, top_k: Optional[int] = None) -> List[tuple]:
    """
    Retrieve documents with similarity scores.
    Useful for understanding retrieval confidence.
    
    Args:
        query: Search query
        top_k: Number of results to return
        
    Returns:
        List of tuples (document, similarity_score)
    """
    from rag.vector_store import load_vector_db
    
    if top_k is None:
        top_k = DEFAULT_TOP_K
    
    db = load_vector_db()
    if db is None:
        logger.warning("Vector DB not loaded")
        return []
    
    try:
        # Similarity search with scores
        results = db.similarity_search_with_score(query, k=top_k)
        
        logger.info(f"Retrieved {len(results)} documents with scores")
        
        return results
    except Exception as e:
        logger.error(f"Retrieval with scores failed: {e}")
        return []


def retrieve_hybrid_with_scores(
    query: str, top_k: Optional[int] = None
) -> List[tuple[Document, float]]:
    """
    Hybrid BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF).

    Pulls top_k*2 candidates from each retriever, fuses with RRF (k=60),
    returns top_k (Document, rrf_score) pairs sorted best-first.
    RRF scores are already in (0, 1] — no further normalisation needed.
    Falls back to dense-only if BM25 corpus is unavailable.
    """
    from rag.bm25_retriever import bm25_retrieve

    if top_k is None:
        top_k = DEFAULT_TOP_K

    pool = top_k * 2
    K = 60  # standard RRF constant

    dense_results: List[tuple[Document, float]] = retrieve_with_scores(query, top_k=pool)
    bm25_results: List[Document] = bm25_retrieve(query, top_k=pool)

    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, Document] = {}

    def _fp(text: str) -> str:
        return text[:150]

    for rank, (doc, _) in enumerate(dense_results, 1):
        fp = _fp(doc.page_content)
        rrf_scores[fp] = rrf_scores.get(fp, 0.0) + 1.0 / (K + rank)
        doc_store[fp] = doc

    for rank, doc in enumerate(bm25_results, 1):
        fp = _fp(doc.page_content)
        rrf_scores[fp] = rrf_scores.get(fp, 0.0) + 1.0 / (K + rank)
        if fp not in doc_store:
            doc_store[fp] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    logger.info(
        "Hybrid retrieval: %d dense + %d BM25 → %d after RRF",
        len(dense_results), len(bm25_results), len(ranked),
    )
    return [(doc_store[fp], score) for fp, score in ranked]


def retrieve_hybrid(query: str, top_k: Optional[int] = None) -> List[Document]:
    """Convenience wrapper — returns only Documents from hybrid retrieval."""
    return [doc for doc, _ in retrieve_hybrid_with_scores(query, top_k=top_k)]


def filter_by_content_type(docs: List[Document], content_type: str = "text") -> List[Document]:
    """
    Filter retrieved documents by content type.
    
    Args:
        docs: List of documents
        content_type: "text", "table", "metric", etc.
        
    Returns:
        Filtered list of documents
    """
    filtered = []
    for doc in docs:
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        doc_type = metadata.get('content_type', 'text')
        
        if doc_type.lower() == content_type.lower():
            filtered.append(doc)
    
    return filtered


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    Remove duplicate documents from results.
    
    Args:
        docs: List of documents
        
    Returns:
        List of unique documents
    """
    seen = set()
    unique = []
    
    for doc in docs:
        content_hash = hash(doc.page_content[:100])  # Hash first 100 chars
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    
    return unique


def rerank_docs_by_relevance(docs: List[Document], query: str) -> List[Document]:
    """
    Re-rank documents by relevance to query (keyword-based).
    Simple ranking: count keyword matches in each document.
    
    Args:
        docs: List of documents
        query: Original search query
        
    Returns:
        Re-ranked list of documents
    """
    def relevance_score(doc: str, q: str) -> int:
        """Count how many words from query appear in document."""
        query_words = set(q.lower().split())
        doc_words = doc.lower().split()
        return sum(1 for word in query_words if word in doc_words)
    
    # Score each document
    scored_docs = [
        (doc, relevance_score(doc.page_content, query))
        for doc in docs
    ]
    
    # Sort by score (descending)
    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked]


class RetrieverConfig:
    """Configuration for retriever behavior."""
    
    def __init__(
        self,
        top_k: int = 10,
        enable_reranking: bool = True,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.0  # 0 = no filtering
    ):
        """
        Initialize retriever configuration.
        
        Args:
            top_k: Number of documents to retrieve
            enable_reranking: Whether to re-rank results
            enable_deduplication: Whether to remove duplicates
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.top_k = top_k
        self.enable_reranking = enable_reranking
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold
    
    def __repr__(self):
        return (
            f"RetrieverConfig(top_k={self.top_k}, "
            f"reranking={self.enable_reranking}, "
            f"dedup={self.enable_deduplication}, "
            f"threshold={self.similarity_threshold})"
        )


# Global config
_config = RetrieverConfig()


def set_retriever_config(config: RetrieverConfig):
    """Set global retriever configuration."""
    global _config
    _config = config
    logger.info(f"Retriever config updated: {config}")


def get_retriever_config() -> RetrieverConfig:
    """Get current retriever configuration."""
    return _config
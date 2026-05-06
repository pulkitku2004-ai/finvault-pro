"""
FinVault → ASTR-O span adapter.

Maps FinVault's retrieval + generation output to the ASTR-O span dict that
ASTROPipeline.process_span() expects.  No ASTR-O code lives here — this is
pure FinVault-side glue.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any


def _chunk_id(text: str) -> str:
    """Stable chunk ID from first 120 chars of content."""
    return "fv_" + hashlib.md5(text[:120].encode("utf-8")).hexdigest()[:12]


def build_finvault_span(
    query: str,
    scored_pairs: List[Tuple[Any, float]],   # (doc_or_str, similarity_score)
    answer: str,
    logprobs: List[Dict],                     # [{"token": str, "logprob": float}]
    retrieval_method: str = "dense",
    trace_id: str | None = None,
) -> dict:
    """
    Build an ASTR-O span from FinVault retrieval + generation output.

    Args:
        query          — original user query
        scored_pairs   — list of (doc, similarity_score) from route_query_with_scores()
                         score is already normalised to [0, 1] (higher = more similar)
        answer         — final LLM answer text
        logprobs       — token logprobs from generate_answer_for_audit()
        retrieval_method — "dense" | "hybrid" | "graph+dense"
        trace_id       — optional; auto-generated if omitted
    """
    span_id  = str(uuid.uuid4())
    trace_id = trace_id or str(uuid.uuid4())
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    retrieved_chunks   = []
    all_ranked_chunks  = []
    seen_ids           = set()

    for rank, (doc, score) in enumerate(scored_pairs, 1):
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        cid  = _chunk_id(text)

        # Deduplicate by chunk_id
        if cid in seen_ids:
            continue
        seen_ids.add(cid)

        retrieved_chunks.append({
            "chunk_id": cid,
            "source":   "hdfc_q3.pdf",   # must match filename in reference_registry.json
            "text":     text,
            "metadata": {"source_tier": "CRITICAL"},
        })

        all_ranked_chunks.append({
            "chunk_id":    cid,
            "rank":        rank,
            "dense_score": round(float(score), 4),
            "sparse_score": 0.0,          # BM25 not wired — honest zero
            "rrf_score":   round(float(score), 4),
        })

    return {
        "span_id":   span_id,
        "trace_id":  trace_id,
        "timestamp": now,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_metadata": {
            "query":            query,
            "retrieval_method": retrieval_method,
            "top_k":            len(retrieved_chunks),
            "retrieved_chunks": [c["chunk_id"] for c in retrieved_chunks],
            "all_ranked_chunks": all_ranked_chunks,
        },
        "llm_response": {
            "text":     answer,
            "logprobs": logprobs,
        },
    }

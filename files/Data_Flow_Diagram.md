User Query
   │
   ▼
Streamlit UI  (app.py)
   │  HTTP POST /query
   ▼
FastAPI Endpoint  (api.py)
   │
   ├─ 1. route_query_with_scores(query)
   │       │
   │       ▼
   │   Query Rewriter  (rag/query_rewriter.py)
   │   expand abbreviations, detect risk/financial intent
   │       │
   │       ▼
   │   LLM Intent Router  (agent_router.py → gpt-4o)
   │   classifies: graph_search | vector_search | calculator
   │       │
   │       ├── graph_search ──► Neo4j Knowledge Graph
   │       │                    executives, risks, metrics, regulations
   │       │                    + hybrid vector supplement (below)
   │       │
   │       └── vector_search
   │               │
   │               ├── Dense Retrieval  (ChromaDB, text-embedding-3-small)
   │               │   returns (doc, L2_distance) pairs
   │               │
   │               └── BM25 Retrieval  (rag/bm25_retriever.py)
   │                   keyword match on persisted corpus
   │                   (auto-bootstrapped from ChromaDB on first use)
   │                       │
   │                       ▼
   │               RRF Fusion  (rag/retriever.py)
   │               score = Σ 1/(60 + rank) across both retrievers
   │               returns (doc, rrf_score) pairs, best-first
   │
   ├─ 2. rerank_results(query, cleaned_context, top_k=6)
   │       Cross-encoder re-scores every (query, chunk) pair
   │       ms-marco-MiniLM-L-6-v2 on MPS (Apple Silicon)
   │
   ├─ 3. generate_answer_for_audit(query, context)  (rag/generator.py)
   │       Stage 1 — Fact Extractor prompt → gpt-4o → bullet facts
   │       Stage 2 — Grounded Synthesis prompt → gpt-4o (logprobs=True)
   │       returns (answer_text, token_logprobs)
   │
   ├─ 4. build_finvault_span(query, scored_pairs, answer, logprobs)
   │       (astr_o_adapter.py)
   │       chunk_id = "fv_" + md5(text[:120])[:12]
   │       source   = "hdfc_q3.pdf"
   │       source_tier = "CRITICAL"
   │
   ├─ 5. ASTROPipeline.process_span(span)  (astr_o_init.py → ~/astr-o)
   │       Layer 1  Metadata bridge — verify chunk source vs registry
   │       Layer 4  Contradiction detector — domain schema regex
   │       Layer 5  Token confidence — logprob ≥ −0.25 on critical tokens
   │       Layer 6  Source verification — AUTHORITATIVE / COMPROMISED
   │       Layer 6b Groundedness — answer words ∈ chunk text
   │       Layer 7  Criteria gate — SAFE if all 4 pass, else FLAGGED
   │       Layer 8  Causal chain mapper (FLAGGED only)
   │       Layer 9  Report generator + HMAC-SHA256 signer
   │       Layer 11 Storage (hot → cold lifecycle)
   │
   └─ 6. format_response(answer, astr_o_result)
           SAFE              → status="verified",   badge="✓ Verified"
           FLAGGED           → status="unverified",  badge="⚠ Unverified"
           PARTIAL_EVALUATION → status="partial",    badge="⚠ Partial"
           ERROR             → status="error",       answer=None
   │
   ▼
API Response JSON
{status, answer, verification{badge, confidence, summary},
 sources[{document, tier_badge, authorization, snippet}],
 metrics{hallucination_score, safety_score}, audit{span_id, signature},
 query, documents_used}
   │
   ▼
Streamlit Chat UI  (app.py)
   Display answer + st.success / st.warning / st.error badge
   Sources expander with 🔴/🟡/⚪ tier badges and auth status

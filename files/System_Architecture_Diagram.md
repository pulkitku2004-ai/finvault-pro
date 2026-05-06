                    ┌───────────────────────┐
                    │       User UI         │
                    │  Streamlit  (app.py)  │
                    │  ✓/⚠ badge · sources  │
                    └──────────┬────────────┘
                               │ HTTP POST /query
                               ▼
                    ┌───────────────────────┐
                    │       FastAPI         │
                    │      (api.py)         │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │    Query Rewriter     │
                    │ (rag/query_rewriter)  │
                    │ expand · classify     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │     Agent Router      │
                    │  LLM intent router    │
                    │  gpt-4o classifier    │
                    └──────────┬────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌──────────────────────────────────────┐    ┌──────────────────┐
│ Graph Search  │    │         Vector Search Path           │    │   Calculator     │
│   (Neo4j)     │    │                                      │    │ (safe eval)      │
│               │    │  ┌─────────────┐  ┌──────────────┐  │    └──────────────────┘
│ executives    │    │  │Dense Retriev│  │BM25 Retrieval│  │
│ risks         │    │  │(ChromaDB +  │  │(rank-bm25,   │  │
│ metrics       │    │  │ text-embed- │  │ persisted    │  │
│ regulations   │    │  │ 3-small)    │  │ corpus)      │  │
└──────┬────────┘    │  └──────┬──────┘  └──────┬───────┘  │
       │             │         │                 │          │
       │             │         └────────┬────────┘          │
       │             │                  ▼                   │
       │             │    ┌─────────────────────────┐       │
       │             │    │    RRF Fusion  (k=60)   │       │
       │             │    │  score = Σ 1/(60+rank)  │       │
       │             │    │  per retriever           │       │
       │             │    └─────────────┬───────────┘       │
       │             └─────────────────-│───────────────────┘
       │                                │
       └────────────────────────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Cross-Encoder       │
                    │   Reranker            │
                    │ ms-marco-MiniLM-L-6   │
                    │ top_k=6 for generation│
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Two-Stage Generator │
                    │      (gpt-4o)         │
                    │ Stage 1: fact extract │
                    │ Stage 2: cite+synth   │
                    │ + token logprobs      │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │  build_finvault_span  │
                    │  (astr_o_adapter.py)  │
                    │ chunk_id · scores ·   │
                    │ answer · logprobs     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   ASTR-O Pipeline     │
                    │  (~/astr-o package)   │
                    │                       │
                    │ L1  Metadata bridge   │
                    │     registry auth     │
                    │ L4  Contradiction     │
                    │     detector          │
                    │ L5  Token confidence  │
                    │     (logprob ≥ −0.25) │
                    │ L6  Source verif.     │
                    │     AUTHORITATIVE?    │
                    │ L6b Groundedness      │
                    │     word overlap=1.0  │
                    │ L7  Criteria gate     │
                    │     SAFE / FLAGGED    │
                    │ L8  Causal chain      │
                    │     (FLAGGED only)    │
                    │ L9  HMAC-SHA256 sign  │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │   Trust Surface       │
                    │  format_response()    │
                    │                       │
                    │ SAFE   → ✓ Verified   │
                    │ FLAGGED→ ⚠ Unverified │
                    │ PARTIAL→ ⚠ Partial    │
                    │ ERROR  → ✗ (no answer)│
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │    API Response       │
                    │ answer · badge ·      │
                    │ sources · metrics ·   │
                    │ audit signature       │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌───────────────────────┐
                    │    Streamlit UI       │
                    │ st.success / warning  │
                    │ tier badges 🔴🟡⚪    │
                    │ sources expander      │
                    └───────────────────────┘

Supporting stores
─────────────────
  ChromaDB      ./vector_db/               dense vectors (61 chunks)
  BM25 corpus   ./vector_db/bm25_corpus.json  keyword index (61 chunks)
  Neo4j         bolt://localhost:7687       knowledge graph
  ASTR-O hot    ./astr_o_data/hot/         recent signed reports
  ASTR-O cold   ./astr_o_data/cold/        archived reports
  Registry      ./reference_registry.json  SHA-256 + HMAC-signed doc hashes

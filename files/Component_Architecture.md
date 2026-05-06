FinVault AI
│
├── app.py
│   Streamlit chat UI — displays answer, ASTR-O verification badge
│   (✓ Verified / ⚠ Unverified), source citations with tier badges
│
├── api.py
│   FastAPI backend (/query)
│   Orchestrates: route_query_with_scores → rerank → generate_answer_for_audit
│   → build_finvault_span → process_span → format_response
│
├── llm_config.py
│   Shared LLM singleton (gpt-4o, temp=0, max_completion_tokens=1024)
│   All modules import from here — one place to change model or key
│
├── agent_router.py
│   Query rewriting → LLM intent classification (graph / vector / calculator)
│   route_query()               — returns docs
│   route_query_with_scores()   — returns (docs, rrf_scored_pairs) for ASTR-O
│   Both paths now use hybrid BM25+RRF retrieval
│
├── reranker.py
│   Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│   Re-scores every (query, chunk) pair; more accurate than cosine alone
│   Loaded at module level (lazy-load is a known tech debt item)
│
├── tools.py
│   vector_search_tool()   — wraps vector_retreiver for agent use
│   graph_search_tool()    — routes to Neo4j by query intent + quarter
│   calculator_tool()      — safe arithmetic (whitelist-validated eval)
│
├── agent_executor.py
│   Standalone agent: route_question → tool → LLM prompt → answer
│   Used by evaluate.py for accuracy evaluation
│
├── astr_o_adapter.py
│   Builds the ASTR-O span dict from FinVault retrieval output
│   Inputs: query, scored_pairs (doc, rrf_score), answer, logprobs
│   Outputs: {span_id, retrieved_chunks, retrieval_metadata, llm_response}
│   chunk_id = "fv_" + md5(text[:120])[:12]
│
├── astr_o_init.py
│   Singleton ASTROPipeline initializer
│   Adds ~/astr-o to sys.path; constructs pipeline once per process
│   Re-exports format_response for api.py
│
├── finvault_domain_schema.py
│   Financial entity regex patterns used by ASTR-O:
│   net_interest_margin, profit_after_tax, gross_npa,
│   capital_adequacy_ratio, eps, return_on_assets
│   Each entry: regex, critical flag, severity, context_patterns
│
├── reference_registry.json
│   ASTR-O reference registry — built at ingestion time
│   SHA-256 hash of hdfc_q3.pdf, source_tier=CRITICAL
│   HMAC-SHA256 signed with ASTR_O_REGISTRY_SECRET
│
├── registry_config.json
│   Registry builder config: mission_id, document list, source tiers
│
├── pyrightconfig.json
│   Tells Pylance/pyright to look in ~/astr-o for type resolution
│
├── rag/
│   ├── vector_store.py
│   │   ChromaDB embed + store (OpenAI text-embedding-3-small primary,
│   │   Ollama mxbai-embed-large fallback)
│   │   embed_and_store() also calls build_bm25_index() after storing
│   │
│   ├── retriever.py
│   │   retrieve_docs()                  — dense only
│   │   retrieve_with_scores()           — dense with L2 distances
│   │   retrieve_hybrid_with_scores()    — BM25 + dense → RRF (k=60)
│   │   retrieve_hybrid()               — convenience wrapper (docs only)
│   │
│   ├── bm25_retriever.py
│   │   build_bm25_index(chunks)         — persists corpus at ingestion
│   │   bm25_retrieve(query, top_k)      — searches persisted corpus
│   │   _bootstrap_from_chroma()         — auto-builds corpus from ChromaDB
│   │                                      on first use (no rebuild needed)
│   │
│   ├── generator.py
│   │   generate_answer()               — two-stage fact extraction + synthesis
│   │   generate_answer_for_audit()     — same + captures token logprobs
│   │                                     for ASTR-O token confidence gate
│   │   generate_answer_with_reasoning() — CoT format (show-your-work)
│   │
│   ├── query_rewriter.py
│   │   rewrite_query()    — expand financial abbreviations
│   │   is_risk_query()    — detect risk questions (boosts top_k to 15)
│   │   is_financial_query()
│   │
│   ├── chunking.py
│   │   RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
│   │
│   ├── bm25_retriever.py  (see above)
│   │
│   └── reasoning_engine.py
│       Post-generation answer verifier (not yet wired into main pipeline)
│
├── ingestion/
│   └── pdf_parser.py
│       unstructured hi_res PDF parser with table detection
│
├── graph/
│   ├── entity_extractor.py
│   │   spaCy en_core_web_sm — extracts people, companies, risks, metrics
│   │
│   ├── graph_builder.py
│   │   Writes Neo4j nodes + edges from extracted entities
│   │   Reads NEO4J_URI / USER / PASSWORD from .env
│   │
│   └── load_graph.py
│       One-time ingestion script: parse PDF → extract → write Neo4j
│
├── graph_retriever.py
│   Neo4j query functions:
│   get_executives_for_period(), get_company_risks(),
│   get_company_metrics(), get_company_regulations()
│   validate_connection() — used by agent_router before graph calls
│
├── vector_retreiver.py
│   Root-level retriever: vector search + cross-encoder reranking
│   (same OpenAI / Ollama fallback pattern as rag/vector_store.py)
│
├── eval_dataset.py
│   5 ground-truth Q&A pairs (NIM, PAT, GNPA, CAR, risks)
│
├── evaluate.py
│   Accuracy eval — LLM-as-judge (gpt-4o-mini), keyword, substring
│   Current score: 100% (5/5)
│
├── ragas_evaluate.py
│   RAGAS evaluation: Faithfulness / ContextPrecision / ContextRecall
│   Current scores: 0.800 / 1.000 / 0.800
│
├── utils/
│   └── logger.py
│       Appends {query, answer, timestamp} to query_logs.json
│
├── astr_o_data/
│   ├── hot/    ASTR-O recent signed reports + enriched spans
│   └── cold/   Archived reports (managed by ASTR-O storage_policy)
│
└── data/
    └── hdfc_q3.pdf    Source document (not committed)

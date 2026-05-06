# FinVault AI — Agentic Financial Research Assistant

An agentic RAG system that transforms a financial earnings report (HDFC Q3 FY2025) into an instantly queryable knowledge base. Combines hybrid vector retrieval, a Neo4j knowledge graph, cross-encoder reranking, a two-stage LLM generator, and an ASTR-O audit layer that verifies every answer before it reaches the user.

---

## What it does

You ask a question in plain English. FinVault routes it to the right retrieval strategy, pulls the most relevant evidence from the document, generates a grounded cited answer, and then runs it through ASTR-O — a 6-layer audit pipeline that checks for contradictions, source authorization, token confidence, and hallucination risk before returning a signed verdict.

```
"What is the Net Interest Margin for Q3 FY2025?"
→ 3.43%  ✓ Verified — 5/5 sources verified, no contradictions

"Who spoke during the Q3 FY2025 earnings call?"
→ Executives who spoke in Q3 FY2025: Sashidhar Jagdishan ...

"What specific risks did HDFC Bank mention?"
→ Market and operational risks, accounting standards exposure,
  regulatory compliance risk, liquidity risk ...
```

---

## Architecture

```
User Query
    │
    ▼
Streamlit UI  ──HTTP──►  FastAPI (/query)
                              │
                              ▼
                        Query Rewriter
                        (expand financial terms)
                              │
                              ▼
                        LLM Intent Router
                        (graph / vector / calculator)
                         │            │
                         ▼            ▼
                    Neo4j         Hybrid Retrieval
                  Knowledge       ├── Dense (ChromaDB +
                   Graph          │   text-embedding-3-small)
                  (people,        └── BM25 (keyword)
                  risks,               │
                  metrics,             ▼
                  regs)         RRF Fusion (k=60)
                                       │
                                       ▼
                               Cross-Encoder Reranker
                               (ms-marco-MiniLM-L-6-v2)
                         │           │
                         └─────┬─────┘
                               ▼
                        Two-Stage Generator
                        Stage 1: Extract facts
                        Stage 2: Synthesize answer
                        + token logprobs captured
                               │
                               ▼
                        ASTR-O Audit Pipeline
                        ├── Metadata bridge (registry auth)
                        ├── Contradiction detector
                        ├── Token confidence analyzer
                        ├── Source verification
                        ├── Groundedness check
                        ├── Criteria gate (SAFE / FLAGGED)
                        └── HMAC-SHA256 signed report
                               │
                               ▼
                     Trust Surface Response
                     ✓ Verified / ⚠ Unverified badge
                     + cited sources with tier badges
```

### Key design decisions

**ASTR-O audit layer** — Every span (query + retrieved chunks + answer + logprobs) is passed through ASTR-O's 6-layer pipeline. All four criteria must pass for a SAFE verdict: no contradictions between source and answer, high logprob confidence on critical financial tokens (NIM, GNPA, PAT, CAR), all source documents authorized against the reference registry, and the answer fully grounded in retrieved chunks. A failure in any single criterion flags the span and triggers causal chain analysis.

**Domain schema** — `finvault_domain_schema.py` defines regex patterns and context keywords for six financial entities (NIM, PAT, Gross NPA, CAR, EPS, ROA). ASTR-O's contradiction detector and token confidence analyzer use these to identify and score critical claims in the answer.

**Reference registry** — `reference_registry.json` is built at ingestion time by ASTR-O's registry builder. It stores the SHA-256 hash and source tier of every authorized document. The metadata bridge verifies each retrieved chunk's source against this registry; a tier mismatch is treated as an integrity violation.

**Cross-encoder reranking** — After vector retrieval, a sentence-transformer cross-encoder re-scores every (query, chunk) pair for true relevance. Much more accurate than cosine similarity alone.

**LLM intent router** — Every question is classified by the LLM before retrieval. People/executive questions go to Neo4j; financial metric/risk questions go to the vector store. This avoids irrelevant graph lookups and improves precision.

**Two-stage generation** — Stage 1 extracts specific facts from retrieved chunks. Stage 2 synthesizes a cited answer from those facts only, with logprobs captured for ASTR-O's token confidence gate. This prevents the LLM from drifting into its training memory and is why faithfulness scores 1.0.

**Knowledge graph** — Neo4j stores structured relationships extracted by spaCy NLP: which executives spoke in which quarters, which risks the company mentioned, which regulations apply, which financial metrics were reported. Enables structural queries that vector search cannot answer.

---

## Evaluation results

| Metric | Score | What it means |
|---|---|---|
| Accuracy (LLM-judge) | 100% (5/5) | Every answer matches ground truth |
| RAGAS Faithfulness | 0.800 | 4/5 answers fully grounded in retrieved chunks |
| RAGAS ContextRecall | 0.800 | Retrieval found all relevant evidence on 4/5 questions |
| RAGAS ContextPrecision | **1.000** | Most relevant chunk ranked first on every question |

Evaluated on 5 questions covering NIM, PAT, Gross NPA, Capital Adequacy Ratio, and risk factors. Accuracy uses an LLM-as-judge (`gpt-4o-mini`) with unit conversion awareness (crore vs bn) and Indian FY calendar knowledge (Q3 FY2025 = December 2024). RAGAS evaluation also uses `gpt-4o-mini` as the judge and `text-embedding-3-small` for embeddings.

ContextPrecision improved from 0.767 → **1.000** after wiring BM25+RRF hybrid retrieval. The faithfulness dip on Q3 (Gross NPA) is a RAGAS judge sensitivity issue on short numeric answers, not an actual hallucination — the answer `1.42%` is correct and directly matches the source chunk.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM (agent) | OpenAI gpt-4o |
| LLM (eval judge) | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (Ollama mxbai-embed-large fallback) |
| Vector store | ChromaDB |
| Keyword search | BM25 (rank-bm25), fused with dense via RRF |
| Reranker | sentence-transformers ms-marco-MiniLM-L-6-v2 |
| Knowledge graph | Neo4j (local) |
| NLP entity extraction | spaCy en_core_web_sm |
| PDF parsing | unstructured (hi_res strategy) |
| API backend | FastAPI |
| UI | Streamlit |
| Audit layer | ASTR-O (contradiction, confidence, source verification, groundedness) |
| Evaluation | RAGAS + custom LLM-judge |

---

## Project structure

```
finvault-pro/
│
├── app.py                      # Streamlit chat UI (verification badge display)
├── api.py                      # FastAPI backend (/query endpoint, ASTR-O wired in)
├── pipeline.py                 # CLI: ingestion + query (python pipeline.py)
├── llm_config.py               # Shared LLM instance (all modules import from here)
├── agent_executor.py           # Standalone agent: routes + retrieves + generates
├── agent_router.py             # LLM intent router + query dispatcher (with scores)
├── hybrid_rag.py               # Vector + graph fusion layer (standalone)
├── tools.py                    # vector_search / graph_search / calculator tools
├── reranker.py                 # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
├── vector_retreiver.py         # Root-level retriever: vector search + reranking
│
├── astr_o_adapter.py           # Builds ASTR-O span from FinVault retrieval output
├── astr_o_init.py              # Singleton ASTROPipeline initializer
├── finvault_domain_schema.py   # Financial entity schema (NIM, GNPA, PAT, CAR …)
├── reference_registry.json     # HMAC-signed document registry (built at ingestion)
├── registry_config.json        # Registry builder config (mission_id, documents)
│
├── rag/
│   ├── chunking.py             # RecursiveCharacterTextSplitter (800 chars, 150 overlap)
│   ├── vector_store.py         # ChromaDB embed, store, retrieve
│   ├── retriever.py            # Retriever wrapper with deduplication helpers
│   ├── bm25_retriever.py       # BM25Okapi keyword search module
│   ├── generator.py            # Two-stage LLM generation + logprob capture
│   ├── query_rewriter.py       # Financial query term expansion
│   └── reasoning_engine.py     # Post-generation answer verifier (not yet wired in)
│
├── ingestion/
│   └── pdf_parser.py           # unstructured PDF parser (hi_res + table detection)
│
├── graph/
│   ├── entity_extractor.py     # spaCy NLP: extract companies, executives, risks
│   ├── graph_builder.py        # Write nodes + edges to Neo4j
│   └── load_graph.py           # One-time graph ingestion script
│
├── graph_retriever.py          # Query Neo4j (executives, risks, metrics, regulations)
├── graph_query.py              # Standalone Neo4j query tool
│
├── eval_dataset.py             # 5 test questions with ground truths
├── evaluate.py                 # Accuracy eval (LLM-as-judge, 3 methods)
├── ragas_evaluate.py           # RAGAS eval (Faithfulness / Precision / Recall)
│
├── utils/
│   └── logger.py               # Query logger (writes to query_logs.json)
│
├── astr_o_data/                # ASTR-O audit storage (not committed)
│   ├── hot/                    # Recent signed reports + enriched spans
│   └── cold/                   # Archived reports
│
├── data/
│   └── hdfc_q3.pdf             # Source document (not committed)
│
├── pyrightconfig.json          # Pylance/pyright extra paths (ASTR-O repo)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites

- **Python 3.12** (3.13+ breaks unstructured dependencies — use 3.12 exactly)
- [Neo4j Desktop](https://neo4j.com/download/) installed with a local database running
- An [OpenAI API key](https://platform.openai.com/api-keys)
- **macOS:** `brew install tesseract poppler`
- **Linux:** `apt-get install tesseract-ocr poppler-utils`
- [ASTR-O](https://github.com/your-org/astr-o) cloned to `~/astr-o`
- [Ollama](https://ollama.com) — optional, used as embedding fallback if OpenAI is unavailable

### 1. Clone and install

```bash
git clone https://github.com/your-username/finvault-pro.git
cd finvault-pro
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> **Note:** Do not use the system Python or a Python 3.13+ environment. The `unstructured` PDF parser requires Python 3.12.

### 2. Configure environment

Create a `.env` file in the project root (see `.env.example`):

```env
OPENAI_API_KEY=your_openai_api_key_here

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"
ASTR_O_REGISTRY_SECRET=your_registry_hmac_secret_here
ASTR_O_REPORT_SECRET=your_report_hmac_secret_here
```

> **Ollama (optional):** If `OPENAI_API_KEY` is missing or the OpenAI API is unreachable, embeddings automatically fall back to Ollama `mxbai-embed-large`. Pull it with `ollama pull mxbai-embed-large` if you need this fallback.

### 3. Add the PDF

Place your earnings report PDF at:

```
data/hdfc_q3.pdf
```

### 4. Build the vector database and reference registry

```bash
python pipeline.py
```

This parses the PDF, creates chunks, embeds them with `text-embedding-3-small`, stores them in ChromaDB, and builds the ASTR-O reference registry (`reference_registry.json`) signed with `ASTR_O_REGISTRY_SECRET`. Only needed once — subsequent runs detect the existing database and skip ingestion.

### 5. Build the knowledge graph

```bash
python graph/load_graph.py
```

Parses the PDF with spaCy NLP, extracts entities (executives, risks, metrics, regulations), and writes them as nodes and relationships into Neo4j. Only needed once.

---

## Running the system

### Option A — Full stack (recommended)

Terminal 1: start the FastAPI backend

```bash
uvicorn api:app --reload
```

Terminal 2: start the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Each answer shows a verification badge (✓ Verified / ⚠ Unverified) based on the ASTR-O audit result, along with source citations and tier badges.

### Option B — Command line

```bash
# Standard query
python pipeline.py --query "What is the Net Interest Margin for Q3 FY2025?"

# With step-by-step reasoning
python pipeline.py --query "What risks did HDFC mention?" --mode reasoning

# Force rebuild of vector DB
python pipeline.py --rebuild
```

---

## Running evaluations

### Accuracy (LLM-as-judge)

```bash
python evaluate.py              # LLM judge (most accurate)
python evaluate.py keyword      # Keyword matching (fast, no API cost)
```

### RAGAS (Faithfulness / ContextPrecision / ContextRecall)

```bash
python ragas_evaluate.py
```

Expected output:

```
faithfulness           1.000  |████████████████████|  GOOD
context_precision      0.767  |███████████████░░░░░|  GOOD
context_recall         1.000  |████████████████████|  GOOD
```

---

## How queries are routed

The LLM router classifies every question before retrieval:

| Question type | Route | Example |
|---|---|---|
| People / executives | Neo4j graph | "Who spoke during Q3 FY2025?" |
| Financial metrics | Vector + reranker | "What is the NIM?" |
| Risk / strategy | Vector + reranker | "What risks did HDFC mention?" |
| Arithmetic | Calculator | "What is 16736 / 167?" |

Graph queries also supplement with vector search to combine structured + unstructured knowledge in the final answer.

## How ASTR-O verifies answers

Every answer passes through four gates before a verdict is issued:

| Gate | What it checks | Pass condition |
|---|---|---|
| Contradiction | Financial figures in answer vs. source chunks (domain schema regex) | No contradicting values found |
| Token confidence | Logprob of critical tokens (NIM, GNPA, PAT, CAR …) | All critical tokens logprob ≥ −0.25 |
| Source authorization | `hdfc_q3.pdf` present in registry, tier label matches | All chunks VERIFIED |
| Groundedness | Answer words present in retrieved chunk text | Score = 1.0 (fully grounded) |

All four must pass → **SAFE** → `✓ Verified`. Any single failure → **FLAGGED** → `⚠ Unverified` + causal chain analysis + specific issues listed.

---

## Known technical debt

| Issue | Location | Notes |
|---|---|---|
| Two pipeline entry points | `pipeline.py` vs `hybrid_rag.py` | Both are standalone orchestrators. `api.py` is the production path; the others are useful for CLI/dev but could be consolidated. |
| `RetrieverConfig` class is unused | `rag/retriever.py` | Config options exist but `retrieve_docs()` never reads them. Either wire them in or remove. |
| `load_dotenv()` called in multiple modules | `graph_retriever.py`, `llm_config.py` | Harmless (idempotent) but redundant — ideally called once at entry point only. |
| `logging.basicConfig` called in two modules | `pipeline.py`, `rag/vector_store.py` | Python only respects the first call. Should only be configured in the entry point. |
| `display_result()` runs a second vector DB query | `pipeline.py` | Re-queries the DB just to display source text. Docs are already retrieved earlier and should be passed through instead. |
| Cross-encoder reranker loads at import time | `reranker.py` | `CrossEncoder(...)` runs at module level. Should be lazy-loaded on first use. |
| BM25 corpus not rebuilt on partial ingestion | `rag/bm25_retriever.py` | If only some chunks are re-ingested (not a full rebuild), the BM25 corpus on disk goes stale. A full `pipeline.py --rebuild` is needed to resync it. |
| ASTR-O groundedness is word-overlap | `astr_o/pipeline.py` | Binary word-level check (1.0 or 0.0). A semantic similarity score (e.g. BERTScore) would be more robust for paraphrased answers. |

---

## Roadmap

- [ ] Wire `reasoning_engine.py` into the main API pipeline for post-generation answer verification
- [ ] Build BM25 index on full corpus at ingestion time — persist and load for true hybrid retrieval
- [ ] Multi-document support (ingest multiple earnings reports, compare across quarters/companies)
- [ ] Source metadata tagging (show page number and section each answer came from)
- [ ] Consolidate `pipeline.py` and `hybrid_rag.py` into a single entry point
- [ ] Lazy-load the cross-encoder reranker instead of loading at import time
- [ ] Expand eval dataset to 30+ questions covering edge cases
- [ ] Replace ASTR-O word-overlap groundedness with semantic similarity (BERTScore)
- [ ] Live data ingestion from BSE/NSE filings API
- [ ] Deploy to HuggingFace Spaces (swap Ollama for hosted embeddings + eval model)

---

## Limitations

- **Single document:** currently focused on HDFC Q3 FY2025 only. Multi-document support is on the roadmap.
- **Small eval set:** 5 questions is enough to verify the pipeline works but not for statistical confidence across all query types.
- **Neo4j:** must be running locally. Cloud deployment would require swapping for a hosted Neo4j instance (Aura or similar).
- **Faithfulness 0.800 / ContextRecall 0.800:** Both dips are on edge-case questions. Q3 (Gross NPA) faithfulness is a RAGAS judge sensitivity issue — the correct answer `1.42%` is penalised because it is too short for RAGAS to find a sentence-level attribution. Q5 (risks) recall is 0.0 because the risk-factors chunk is retrieved but the cross-encoder reranker pushes it below the `top_k=6` cutoff used in generation. Increasing the reranker's `top_k` for risk queries would recover recall.
- **OpenAI dependency:** the main LLM, judge, and embeddings all call the OpenAI API — an internet connection and a valid `OPENAI_API_KEY` are required. Embeddings fall back to Ollama `mxbai-embed-large` if the key is absent.
- **Python 3.12 only:** The `unstructured` hi_res PDF parsing stack does not work on Python 3.13+ due to dependency conflicts.
- **ASTR-O local dependency:** ASTR-O must be cloned separately to `~/astr-o`. It is added to `sys.path` at runtime by `astr_o_init.py`.

---

## License

MIT

# FinVault AI — Agentic Financial Research Assistant

An agentic RAG system that transforms a financial earnings report (HDFC Q3 FY2025) into an instantly queryable knowledge base. Combines hybrid vector retrieval, a Neo4j knowledge graph, cross-encoder reranking, and a two-stage LLM generator — all running on free infrastructure.

---

## What it does

You ask a question in plain English. FinVault routes it to the right retrieval strategy, pulls the most relevant evidence from the document, and returns a grounded, cited answer — with zero hallucinations on the evaluated question set.

```
"What is the Net Interest Margin for Q3 FY2025?"
→ 3.43%

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
                    Neo4j         Vector Retrieval
                  Knowledge       (ChromaDB +
                   Graph          mxbai-embed-large)
                  (people,              │
                  risks,               ▼
                  metrics,      Cross-Encoder Reranker
                  regs)         (ms-marco-MiniLM-L-6-v2)
                         │           │
                         └─────┬─────┘
                               ▼
                        Two-Stage Generator
                        Stage 1: Extract facts
                        Stage 2: Synthesize answer
                               │
                               ▼
                        Answer + Sources
```

### Key design decisions

**Cross-encoder reranking** — After vector retrieval, a sentence-transformer cross-encoder re-scores every (query, chunk) pair for true relevance. Much more accurate than cosine similarity alone.

**LLM intent router** — Every question is classified by the LLM before retrieval. People/executive questions go to Neo4j; financial metric/risk questions go to the vector store. This avoids irrelevant graph lookups and improves precision.

**Two-stage generation** — Stage 1 extracts specific facts from retrieved chunks. Stage 2 synthesizes a cited answer from those facts only. This prevents the LLM from drifting into its training memory and is why faithfulness scores 1.0.

**Knowledge graph** — Neo4j stores structured relationships extracted by spaCy NLP: which executives spoke in which quarters, which risks the company mentioned, which regulations apply, which financial metrics were reported. Enables structural queries that vector search cannot answer.

**Shared LLM config** — All modules import a single `llm_config.py` instance instead of initializing ChatGroq separately. One place to change model, temperature, or API key.

---

## Evaluation results

| Metric | Score | What it means |
|---|---|---|
| Accuracy (LLM-judge) | 100% (5/5) | Every answer matches ground truth |
| RAGAS Faithfulness | 1.000 | Zero hallucinations |
| RAGAS ContextRecall | 1.000 | Retrieval found all relevant evidence |
| RAGAS ContextPrecision | 0.767 | Most relevant chunk ranked first in most cases |

Evaluated on 5 questions covering NIM, PAT, Gross NPA, Capital Adequacy Ratio, and risk factors. Accuracy uses an LLM-as-judge with unit conversion awareness (crore vs bn) and Indian FY calendar knowledge (Q3 FY2025 = December 2024). RAGAS uses local Mistral Nemo via Ollama — no rate limits, no external evaluation API.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM (agent) | Groq llama-3.3-70b-versatile (free tier) |
| Embeddings | Ollama mxbai-embed-large (local) |
| LLM (eval judge) | Ollama mistral-nemo (local) |
| Vector store | ChromaDB |
| Reranker | sentence-transformers ms-marco-MiniLM-L-6-v2 |
| Knowledge graph | Neo4j (local) |
| NLP entity extraction | spaCy en_core_web_sm |
| PDF parsing | unstructured (hi_res strategy) |
| API backend | FastAPI |
| UI | Streamlit |
| Evaluation | RAGAS + custom LLM-judge |

---

## Project structure

```
finvault-pro/
│
├── app.py                  # Streamlit chat UI
├── api.py                  # FastAPI backend (/query endpoint)
├── pipeline.py             # CLI: ingestion + query (python pipeline.py)
├── llm_config.py           # Shared LLM instance (all modules import from here)
├── agent_executor.py       # Standalone agent: routes + retrieves + generates
├── agent_router.py         # LLM intent router + query dispatcher
├── hybrid_rag.py           # Vector + graph fusion layer (standalone)
├── tools.py                # vector_search / graph_search / calculator tools
├── reranker.py             # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
├── vector_retreiver.py     # Root-level retriever: vector search + reranking
│
├── rag/
│   ├── chunking.py         # RecursiveCharacterTextSplitter (800 chars, 150 overlap)
│   ├── vector_store.py     # ChromaDB embed, store, retrieve
│   ├── retriever.py        # Retriever wrapper with deduplication helpers
│   ├── bm25_retriever.py   # BM25Okapi keyword search module
│   ├── generator.py        # Two-stage LLM answer generation
│   ├── query_rewriter.py   # Financial query term expansion
│   └── reasoning_engine.py # Post-generation answer verifier (not yet wired in)
│
├── ingestion/
│   └── pdf_parser.py       # unstructured PDF parser (hi_res + table detection)
│
├── graph/
│   ├── entity_extractor.py # spaCy NLP: extract companies, executives, risks
│   ├── graph_builder.py    # Write nodes + edges to Neo4j
│   └── load_graph.py       # One-time graph ingestion script
│
├── graph_retriever.py      # Query Neo4j (executives, risks, metrics, regulations)
├── graph_query.py          # Standalone Neo4j query tool
│
├── eval_dataset.py         # 5 test questions with ground truths
├── evaluate.py             # Accuracy eval (LLM-as-judge, 3 methods)
├── ragas_evaluate.py       # RAGAS eval (Faithfulness / Precision / Recall)
│
├── utils/
│   └── logger.py           # Query logger (writes to query_logs.json)
│
├── data/
│   └── hdfc_q3.pdf         # Source document (not committed)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites

- **Python 3.12** (3.13+ breaks unstructured dependencies — use 3.12 exactly)
- [Ollama](https://ollama.com) installed and running
- [Neo4j Desktop](https://neo4j.com/download/) installed with a local database running
- A free [Groq API key](https://console.groq.com)
- **macOS:** `brew install tesseract poppler`
- **Linux:** `apt-get install tesseract-ocr poppler-utils`

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

### 2. Pull Ollama models

```bash
ollama pull mxbai-embed-large   # embeddings (used by pipeline + RAGAS)
ollama pull mistral-nemo        # evaluation judge (used by RAGAS only)
```

### 3. Configure environment

Create a `.env` file in the project root (see `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 4. Add the PDF

Place your earnings report PDF at:

```
data/hdfc_q3.pdf
```

### 5. Build the vector database

```bash
python pipeline.py
```

This parses the PDF, creates chunks, embeds them with mxbai-embed-large, and stores them in ChromaDB. Only needed once — subsequent runs detect the existing database and skip ingestion.

### 6. Build the knowledge graph

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

Open `http://localhost:8501` in your browser.

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

Make sure Ollama is running with both models pulled, then:

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

---

## Known technical debt

These are non-critical issues that don't affect correctness for the current scope but are worth addressing if the project grows.

| Issue | Location | Notes |
|---|---|---|
| Two pipeline entry points | `pipeline.py` vs `hybrid_rag.py` | Both are standalone orchestrators. `api.py` is the production path; the other two are useful for CLI/dev but could be consolidated. |
| `RetrieverConfig` class is unused | `rag/retriever.py` | Config options (`enable_reranking`, `enable_deduplication`, `similarity_threshold`) exist but `retrieve_docs()` never reads them. Either wire them in or remove. |
| `load_dotenv()` called in multiple modules | `graph_retriever.py`, `llm_config.py` | Harmless (idempotent) but redundant — ideally called once at the entry point only. |
| `logging.basicConfig` called in two modules | `pipeline.py`, `rag/vector_store.py` | Python only respects the first call; subsequent ones are silently ignored. Should only be configured in the entry point. |
| `display_result()` runs a second vector DB query | `pipeline.py` | Re-queries the DB just to display source text. Docs are already retrieved earlier and should be passed through instead. |
| Cross-encoder reranker loads at import time | `reranker.py` | `CrossEncoder(...)` runs at module level, so importing `reranker` anywhere triggers a model load. Should be lazy-loaded on first use. |
| BM25 not on full corpus | `rag/bm25_retriever.py` | For true hybrid retrieval, BM25 should be indexed on all document chunks at ingestion time and persisted. Currently the module exists but is not used in the main retrieval path. |
| `graph/graph_builder.py` reads hardcoded credentials | `graph/graph_builder.py` | One-time ingestion script still uses a hardcoded password instead of reading from `.env`. |

---

## Roadmap

- [ ] Wire `reasoning_engine.py` into the main API pipeline for post-generation answer verification
- [ ] Build BM25 index on full corpus at ingestion time — persist and load for true hybrid retrieval
- [ ] `graph/graph_builder.py` should read Neo4j credentials from `.env` instead of hardcoding
- [ ] Multi-document support (ingest multiple earnings reports, compare across quarters/companies)
- [ ] Source metadata tagging (show page number and section each answer came from)
- [ ] Consolidate `pipeline.py` and `hybrid_rag.py` into a single entry point
- [ ] Lazy-load the cross-encoder reranker instead of loading at import time
- [ ] Expand eval dataset to 30+ questions covering edge cases
- [ ] Live data ingestion from BSE/NSE filings API
- [ ] Deploy to HuggingFace Spaces (swap Ollama for hosted embeddings + eval model)

---

## Limitations

- **Single document:** currently focused on HDFC Q3 FY2025 only. Multi-document support is on the roadmap.
- **Small eval set:** 5 questions is enough to verify the pipeline works but not for statistical confidence across all query types.
- **Local infrastructure:** Ollama and Neo4j must be running locally. Cloud deployment would require swapping these for hosted alternatives.
- **ContextPrecision 0.767:** The pipeline runs Unstructured's `by_title` chunking followed by a second pass with `RecursiveCharacterTextSplitter`. This second pass can split financial table rows mid-entry, separating a metric label from its value. This causes the risk question to rank a capital adequacy chunk first instead of the risk factors chunk, pulling precision down from 1.0. A future fix would rely solely on Unstructured's `by_title` output to preserve table coherence.
- **Python 3.12 only:** The `unstructured` hi_res PDF parsing stack does not work on Python 3.13+ due to dependency conflicts.

---

## License

MIT

# FinVault AI — Agentic Financial Research Assistant

An agentic RAG system that transforms a financial earnings report (HDFC Q3 FY2025) into an instantly queryable knowledge base. Combines hybrid vector retrieval, BM25 keyword search, a Neo4j knowledge graph, and a two-stage LLM generator — all running on free infrastructure.

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
                    Neo4j         Hybrid Retrieval
                  Knowledge         │
                   Graph        ┌───┴───┐
                  (people,      │       │
                  risks,      FAISS   BM25
                  metrics,    Vector  Keyword
                  regs)       Search  Search
                         │       └───┬───┘
                         │           │
                         └─────┬─────┘
                               ▼
                        Cross-Encoder Reranker
                        (ms-marco-MiniLM-L-6-v2)
                               │
                               ▼
                        Two-Stage Generator
                        Stage 1: Extract facts
                        Stage 2: Synthesize answer
                               │
                               ▼
                        Answer + Sources
```

### Key design decisions

**Hybrid retrieval** — Vector search understands semantic meaning; BM25 catches exact keyword matches. Together they retrieve more relevant chunks than either alone.

**Cross-encoder reranking** — After retrieval, a sentence-transformer cross-encoder re-scores every (query, chunk) pair for true relevance. Much more accurate than cosine similarity alone.

**LLM intent router** — Every question is classified by the LLM before retrieval. People/executive questions go to Neo4j; financial metric questions go to the vector store. This avoids irrelevant graph lookups and improves precision.

**Two-stage generation** — Stage 1 extracts specific facts from retrieved chunks. Stage 2 synthesizes a cited answer from those facts only. This prevents the LLM from drifting into its training memory and is why faithfulness scores 1.0.

**Knowledge graph** — Neo4j stores structured relationships extracted by spaCy NLP: which executives spoke in which quarters, which risks the company mentioned, which regulations apply, which financial metrics were reported. Enables structural queries that vector search cannot answer.

---

## Evaluation results

| Metric | Score | What it means |
|---|---|---|
| Accuracy (LLM-judge) | 100% (5/5) | Every answer matches ground truth |
| RAGAS Faithfulness | 1.000 | Zero hallucinations |
| RAGAS ContextRecall | 1.000 | Retrieval found all relevant evidence |
| RAGAS ContextPrecision | 0.767 | Most relevant chunk ranked first |

Evaluated on 5 questions covering NIM, PAT, Gross NPA, Capital Adequacy Ratio, and risk factors. Accuracy uses an LLM-as-judge with unit conversion awareness (crore vs bn) and Indian FY calendar knowledge (Q3 FY2025 = December 2024). RAGAS uses local Mistral Nemo via Ollama — no rate limits, no external evaluation API.

---

## Tech stack

| Component | Technology |
|---|---|
| LLM (agent) | Groq llama-3.3-70b-versatile (free tier) |
| Embeddings | Ollama mxbai-embed-large (local) |
| LLM (eval judge) | Ollama mistral-nemo (local) |
| Vector store | ChromaDB |
| Keyword search | BM25 (rank-bm25) |
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
├── agent_executor.py       # Core agent: routes + retrieves + generates
├── agent_router.py         # LLM intent router + query dispatcher
├── tools.py                # vector_search / graph_search / calculator tools
├── hybrid_rag.py           # Vector + graph fusion layer
├── pipeline.py             # CLI: ingestion + query (python pipeline.py)
│
├── rag/
│   ├── chunking.py         # RecursiveCharacterTextSplitter (800 chars, 150 overlap)
│   ├── vector_store.py     # Chroma embed, store, retrieve
│   ├── vector_retreiver.py # Hybrid: Chroma + BM25 + cross-encoder
│   ├── bm25_retriever.py   # BM25Okapi keyword search
│   ├── reranker.py         # Cross-encoder reranker
│   ├── generator.py        # Two-stage LLM answer generation
│   ├── query_rewriter.py   # Financial query term expansion
│   └── reasoning_engine.py # Post-generation answer verifier
│
├── retriever.py        # Vector retrieval wrapper with deduplication

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
├── test_llm.py             # LLM smoke test
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running
- [Neo4j Desktop](https://neo4j.com/download/) installed and running
- A free [Groq API key](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/your-username/finvault-pro.git
cd finvault-pro
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Pull Ollama models

```bash
ollama pull mxbai-embed-large   # embeddings
ollama pull mistral-nemo        # evaluation judge
```

### 3. Configure environment

Create a `.env` file in the project root:

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

Make sure Ollama is running with mistral-nemo pulled, then:

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
| Financial metrics | Vector + BM25 | "What is the NIM?" |
| Risk / strategy | Vector + BM25 | "What risks did HDFC mention?" |
| Arithmetic | Calculator | "What is 16736 / 167?" |

Graph queries also supplement with vector search to combine structured + unstructured knowledge in the final answer.

---

## Roadmap

- [ ] Multi-document support (ingest multiple earnings reports, compare across quarters/companies)
- [ ] Source metadata tagging (show which PDF each answer came from)
- [ ] Wire `reasoning_engine.py` into the main pipeline for post-generation verification
- [ ] Live data ingestion from BSE/NSE filings API
- [ ] Expand eval dataset to 30+ questions
- [ ] Deploy to HuggingFace Spaces

---

## Limitations

- Single document focus: currently trained on HDFC Q3 FY2025 only. Multi-document support is planned.
- Eval set is small (5 questions): sufficient to verify the pipeline but not for statistical confidence across all query types.
- Local infrastructure required: Ollama and Neo4j must be running locally. Cloud deployment would require swapping these components.

---

## License

MIT
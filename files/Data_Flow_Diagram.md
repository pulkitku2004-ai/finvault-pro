User Query
   │
   ▼
Streamlit UI
   │
   ▼
FastAPI Endpoint (/query)
   │
   ▼
Query Router
   │
   ├── Graph Query → Neo4j Knowledge Graph
   │
   ├── Vector Search → FAISS Embedding Index
   │
   └── BM25 Search → Keyword Retriever
   │
   ▼
Hybrid Retrieval
   │
   ▼
Reranker (Cross Encoder)
   │
   ▼
Top Relevant Context
   │
   ▼
LLM Generation (Llama3)
   │
   ▼
Answer Verification
   │
   ▼
API Response
   │
   ▼
Streamlit Chat UI
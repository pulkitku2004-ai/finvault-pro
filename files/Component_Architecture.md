FinVault AI
│
├── app.py
│   Streamlit Chat Interface
│
├── api.py
│   FastAPI backend
│
├── agent_router.py
│   Query classification and routing
│
├── reranker.py
│   Cross encoder ranking model
│
├── eval_dataset.py
│   RAG evaluation dataset
│
├── rag/
│   ├── generator.py
│   │   LLM answer generation
│   │
│   ├── reasoning_engine.py
│   │   Answer verification / correction
│   │
│   └── retriever.py
│       Hybrid search logic
│
├── utils/
│   └── logger.py
│       Query logging
│
├── ragas_evaluate.py
│   RAGAS evaluation pipeline
│
└── data/
    Financial report embeddings
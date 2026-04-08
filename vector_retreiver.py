
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Use your existing BM25
try:
    from rag.bm25_retriever import bm25_search
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Configuration
PERSIST_DIRECTORY = "./vector_db"
COLLECTION_NAME = "finvault_docs"

# Initialize embeddings and vector DB
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

try:
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
    DB_READY = True
except Exception as e:
    vector_retriever = None
    DB_READY = False
    print(f"⚠️  Vector DB error: {str(e)[:50]}")


def rerank_results(query, docs, top_k=6):
    """Try to use cross-encoder reranker, fallback to top-k slice."""
    if not docs:
        return []

    try:
        from reranker import rerank_results as cross_encoder_rerank
        return cross_encoder_rerank(query, docs, top_k=top_k)
    except Exception:
        return docs[:top_k]


def retrieve_docs(query):
    """Retrieve documents using vector search, then rerank with cross-encoder."""

    if not DB_READY or vector_retriever is None:
        return []

    try:
        vector_docs = vector_retriever.invoke(query)

        if not vector_docs:
            return []

        vector_texts = []
        for doc in vector_docs:
            if hasattr(doc, "page_content"):
                vector_texts.append(doc.page_content)
            else:
                vector_texts.append(str(doc))

        reranked = rerank_results(query, vector_texts, top_k=6)
        return reranked if reranked else vector_texts[:6]

    except Exception as e:
        print(f"Retrieval error: {str(e)[:50]}")
        return []
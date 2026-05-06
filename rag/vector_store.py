
import os
import shutil
import time
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = "./vector_db"
COLLECTION_NAME = "finvault_docs"

# Global instance (persist during session)
_vector_db_instance = None
_embeddings_instance = None


def get_embeddings():
    """OpenAI text-embedding-3-small (primary), Ollama mxbai-embed-large (fallback)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            from langchain_openai import OpenAIEmbeddings
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            _embeddings_instance = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
            )
            logger.info("Embeddings: OpenAI text-embedding-3-small")
        except Exception as e:
            logger.warning("OpenAI embeddings unavailable (%s) — falling back to Ollama mxbai-embed-large", e)
            from langchain_ollama import OllamaEmbeddings
            _embeddings_instance = OllamaEmbeddings(model="mxbai-embed-large")
            logger.info("Embeddings: Ollama mxbai-embed-large (fallback)")
    return _embeddings_instance


def ensure_clean_db_directory():
    """
    Remove vector DB directory completely.
    Critical after model changes or persistent errors.
    """
    global _vector_db_instance, _embeddings_instance
    _vector_db_instance = None
    _embeddings_instance = None

    # Reset chromadb's internal client state before deleting disk data
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            import chromadb
            from chromadb.config import Settings
            _tmp = chromadb.PersistentClient(
                path=PERSIST_DIRECTORY,
                settings=Settings(allow_reset=True),
            )
            _tmp.reset()
        except Exception:
            pass

        logger.info(f"🧹 Removing {PERSIST_DIRECTORY}...")
        try:
            shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
            time.sleep(1)
        except Exception as e:
            logger.warning(f"⚠️  Could not remove directory: {e}")

    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    logger.info(f"✅ Created clean {PERSIST_DIRECTORY}")


def embed_and_store(chunks: List[Document]) -> bool:
    """
    Embed documents and store in Chroma vector DB.
    
    Args:
        chunks: List of Document objects to embed
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _vector_db_instance, _embeddings_instance

    if not chunks:
        logger.error("❌ No chunks provided to embed_and_store()")
        return False

    try:
        # Drop Python references so GC can close old SQLite handles
        _vector_db_instance = None
        _embeddings_instance = None
        import gc; gc.collect()

        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        # Delete stale collection via ChromaDB API (keeps the SQLite file alive
        # — avoids SQLITE_READONLY_DBMOVED when a prior client still has it open)
        import chromadb
        _tmp_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            _tmp_client.delete_collection(COLLECTION_NAME)
            logger.info("🗑️  Deleted stale collection")
        except Exception:
            pass
        del _tmp_client; gc.collect()

        embeddings = get_embeddings()
        logger.info(f"Embedding {len(chunks)} chunks...")

        # Create Chroma DB with persistent storage
        _vector_db_instance = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY
        )
        
        # Verify storage
        count = _vector_db_instance._collection.count()
        logger.info(f"✅ Embeddings stored successfully. Total vectors: {count}")

        if count == 0:
            logger.error("❌ No vectors were stored!")
            return False

        # Persist BM25 corpus alongside the vector DB
        try:
            from rag.bm25_retriever import build_bm25_index
            build_bm25_index(chunks)
        except Exception as e:
            logger.warning("BM25 index build failed (non-fatal): %s", e)

        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding failed: {type(e).__name__}: {e}")
        return False


def load_vector_db() -> Optional[Chroma]:
    """
    Load vector database for querying.
    Returns cached instance if available.
    
    Returns:
        Chroma instance or None if DB doesn't exist
    """
    global _vector_db_instance
    
    # Return cached instance if available
    if _vector_db_instance is not None:
        return _vector_db_instance
    
    # Check if DB exists on disk
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.warning(f"⚠️  Vector DB not found at {PERSIST_DIRECTORY}")
        return None
    
    try:
        embeddings = get_embeddings()
        _vector_db_instance = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        return _vector_db_instance
        
    except Exception as e:
        logger.error(f"❌ Failed to load vector DB: {e}")
        return None


def get_vector_count() -> int:
    """
    Get count of vectors in the database.
    
    Returns:
        int: Number of vectors, or 0 if DB doesn't exist
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        return 0
    
    try:
        db = load_vector_db()
        if db is None:
            return 0
        count = db._collection.count()
        logger.info(f"📊 Vector DB Status: {count} vectors found")
        return count
    except Exception as e:
        logger.warning(f"⚠️  Could not count vectors: {e}")
        return 0


def retrieve_docs(query: str, top_k: int = 5) -> List[Document]:
    """
    Retrieve documents similar to the query.
    
    Args:
        query: Search query string
        top_k: Number of documents to return
        
    Returns:
        List of Document objects
    """
    db = load_vector_db()
    
    if db is None:
        logger.warning("⚠️  Vector DB not loaded. Returning empty results.")
        return []
    
    try:
        results = db.similarity_search(query, k=top_k)
        logger.info(f"🔍 Retrieved {len(results)} documents")
        return results
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}")
        return []


def clear_database():
    """Completely wipe the vector database. Use with caution."""
    logger.warning("⚠️  CLEARING DATABASE...")
    ensure_clean_db_directory()
    logger.info("✅ Database cleared.")
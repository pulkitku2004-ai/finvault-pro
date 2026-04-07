"""
Unified Vector Store Module - FinVault AI
Uses Chroma + OllamaEmbeddings for consistent embedding/retrieval
"""

import os
import shutil
import time
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PERSIST_DIRECTORY = "./vector_db"
COLLECTION_NAME = "finvault_docs"
EMBEDDING_MODEL = "mxbai-embed-large"

# Global instance (persist during session)
_vector_db_instance = None
_embeddings_instance = None


def get_embeddings():
    """Get or create embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings_instance


def ensure_clean_db_directory():
    """
    Remove vector DB directory completely.
    Critical after model changes or persistent errors.
    """
    if os.path.exists(PERSIST_DIRECTORY):
        logger.info(f"🧹 Removing {PERSIST_DIRECTORY}...")
        try:
            shutil.rmtree(PERSIST_DIRECTORY, ignore_errors=True)
            time.sleep(1)  # Let filesystem settle
        except Exception as e:
            logger.warning(f"⚠️  Could not remove directory: {e}")
    
    # Ensure clean directory
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
    global _vector_db_instance
    
    if not chunks:
        logger.error("❌ No chunks provided to embed_and_store()")
        return False
    
    try:
        # Clean start
        ensure_clean_db_directory()
        
        embeddings = get_embeddings()
        logger.info(f"Embedding {len(chunks)} chunks using {EMBEDDING_MODEL}...")
        
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
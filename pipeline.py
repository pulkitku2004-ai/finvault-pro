"""
FinVault AI - Enhanced Main Pipeline
Orchestrates document ingestion, retrieval, and answer generation
With improved debugging, confidence scoring, and test modes
"""

import os
import logging
import sys
from typing import Optional
from ingestion.pdf_parser import parse_pdf
from rag.chunking import chunk_documents
from rag.vector_store import embed_and_store, get_vector_count, load_vector_db
from rag.generator import generate_answer, generate_answer_with_reasoning
from agent_router import route_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VECTOR_DB_PATH = "./vector_db"
PDF_PATH = "data/hdfc_q3.pdf"

# Configuration
DEFAULT_TOP_K = 10
ENABLE_CONFIDENCE_SCORING = True
ENABLE_REASONING = False  # Set to True for detailed explanations


def run_ingestion(pdf_path: str = PDF_PATH) -> bool:
    """
    Run document ingestion pipeline.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        bool: True if successful
    """
    
    print("\n📥 Vector DB is empty. Building database...\n")
    
    # Step 1: Load PDF
    if not os.path.exists(pdf_path):
        logger.error(f"❌ PDF not found: {pdf_path}")
        return False
    
    logger.info(f"Loading PDF from {pdf_path}...")
    docs = parse_pdf(pdf_path)
    logger.info(f"✅ Loaded {len(docs)} documents from PDF")
    
    # Step 2: Chunk documents
    chunks = chunk_documents(docs)
    logger.info(f"✅ Created {len(chunks)} chunks")
    
    if not chunks:
        logger.error("❌ ERROR: No chunks created! Check PDF parsing.")
        return False
    
    # Show sample
    logger.info(f"\nSample chunk (first 300 chars):")
    print(f"  {chunks[0].page_content[:300]}...\n")
    
    # Step 3: Embed and store
    success = embed_and_store(chunks)
    
    if not success:
        logger.error("❌ ERROR: Embedding failed.")
        return False
    
    # Step 4: Verify
    vector_count = get_vector_count()
    if vector_count == 0:
        logger.error("❌ ERROR: Ingestion failed. Vector DB is still empty.")
        return False
    
    logger.info(f"\n✅ Ingestion complete. {vector_count} vectors ready.\n")
    return True


def run_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    use_confidence: bool = ENABLE_CONFIDENCE_SCORING,
    use_reasoning: bool = ENABLE_REASONING
) -> Optional[dict]:
    """
    Run query answering pipeline.
    
    Args:
        query: User's question
        top_k: Number of documents to retrieve
        use_confidence: Whether to score answer confidence
        use_reasoning: Whether to include step-by-step reasoning
        
    Returns:
        dict: Result with answer, sources, and metadata
    """
    
    print("\n" + "="*60)
    print(f"Query: {query}")
    print("="*60)
    
    # Route and retrieve
    results = route_query(query, top_k=top_k)
    
    logger.info(f"Retrieved {len(results) if results else 0} documents")
    
    if not results:
        print("\n⚠️  No documents found for this query.")
        return {
            "query": query,
            "answer": "Not found in provided documents.",
            "sources": [],
            "confidence": "low",
            "num_docs_retrieved": 0
        }
    
    # Generate answer with selected mode
    print("\n--- Generating Answer ---")
    
    if use_reasoning:
        result = generate_answer_with_reasoning(query, results)
    
    else:
        answer = generate_answer(query, results)
        result = {
            "query": query,
            "answer": answer,
            "sources": [f"Doc {i+1}" for i in range(len(results))],
            "num_docs_retrieved": len(results)
        }
    
    # Ensure query is in result
    if "query" not in result:
        result["query"] = query
    
    return result


def display_result(result: dict) -> None:
    """
    Display query result in formatted way.
    
    Args:
        result: Result dictionary from run_query
    """
    
    print("\n" + "="*60)
    print("FINVAULT AI ANSWER")
    print("="*60)
    
    print(f"\n{result.get('answer', 'No answer')}\n")
    
    # Display confidence if available
    if result.get('confidence'):
        confidence = result['confidence'].upper()
        emoji = "🟢" if confidence == "HIGH" else "🟡" if confidence == "MEDIUM" else "🔴"
        print(f"{emoji} Confidence: {confidence}")
    
    # Display sources
    sources = result.get('sources', [])
    if sources:
        print("\n" + "="*60)
        print("SOURCES USED")
        print("="*60)
        
        retrieved_docs = load_vector_db().similarity_search(result.get('query', ''), k=len(sources))
        for i, doc in enumerate(retrieved_docs[:len(sources)], 1):
            print(f"\n[Source {i}]")
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            print(content[:300])
            print("\n" + "-"*60)
    
    # Display reasoning if available
    if result.get('key_facts'):
        print("\n" + "="*60)
        print("KEY FACTS")
        print("="*60)
        for fact in result.get('key_facts', []):
            print(f"  • {fact}")
    
    if result.get('reasoning'):
        print("\n" + "="*60)
        print("REASONING")
        print("="*60)
        print(f"\n{result['reasoning']}\n")


def main(
    query: Optional[str] = None,
    mode: str = "standard",
    force_rebuild: bool = False
) -> None:
    """
    Main pipeline orchestration.
    
    Args:
        query: Optional query to run (if None, uses default)
        mode: "standard" (default), "confidence", or "reasoning"
        force_rebuild: Force rebuild of vector DB
    """
    
    print("\n" + "="*60)
    print("FinVault AI - Financial Research Assistant")
    print("="*60)
    
    # Configuration based on mode
    mode = mode.lower()
    use_confidence = mode in ["confidence", "full"]
    use_reasoning = mode == "reasoning"
    
    if mode == "reasoning":
        use_confidence = True  # Reasoning includes confidence
    
    logger.info(f"Mode: {mode} (confidence={use_confidence}, reasoning={use_reasoning})")
    
    # ==========================================
    # PHASE 1: CHECK VECTOR DB STATUS
    # ==========================================
    
    logger.info("--- VECTOR DB DEBUG ---")
    vector_count = get_vector_count()
    
    # ==========================================
    # PHASE 2: INGEST IF NEEDED
    # ==========================================
    
    if vector_count == 0 or force_rebuild:
        if force_rebuild and vector_count > 0:
            logger.warning("Force rebuild requested. Clearing DB...")
            from rag.vector_store import clear_database
            clear_database()
        
        success = run_ingestion(PDF_PATH)
        if not success:
            logger.error("Ingestion failed. Exiting.")
            sys.exit(1)
    else:
        logger.info(f"✅ Vector DB found with {vector_count} vectors. Skipping ingestion.\n")
    
    # ==========================================
    # PHASE 3: QUERY AND ANSWER
    # ==========================================
    
    if query is None:
        query = "What risks did HDFC mention in Q3 earnings?"
    
    result = run_query(
        query,
        top_k=DEFAULT_TOP_K,
        use_confidence=use_confidence,
        use_reasoning=use_reasoning
    )
    
    if result:
        display_result(result)
    
    # ==========================================
    # PHASE 4: SUMMARY
    # ==========================================
    
    print("\n" + "="*60)
    print("✅ Pipeline complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FinVault AI - Financial Research Assistant")
    parser.add_argument("--query", type=str, default=None, help="Question to ask")
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "confidence", "reasoning"],
        help="Answer generation mode"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of vector database"
    )
    
    args = parser.parse_args()
    
    main(
        query=args.query,
        mode=args.mode,
        force_rebuild=args.rebuild
    )
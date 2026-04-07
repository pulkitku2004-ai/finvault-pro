"""
Query Router - FinVault AI
Routes queries to appropriate retrieval strategy
Enhanced with better context and higher top_k
"""

import logging
#from langchain_ollama import ChatOllama
from rag.retriever import retrieve_docs
from rag.query_rewriter import rewrite_query, is_risk_query, is_financial_query
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph_retriever import validate_connection
from tools import graph_search_tool
import os
#from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# 1. Load your .env file
load_dotenv()

# 2. Configure the Groq Judge (Llama 3.3 70B is elite for financial RAG)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Aligned Groq Model ID
    api_key=os.getenv("GROQ_API_KEY"), # Ensure this is in your .env
    temperature=0, # Set to 0 for financial accuracy (no "creative" numbers)
    max_tokens=1024, # Increased slightly to allow for detailed risk explanations
    n=1,
    model_kwargs={
        "top_p": 1,
        #stop
    }
)

def route_question(question: str) -> str:
    """
    Uses LLM to classify the intent of the question.
    
    Args:
        question: The user's question
        
    Returns:
        str: Tool choice - "graph_search", "vector_search", or "calculator"
    """
    prompt = f"""You are a strict routing system for a financial AI.

TOOLS:

graph_search
Use ONLY for questions about:
- people
- executives
- speakers
- roles
- leadership
- relationships between people

Examples:
Who is the CEO?
Who spoke during the earnings call?
Which executives were present?

vector_search
Use for ALL questions about:
- financial results
- risks
- metrics
- ratios
- strategy
- commentary
- explanations
- anything inside documents

calculator
Use ONLY for arithmetic calculations.

Question: {question}

Respond ONLY with ONE of these exact words:
graph_search
vector_search
calculator
"""
    
    try:
        response = llm.invoke(prompt)
        tool = response.content.strip().lower()
        logger.info(f"📍 Router selected: {tool}")
        return tool
    except Exception as e:
        logger.error(f"❌ Routing failed: {e}")
        return "vector_search"  # Default fallback


def route_query(query: str, top_k: int = 10) -> list:
    """
    Main routing function - takes user query and routes to appropriate tool.
    
    Args:
        query: User's question
        top_k: Number of documents to retrieve (default: 10)
        
    Returns:
        List of relevant documents or tool response
    """
    
    # 1. Rewrite query for better retrieval
    search_query = rewrite_query(query)
    
    print("\n--- Query Transformation ---")
    print(f"Original: {query}")
    print(f"Rewritten: {search_query}")
    
    # 2. Decide which tool to use
    selected_tool = route_question(search_query)
    
    # 3. Route to appropriate handler
    if "graph_search" in selected_tool:
        print("Selected Tool: graph_search")

        # Fix (Issue 1): graph_retriever.py is fully implemented and connected.
        # Previously this block had a TODO and always fell back to vector search,
        # meaning the knowledge graph was never used through the main pipeline.
        # Now it calls graph_search_tool() which queries Neo4j directly, then
        # enriches with vector search so the answer has both graph + doc context.
        if validate_connection():
            logger.info("Routing to graph search (Neo4j connected)")
            graph_context = graph_search_tool(search_query)

            # Always supplement with vector search for richer context
            vector_docs   = retrieve_docs(search_query, top_k=top_k)
            vector_texts  = [
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in vector_docs
            ]

            # Return graph context as first result, vector docs appended after
            combined = [graph_context] + vector_texts if graph_context != "No graph information available for this query." else vector_texts
            logger.info("Graph + vector context combined (%d docs)", len(combined))
            return combined
        else:
            logger.warning("Neo4j offline — falling back to vector search")
            return retrieve_docs(search_query, top_k=top_k)
    
    elif "calculator" in selected_tool:
        print("Selected Tool: calculator")
        logger.warning("⚠️  Calculator not implemented.")
        return []
    
    else:
        # Default: vector search
        print("Selected Tool: vector_search")
        logger.info("🔎 Running Vector Search...")
        
        # Determine optimal top_k based on query type
        if is_risk_query(query):
            optimal_k = max(top_k, 15)  # More docs for risk queries
            logger.info(f"Risk query detected. Increasing top_k to {optimal_k}")
        elif is_financial_query(query):
            optimal_k = top_k
        else:
            optimal_k = top_k
        
        results = retrieve_docs(search_query, top_k=optimal_k)
        
        if not results:
            logger.warning("⚠️  No documents found for this query")
        else:
            logger.info(f"✅ Found {len(results)} relevant documents")
        
        return results
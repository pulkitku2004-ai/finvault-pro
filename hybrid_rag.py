
import logging
from agent_executor import agent_query
from llm_config import llm

logger = logging.getLogger(__name__)

# Try to import graph tools
try:
    from tools import graph_search_tool
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
def hybrid_query(question, period="Q3 FY2025"):
    """
    Hybrid query combining agent_executor with optional graph context.
    Uses your existing agent_executor for vector/graph/calculator routing.
    """
    
    try:
        # Get answer from your agent_executor (handles routing automatically)
        answer = agent_query(question)
        
        # Optionally enhance with explicit graph context if available
        if GRAPH_AVAILABLE:
            try:
                graph_context = graph_search_tool(question)
                
                if graph_context and graph_context != "No context":
                    # Enhance answer with graph context
                    enhancement_prompt = f"""
The following is an answer to a question:

INITIAL ANSWER:
{answer}

ADDITIONAL CONTEXT FROM GRAPH:
{graph_context}

Combine them into a single coherent answer that leverages both sources:
"""
                    response = llm.invoke(enhancement_prompt)
                    return response.content if response else answer
            except Exception as e:
                logger.warning("Graph enhancement failed: %s", e)
        
        return answer
    
    except Exception as e:
        print(f"Hybrid query error: {e}")
        return "Unable to process query"

if __name__ == "__main__":
    question = "What risks did HDFC mention in Q3 earnings?"
    answer = hybrid_query(question)
    print(f"\nQ: {question}")
    print(f"\nA: {answer}\n")
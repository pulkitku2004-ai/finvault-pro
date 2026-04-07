"""
Hybrid RAG - FinVault AI
Combines vector + graph retrieval using your agent_executor
"""

from agent_executor import agent_query
#from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
#from langchain_openai import ChatOpenAI

# Try to import graph tools
try:
    from tools import graph_search_tool
    GRAPH_AVAILABLE = True
except:
    GRAPH_AVAILABLE = False

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
            except:
                pass  # Return original answer if graph enhancement fails
        
        return answer
    
    except Exception as e:
        print(f"Hybrid query error: {e}")
        return "Unable to process query"

if __name__ == "__main__":
    question = "What risks did HDFC mention in Q3 earnings?"
    answer = hybrid_query(question)
    print(f"\nQ: {question}")
    print(f"\nA: {answer}\n")
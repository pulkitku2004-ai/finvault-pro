"""
Query Rewriter - FinVault AI
Transforms user queries into better search terms for financial documents
"""

import logging
#from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_groq import ChatGroq
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

def simple_rewrite(query: str) -> str:
    """
    Simple pattern-based query rewriting for financial terms.
    Fast and doesn't require LLM calls.
    
    Args:
        query: Original user question
        
    Returns:
        str: Enhanced search query
    """
    query_lower = query.lower()
    original_query = query
    
    # Risk-related expansions
    if "risk" in query_lower:
        query = query.replace("risk", "risk factors operational regulatory market credit liquidity")
    
    # Financial metrics expansions
    if "metric" in query_lower:
        query = query.replace("metric", "financial results key metrics KPIs performance")
    
    if "ratio" in query_lower:
        query = query.replace("ratio", "ROA ROE NIM CAR ratios")
    
    # Strategy/outlook expansions
    if "strategy" in query_lower or "outlook" in query_lower:
        query = query.replace("strategy", "strategic direction growth initiatives plans")
        query = query.replace("outlook", "guidance future prospects expectations")
    
    # Deposit/funding expansions
    if "deposit" in query_lower or "funding" in query_lower:
        query = query.replace("deposit", "deposits CASA funding sources")
        query = query.replace("funding", "borrowing sources capital")
    
    # Asset quality expansions
    if "npa" in query_lower or "credit quality" in query_lower:
        query = query.replace("npa", "NPA gross net non-performing asset quality")
        query = query.replace("credit quality", "asset quality NPA provision coverage")
    
    # Profitability expansions
    if "profit" in query_lower or "earnings" in query_lower:
        query = query.replace("profit", "profit PAT net income profitability")
        query = query.replace("earnings", "earnings EPS net profit PAT")
    
    # Regulatory/compliance
    if "regulatory" in query_lower or "compliance" in query_lower:
        query = query.replace("regulatory", "regulatory requirements Basel compliance capital")
        query = query.replace("compliance", "compliance regulation regulatory requirements")
    
    if original_query != query:
        logger.info(f"Simple rewrite: '{original_query}' → '{query}'")
    
    return query


def llm_rewrite(query: str) -> str:
    """
    LLM-based query rewriting for complex financial questions.
    More powerful but slower (requires LLM call).
    
    Args:
        query: Original user question
        
    Returns:
        str: Enhanced search query
    """
    prompt = f"""You are a financial query optimization system.

Transform this user question into better search terms for a financial document database.

RULES:
1. Expand acronyms and terminology
2. Add related financial concepts
3. Keep it concise (under 20 words)
4. Preserve original intent

Examples:
- "What are risks?" → "risk factors operational regulatory market credit liquidity risks"
- "Tell me about deposits" → "deposits funding sources CASA customer deposits"
- "Net margins" → "net interest margin NIM profitability"

USER QUESTION: {query}

REWRITTEN QUERY (just the query, no explanation):
"""
    
    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        
        if rewritten != query:
            logger.info(f"LLM rewrite: '{query}' → '{rewritten}'")
        
        return rewritten
    except Exception as e:
        logger.warning(f"LLM rewrite failed: {e}. Using original query.")
        return query


def rewrite_query(query: str, use_llm: bool = False) -> str:
    """
    Main query rewriting function.
    
    Args:
        query: Original user question
        use_llm: If True, use LLM-based rewriting (slower but smarter)
                 If False, use pattern-based rewriting (fast)
        
    Returns:
        str: Optimized search query
    """
    if use_llm:
        return llm_rewrite(query)
    else:
        return simple_rewrite(query)


def get_risk_keywords() -> list:
    """Get list of risk-related keywords for targeted search."""
    return [
        "risk", "risk factors", "regulatory risk", "operational risk",
        "market risk", "credit risk", "liquidity risk", "concentration risk",
        "risks", "challenges", "threats", "vulnerabilities"
    ]


def get_financial_keywords() -> list:
    """Get list of financial metric keywords."""
    return [
        "revenue", "profit", "earnings", "EPS", "ROA", "ROE", "NIM",
        "deposits", "advances", "CASA", "NPA", "CAR", "capital",
        "growth", "margin", "yield", "spread"
    ]


def is_risk_query(query: str) -> bool:
    """Check if query is about risks."""
    risk_keywords = get_risk_keywords()
    return any(keyword.lower() in query.lower() for keyword in risk_keywords)


def is_financial_query(query: str) -> bool:
    """Check if query is about financial metrics."""
    financial_keywords = get_financial_keywords()
    return any(keyword.lower() in query.lower() for keyword in financial_keywords)
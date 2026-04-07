from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from agent_router import route_query
from rag.generator import generate_answer
from utils.logger import log_query
from reranker import rerank_results

app = FastAPI(title="FinVault AI")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "FinVault AI Financial Research API"}

@app.post("/query")
def query_system(request: QueryRequest):
    query = request.query
    
    # 1. Route to retriever and get raw context data
    context_data = route_query(query)

    # 2. NORMALIZE: Convert objects/strings into a clean list of text
    # We define 'cleaned_context' HERE before we ever use it.
    cleaned_context = []
    if isinstance(context_data, list):
        for doc in context_data:
            if hasattr(doc, "page_content"):
                cleaned_context.append(doc.page_content)
            else:
                cleaned_context.append(str(doc))
    else:
        cleaned_context = [str(context_data)] if context_data else []

    # 3. RERANK: Now that cleaned_context is defined, we can process it
    if len(cleaned_context) > 1:
        # We use top_k=6 for better RAGAS results as discussed
        cleaned_context = rerank_results(query, cleaned_context, top_k=6)

    doc_count = len(cleaned_context)
    
    # 4. Generate and Verify answer
    answer = generate_answer(query, cleaned_context)  

    # 5. Log and Final Cleanup
    log_query(query, answer)
    clean_answer = answer.replace("Corrected answer:", "").replace("Verified answer:", "").strip()
    
    print(f"Query: {query} | Docs found: {doc_count}")
    
    return {
        "query": query,
        "answer": clean_answer,
        "documents_used": doc_count,
        "sources": cleaned_context[:3] if isinstance(cleaned_context, list) else [],
        "status": "success"
    }
from fastapi import FastAPI
from pydantic import BaseModel

from agent_router import route_query_with_scores
from rag.generator import generate_answer_for_audit
from utils.logger import log_query
from reranker import rerank_results
from astr_o_adapter import build_finvault_span
from astr_o_init import get_pipeline, format_response

app = FastAPI(title="FinVault AI")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "FinVault AI Financial Research API"}


@app.post("/query")
def query_system(request: QueryRequest):
    query = request.query

    # 1. Route and retrieve with similarity scores (needed by ASTR-O)
    docs, scored_pairs = route_query_with_scores(query)

    # 2. Normalise to plain strings
    cleaned_context = [
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in docs
    ]

    # 3. Rerank for generation quality
    if len(cleaned_context) > 1:
        cleaned_context = rerank_results(query, cleaned_context, top_k=6)

    doc_count = len(cleaned_context)

    # 4. Generate answer + token logprobs (ASTR-O needs logprobs)
    answer, logprobs = generate_answer_for_audit(query, cleaned_context)

    # 5. Build ASTR-O span and run the audit pipeline
    span = build_finvault_span(
        query=query,
        scored_pairs=scored_pairs,
        answer=answer,
        logprobs=logprobs,
    )
    astr_o_result = get_pipeline().process_span(span)

    # 6. Format the trust-surface response
    formatted = format_response(answer, astr_o_result)

    # 7. Log and return
    log_query(query, answer)

    return {
        **formatted,
        "query": query,
        "documents_used": doc_count,
    }

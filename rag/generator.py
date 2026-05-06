
import logging
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from llm_config import llm

_llm_logprobs = llm.bind(logprobs=True)

logger = logging.getLogger(__name__)
def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    Improved two-stage generation for maximum faithfulness.
    1. Extract specific facts.
    2. Synthesize factual answer.
    """
    if not context_docs:
        return "Not found in provided documents."

    formatted_docs = []
    for i, doc in enumerate(context_docs, 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        formatted_docs.append(f"[Doc {i}]\n{content}")
    context = "\n\n---\n\n".join(formatted_docs)

    # STAGE 1: Extract ONLY specific facts (Prevents local LLM "drifting")
    extraction_prompt = f"""You are a Fact Extractor. 
Read the DOCUMENTS and list ONLY specific facts, numbers, and dates related to: "{query}"

DOCUMENTS:
{context}

EXTRACTED FACTS (Bullet points only):"""

    # STAGE 2: Grounded synthesis
    try:
        facts_res = llm.invoke(extraction_prompt)
        facts = facts_res.content.strip()

        final_prompt = f"""You are FinVault AI. Answer the USER QUESTION using ONLY the provided FACTS.

FACTS:
{facts}

USER QUESTION:
{query}

STRICT RULES:
1. Every claim MUST cite the fact source (e.g. [Doc 1])
2. If facts don't answer the question, say "Information not available"
3. Do NOT use external knowledge or internal memory
4. Keep the answer concise and strictly factual

ANSWER:"""
        
        response = llm.invoke(final_prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"❌ Improved generation failed: {e}")
        return "Error generating factual answer."

def generate_answer_with_reasoning(query: str, context_docs: List[Document]) -> Dict[str, any]:
    """
    Forces the LLM to 'Show its work'. 
    This is proven to increase RAGAS Faithfulness scores on smaller models.
    """
    if not context_docs:
        return {"answer": "No documents found.", "reasoning": "None", "confidence": "low"}

    formatted_docs = [f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(context_docs)]
    context = "\n\n".join(formatted_docs)

    reasoning_prompt = f"""Analyze the provided documents to answer: {query}

DOCUMENTS:
{context}

STEP-BY-STEP ANALYSIS:
1. Identify specific numerical data/metrics in context.
2. Identify specific statements by executives or dates.
3. Compare these to the user question.
4. Synthesize final answer.

FINAL ANSWER FORMAT:
REASONING: [Your internal analysis]
ANSWER: [Direct answer with citations]
"""

    try:
        response = llm.invoke(reasoning_prompt)
        raw = response.content.strip()

        # Parse REASONING and ANSWER blocks from the response
        reasoning = ""
        answer    = raw  # fallback: return full response if parsing fails

        if "REASONING:" in raw and "ANSWER:" in raw:
            parts     = raw.split("ANSWER:", 1)
            answer    = parts[1].strip()
            reasoning = parts[0].replace("REASONING:", "").strip()
        elif "ANSWER:" in raw:
            answer = raw.split("ANSWER:", 1)[1].strip()

        return {
            "answer":    answer,
            "reasoning": reasoning,
            "confidence": "high" if reasoning else "medium",
        }

    except Exception as e:
        logger.error(f"❌ Reasoning generation failed: {e}")
        return {
            "answer":    "Error generating answer.",
            "reasoning": "",
            "confidence": "low",
        }


def generate_answer_for_audit(
    query: str, context_docs: List[Document]
) -> Tuple[str, List[Dict]]:
    """
    Two-stage generation that also returns token logprobs from the final
    synthesis step for ASTR-O token confidence analysis.

    Returns:
        (answer_text, logprobs)  where logprobs = [{"token": str, "logprob": float}]
    """
    if not context_docs:
        return "Not found in provided documents.", []

    formatted_docs = []
    for i, doc in enumerate(context_docs, 1):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        formatted_docs.append(f"[Doc {i}]\n{content}")
    context = "\n\n---\n\n".join(formatted_docs)

    # Stage 1 — fact extraction (no logprobs needed here)
    extraction_prompt = f"""You are a Fact Extractor.
Read the DOCUMENTS and list ONLY specific facts, numbers, and dates related to: "{query}"

DOCUMENTS:
{context}

EXTRACTED FACTS (Bullet points only):"""

    try:
        facts_res = llm.invoke(extraction_prompt)
        facts = facts_res.content.strip()

        # Stage 2 — grounded synthesis WITH logprobs
        final_prompt = f"""You are FinVault AI. Answer the USER QUESTION using ONLY the provided FACTS.

FACTS:
{facts}

USER QUESTION:
{query}

STRICT RULES:
1. Every claim MUST cite the fact source (e.g. [Doc 1])
2. If facts don't answer the question, say "Information not available"
3. Do NOT use external knowledge or internal memory
4. Keep the answer concise and strictly factual

ANSWER:"""

        response = _llm_logprobs.invoke(final_prompt)
        answer = response.content.strip()

        # Extract logprobs from response metadata
        raw = response.response_metadata.get("logprobs", {}) or {}
        content_logprobs = raw.get("content", []) or []
        logprobs = [
            {"token": entry["token"], "logprob": entry["logprob"]}
            for entry in content_logprobs
            if "token" in entry and "logprob" in entry
        ]

        return answer, logprobs

    except Exception as e:
        logger.error(f"❌ Audit generation failed: {e}")
        return "Error generating factual answer.", []
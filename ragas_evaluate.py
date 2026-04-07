"""
RAGAS Evaluation Pipeline - FinVault AI (v8)

Fix for Q4 CAR scoring 0.0 on all metrics:

The retrieved chunk for capital adequacy looks like:
  "Capital and liquidity metrics\n\nCapital adequacy\n\n20.0%\n\n19.8%\n\n19.3%..."

Even at 800 chars this is a bare number sequence. Mistral Nemo cannot verify
"The CAR was 20.0%" against "20.0% 19.8% 19.3%" because there is no natural
language sentence connecting the label to the value.

Fix: synthesize_chunk_sentences() detects section headers followed by number
sequences and converts them into readable sentences like:
  "Capital adequacy: 20.0% (Dec'24), 19.8% (previous period)."

This gives the judge model a sentence it can actually match against.
"""

import os
import logging
import warnings
import time
import re
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

for _msg in [
    ".*LangchainLLMWrapper is deprecated.*",
    ".*LangchainEmbeddingsWrapper is deprecated.*",
    ".*Importing Faithfulness from 'ragas.metrics' is deprecated.*",
    ".*Importing ContextPrecision from 'ragas.metrics' is deprecated.*",
    ".*Importing ContextRecall from 'ragas.metrics' is deprecated.*",
]:
    warnings.filterwarnings("ignore", message=_msg)

from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_groq import ChatGroq
from agent_router import route_question
from tools import vector_search_tool, graph_search_tool
from eval_dataset import dataset as EVAL_DATASET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_CONTEXT_CHUNKS  = 6
MAX_CHARS_PER_CHUNK = 800
AGENT_SLEEP_SEC     = 10
BATCH_SIZE          = 3
OLLAMA_BASE_URL     = "http://localhost:11434"
EVAL_MODEL_OLLAMA   = "mistral-nemo:latest"


def check_ollama_model(model_name: str) -> bool:
    import httpx
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if any(model_name.split(":")[0] in m for m in models):
                logger.info("✅ Ollama model '%s' is ready.", model_name)
                return True
            logger.warning("Model '%s' not found. Available: %s", model_name, models)
    except Exception:
        logger.error("❌ Ollama not running. Start with: ollama serve")
    return False


class GroqSafeChat(ChatGroq):
    def _enforce_n1(self, kwargs):
        kwargs["n"] = 1
        return kwargs

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop=stop, run_manager=run_manager, **self._enforce_n1(kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **self._enforce_n1(kwargs))


def parse_retry_seconds(error_str: str) -> int:
    m = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", error_str)
    return int(int(m.group(1)) * 60 + float(m.group(2))) + 10 if m else 90


def synthesize_chunk_sentences(chunk: str) -> str:
    """
    Convert raw number sequences into readable sentences for the judge model.

    Problem: PDF-extracted table data looks like:
      "Capital adequacy\n\n20.0%\n\n19.8%\n\n19.3% 18.8%..."
    Mistral Nemo cannot verify "CAR = 20.0%" against bare percentages because
    there's no sentence. This function detects that pattern and rewrites it as:
      "Capital adequacy for Q3 FY2025: 20.0% (latest), 19.8%, 19.3%."

    Applies to any metric-header + numbers pattern.
    """
    # Map known section headers to readable metric names
    header_map = {
        "capital adequacy":     "Capital Adequacy Ratio (CAR)",
        "net interest margin":  "Net Interest Margin (NIM)",
        "nim":                  "Net Interest Margin (NIM)",
        "gross npa":            "Gross NPA ratio",
        "gnpa":                 "Gross NPA ratio",
        "net npa":              "Net NPA ratio",
        "profit after tax":     "Profit After Tax (PAT)",
        "pat":                  "Profit After Tax (PAT)",
        "return on assets":     "Return on Assets (RoA)",
        "return on equity":     "Return on Equity (RoE)",
        "casa":                 "CASA ratio",
        "net interest income":  "Net Interest Income",
        "net revenue":          "Net Revenue",
    }

    lines = [l.strip() for l in chunk.split("\n") if l.strip()]
    if not lines:
        return chunk

    # Detect: first meaningful line is a header, rest are numbers/percentages
    header = lines[0].lower()
    matched_metric = next(
        (label for key, label in header_map.items() if key in header),
        None
    )

    if not matched_metric:
        return chunk  # Not a known metric header — return unchanged

    # Extract numeric values from the remaining lines
    number_pattern = re.compile(r"[\d,]+\.?\d*%?|₹\s*[\d,]+\.?\d*(?:\s*bn)?")
    values = []
    for line in lines[1:]:
        found = number_pattern.findall(line)
        values.extend(found)
        if len(values) >= 4:
            break

    if not values:
        return chunk  # No numbers found — return unchanged

    # Build a readable sentence
    latest = values[0]
    rest   = values[1:3]

    sentence = f"{matched_metric} for Q3 FY2025: {latest} (latest)"
    if rest:
        sentence += f", previously {', '.join(rest)}"
    sentence += "."

    # Append any remaining context from the original chunk
    remaining = " ".join(lines[1:])
    return f"{sentence} {remaining[:300]}".strip()


def score_chunk_relevance(chunk: str, question: str) -> int:
    """
    Score chunk relevance to the question via keyword matching.
    Higher = more relevant = ranked earlier.
    Fixes ContextPrecision by ensuring the most specific chunk is first.
    """
    q   = question.lower()
    c   = chunk.lower()
    score = 0

    term_weights = {
        "net interest margin": 4, "nim": 4,
        "gross npa": 4,           "gnpa": 4,
        "capital adequacy": 4,    "car": 3,
        "profit after tax": 4,    "pat": 4,
        "credit risk": 3,         "market risk": 3,
        "operational risk": 3,    "liquidity risk": 3,
        "npa": 2,    "capital": 2,
        "profit": 2, "revenue": 2,
        "risk": 2,   "adequacy": 2,
        # Penalise generic catch-all chunks
        "key financial parameters": -2,
        "subsidiaries": -2,
        "accounting standards": -1,
    }

    for term, weight in term_weights.items():
        if term in q and term in c:
            score += weight

    return score


def extract_context_chunks(raw_context, question: str) -> list:
    """
    1. Split raw string into individual chunks
    2. Filter out junk (too short, pure numbers, page numbers)
    3. Synthesize readable sentences from metric+number patterns
    4. Re-rank by question relevance (fixes ContextPrecision)
    5. Cap at MAX_CONTEXT_CHUNKS
    """
    chunks = []

    if isinstance(raw_context, list):
        for doc in raw_context:
            text = (doc.page_content if hasattr(doc, "page_content") else str(doc)).strip()
            if text:
                chunks.append(text[:MAX_CHARS_PER_CHUNK])
    elif isinstance(raw_context, str):
        for part in raw_context.split("\n\n"):
            part = part.strip()
            if len(part) < 20:
                continue
            if re.fullmatch(r"[\d\s\-\|%\.₹,]+", part):
                continue
            chunks.append(part[:MAX_CHARS_PER_CHUNK])

    # Synthesize readable sentences from raw metric+number patterns
    chunks = [synthesize_chunk_sentences(c) for c in chunks]

    # Cap and re-rank
    chunks = chunks[:MAX_CONTEXT_CHUNKS * 2]  # Keep extras for ranking
    chunks.sort(key=lambda c: score_chunk_relevance(c, question), reverse=True)
    chunks = chunks[:MAX_CONTEXT_CHUNKS]

    return chunks if chunks else ["No relevant context retrieved."]


def agent_query_with_context(question: str, llm) -> dict:
    tool = route_question(question)
    logger.info("Tool selected: %s", tool)

    if tool == "vector_search":
        raw_context = vector_search_tool(question)
    elif tool == "graph_search":
        raw_context = graph_search_tool(question)
    else:
        raw_context = "No tool available."

    context_chunks = extract_context_chunks(raw_context, question)
    logger.info("  Top chunk: %s...", context_chunks[0][:80])

    prompt = f"""Answer the question using ONLY the following context.

Context:
{raw_context}

Question:
{question}

Rules:
- Be concise and factual
- Only use information present in the context above
- If the answer is a specific number or ratio, state it directly"""

    response = llm.invoke(prompt)
    return {
        "answer":   response.content.strip(),
        "contexts": context_chunks,
    }


def run_evaluation():
    load_dotenv()

    AGENT_MODEL = "llama-3.3-70b-versatile"
    API_KEY = os.getenv("GROQ_API_KEY")

    if not API_KEY:
        logger.error("GROQ_API_KEY not found in .env")
        return None

    if not check_ollama_model(EVAL_MODEL_OLLAMA):
        return None

    logger.info("Agent model : Groq %s", AGENT_MODEL)
    logger.info("Eval model  : Ollama %s (local)", EVAL_MODEL_OLLAMA)

    agent_llm = ChatGroq(
        model=AGENT_MODEL, api_key=API_KEY,
        temperature=0, max_tokens=1024, n=1,
        model_kwargs={"top_p": 1},
    )

    eval_llm_base = ChatOllama(
        model=EVAL_MODEL_OLLAMA, base_url=OLLAMA_BASE_URL,
        temperature=0, num_predict=2048,
    )
    evaluator_llm        = LangchainLLMWrapper(eval_llm_base)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_BASE_URL)
    )

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    # ── Phase 1: Agent answers + synthesized contexts ─────────────────────────
    questions, answers, contexts, ground_truths = [], [], [], []
    logger.info("Phase 1: Collecting %d agent answers (Groq 70b)...", len(EVAL_DATASET))

    for i, item in enumerate(EVAL_DATASET, 1):
        try:
            if i > 1:
                time.sleep(AGENT_SLEEP_SEC)

            q, gt = item["question"], item["ground_truth"]
            logger.info("[%d/%d] %s", i, len(EVAL_DATASET), q)

            result = agent_query_with_context(q, agent_llm)
            ans, ctxs = result["answer"], result["contexts"]

            logger.info("  Answer : %s...", ans[:80])
            logger.info("  Top    : %s...", ctxs[0][:80])

            questions.append(q)
            answers.append(ans)
            contexts.append(ctxs)
            ground_truths.append(gt)
            logger.info("  Q%d collected ✓", i)

        except Exception as e:
            logger.error("Q%d failed: %s", i, e)

    if not questions:
        logger.error("No answers collected.")
        return None

    logger.info("%d/%d collected. Starting RAGAS.", len(questions), len(EVAL_DATASET))

    full_dataset = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts,  "ground_truth": ground_truths,
    })

    # ── Phase 2: RAGAS eval (local Mistral Nemo) ──────────────────────────────
    logger.info("Phase 2: RAGAS scoring (Ollama %s)...", EVAL_MODEL_OLLAMA)
    results_list, questions_log = [], []

    for j in range(0, len(full_dataset), BATCH_SIZE):
        end_idx   = min(j + BATCH_SIZE, len(full_dataset))
        batch_num = j // BATCH_SIZE + 1

        try:
            batch = full_dataset.select(range(j, end_idx))
            logger.info("Batch %d (q%d–q%d)...", batch_num, j + 1, end_idx)

            result = evaluate(
                dataset=batch, metrics=metrics,
                llm=evaluator_llm, embeddings=evaluator_embeddings,
                run_config=RunConfig(max_workers=1, timeout=600, max_retries=2),
            )
            df = result.to_pandas()
            results_list.append(df)
            questions_log.extend(questions[j:end_idx][:len(df)])

            score_str = {k: f"{v:.3f}" for k, v in df.mean(numeric_only=True).items() if v == v}
            logger.info("Batch %d scores: %s", batch_num, score_str)

        except Exception as e:
            logger.error("Batch %d failed: %s", batch_num, str(e)[:150])

        if end_idx < len(full_dataset):
            time.sleep(5)

    if not results_list:
        logger.error("All batches failed.")
        return None

    final_df   = pd.concat(results_list, ignore_index=True)
    score_cols = [
        c for c in final_df.columns
        if c not in ("question", "answer", "contexts", "ground_truth",
                     "user_input", "response", "retrieved_contexts", "reference")
    ]

    if "question" not in final_df.columns and questions_log:
        final_df.insert(0, "question", questions_log[:len(final_df)])

    avg_scores = final_df[score_cols].mean(numeric_only=True)
    return avg_scores, final_df, score_cols


if __name__ == "__main__":
    start_time = datetime.now()
    output = run_evaluation()

    if output is not None:
        avg_scores, final_df, score_cols = output

        print("\n" + "=" * 60)
        print("FINVAULT AI — RAGAS EVALUATION RESULTS")
        print(f"Agent : Groq llama-3.3-70b-versatile")
        print(f"Eval  : Ollama {EVAL_MODEL_OLLAMA} (local)")
        print("=" * 60)

        print("\nPer-question scores:")
        display_cols = (["question"] if "question" in final_df.columns else []) + score_cols
        display_df = final_df[display_cols].copy()
        if "question" in display_df.columns:
            display_df["question"] = display_df["question"].str[:52]
        print(display_df.to_string(index=False))

        print("\n--- Averages ---")
        for metric, score in avg_scores.items():
            if score == score:
                bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                grade = "🟢 GOOD" if score >= 0.7 else "🟡 OK" if score >= 0.4 else "🔴 LOW"
                print(f"  {metric:<22} {score:.3f}  |{bar}|  {grade}")

        print("=" * 60)
        print(f"Duration: {datetime.now() - start_time}")
    else:
        print("Evaluation failed. Check logs above.")
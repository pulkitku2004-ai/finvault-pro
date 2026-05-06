FinVault AI — Evaluation Pipeline
==================================

Two complementary evaluation tracks run against a shared 5-question dataset
(NIM, PAT, Gross NPA, Capital Adequacy Ratio, risk factors from HDFC Q3 FY2025).

──────────────────────────────────────────────────────────
Track 1 — Accuracy  (evaluate.py)
──────────────────────────────────────────────────────────

eval_dataset.py  (5 Q&A pairs with ground truths)
      │
      ▼
agent_query(question)           ← agent_executor.py
  └─ route_question → tool → LLM (gpt-4o)
      │
      ▼
LLM-as-Judge  (gpt-4o-mini)
  Prompt includes:
  - Indian unit conversion rules (crore ↔ bn ↔ billion)
  - FY calendar (Q3 FY2025 = Dec 2024)
  - 5% rounding tolerance
  - Extra-context allowance (correct + more = CORRECT)
      │
      ▼
Verdict per question: CORRECT | BORDERLINE | INCORRECT
      │
      ▼
Accuracy Score

  Result: 100% (5/5)  — all answers match ground truth

Fallback methods available:
  keyword   — regex number + keyword overlap (≥60% = pass)
  substring — exact ground-truth substring match

──────────────────────────────────────────────────────────
Track 2 — RAGAS  (ragas_evaluate.py)
──────────────────────────────────────────────────────────

eval_dataset.py  (same 5 questions)
      │
      ▼
Phase 1 — Collect agent answers
  For each question:
  └─ route_query_with_scores (hybrid BM25+RRF + cross-encoder reranker)
     └─ generate_answer_for_audit (two-stage gpt-4o, logprobs captured)
     Logs: selected tool, top retrieved chunk, final answer
      │
      ▼
Phase 2 — RAGAS scoring
  Judge LLM   : gpt-4o-mini
  Embeddings  : text-embedding-3-small
  Batched 3+2 to avoid rate limits

  Metrics computed per question, then averaged:

  ┌─────────────────────┬────────┬─────────────────────────────────────────┐
  │ Metric              │ Score  │ What it measures                        │
  ├─────────────────────┼────────┼─────────────────────────────────────────┤
  │ Faithfulness        │ 0.800  │ Claims in answer traceable to context   │
  │ ContextPrecision    │ 1.000  │ Most relevant chunk ranked first        │
  │ ContextRecall       │ 0.800  │ All ground-truth evidence retrieved     │
  └─────────────────────┴────────┴─────────────────────────────────────────┘

Per-question detail:

  Q1 NIM           faith=1.0  prec=1.0  recall=1.0   ✓
  Q2 PAT           faith=1.0  prec=1.0  recall=1.0   ✓
  Q3 Gross NPA     faith=0.0  prec=1.0  recall=1.0   △ (see note)
  Q4 CAR           faith=1.0  prec=1.0  recall=1.0   ✓
  Q5 Risks         faith=1.0  prec=1.0  recall=0.0   △ (see note)

Notes:
  Q3 faithfulness=0.0 — RAGAS judge penalised the answer "1.42%" as too
    short to find sentence-level attribution. The value is correct and
    directly present in the source chunk. Not a real hallucination.

  Q5 recall=0.0 — Risk-factors chunk is retrieved by hybrid search but
    the cross-encoder reranker pushes it below the top_k=6 cutoff used
    in generation. Fix: raise reranker top_k for is_risk_query() paths.

──────────────────────────────────────────────────────────
Retrieval improvements history
──────────────────────────────────────────────────────────

  Before BM25+RRF  →  ContextPrecision = 0.767
  After  BM25+RRF  →  ContextPrecision = 1.000

  Root cause of 0.767: dense retrieval ranked a capital adequacy chunk
  first for the risk-factors query because RecursiveCharacterTextSplitter
  split table rows, bleeding CAR content into adjacent chunks. BM25
  exact-keyword matching on "risk" / "risks" promoted the correct chunk
  to rank 1 in every question.

──────────────────────────────────────────────────────────
ASTR-O span-level audit (per query, not batch)
──────────────────────────────────────────────────────────

Every production query also passes through ASTROPipeline.process_span():

  Gate 1  Contradiction     — domain schema regex vs source chunks
  Gate 2  Token confidence  — logprob ≥ −0.25 on NIM/GNPA/PAT/CAR tokens
  Gate 3  Source auth       — hdfc_q3.pdf in registry, tier=CRITICAL match
  Gate 4  Groundedness      — answer words ∈ retrieved chunk text

  All 4 pass → SAFE → ✓ Verified badge
  Any fail   → FLAGGED → ⚠ Unverified + causal chain analysis

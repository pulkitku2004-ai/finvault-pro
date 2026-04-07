"""
Evaluation Script - FinVault AI (Fixed)

Changes from previous version:
  1. Judge prompt now explicitly handles unit conversions (crore vs bn vs billion)
     and Indian financial year calendar (Q3 FY2025 = December 2024).
     This fixes the false failures on Q2 (PAT units) and Q3 (NPA date).

  2. Judge prompt adds a "BORDERLINE" verdict for close calls, so you can
     see which answers are almost-correct vs clearly wrong.

  3. Failure summary now shows the full reason, not truncated at 70 chars.
"""

import os
import re
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from eval_dataset import dataset
from agent_executor import agent_query

load_dotenv()


def get_judge_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,
        max_tokens=300,
        n=1,
    )


def llm_judge_accuracy(question: str, ground_truth: str, answer: str, llm) -> tuple:
    """
    LLM-as-judge accuracy scoring.

    Key improvements over v1:
    - Explicitly handles Indian financial unit conversions (crore/bn/billion/lakh)
    - Knows Q3 FY2025 = Dec 2024, Q2 FY2025 = Sep 2024, etc.
    - Treats rounding within 5% as acceptable for financial figures
    - Returns CORRECT / BORDERLINE / INCORRECT with a clear reason
    """
    prompt = f"""You are a strict financial fact-checker for an Indian bank's earnings report.

QUESTION: {question}

GROUND TRUTH (the correct answer):
{ground_truth}

AI ANSWER (to be evaluated):
{answer}

IMPORTANT CONTEXT — apply these rules before judging:
1. Indian financial units are equivalent:
   - 1 crore = 10 million = 0.01 billion
   - 167 bn = 16,700 crore (approximately) → treat as SAME number
   - Always check if units differ before calling something wrong
2. Indian financial year calendar:
   - Q3 FY2025 = December 2024 (Oct–Dec 2024 quarter)
   - Q2 FY2025 = September 2024
   - These refer to the same period — do not penalise for using one vs the other
3. Rounding: financial figures rounded within 5% are acceptable
   - e.g. 3.42% vs 3.43% = CORRECT, 1.42% vs 1.4% = CORRECT
4. Extra context: if the AI gives correct facts plus additional explanation, that is still CORRECT

VERDICT OPTIONS:
- CORRECT: the AI answer contains the key fact from ground truth (allowing for units/rounding/rephrasing)
- BORDERLINE: partially correct — has some facts right but missing something important
- INCORRECT: clearly wrong fact, wrong number, or says "not available" when ground truth has an answer

Respond in EXACTLY this format (no extra text):
VERDICT: CORRECT
REASON: (one concise sentence)"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()

        text_upper = text.upper()
        if "VERDICT: CORRECT" in text_upper:
            is_correct = True
            label = "PASS"
        elif "VERDICT: BORDERLINE" in text_upper:
            is_correct = False  # Counts as fail for accuracy score
            label = "BORDERLINE"
        else:
            is_correct = False
            label = "FAIL"

        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "No reason given"

        return is_correct, label, reason

    except Exception as e:
        return False, "ERROR", f"Judge error: {str(e)[:80]}"


def keyword_accuracy(ground_truth: str, answer: str) -> tuple:
    """Fallback: extract numbers and key terms, check presence in answer."""
    gt_lower  = ground_truth.lower()
    ans_lower = answer.lower()

    numbers  = re.findall(r"\d+\.?\d*%?", ground_truth)
    stopwords = {"the", "and", "for", "this", "that", "with", "from", "bank",
                 "hdfc", "was", "were", "have", "been", "approximately"}
    words    = re.findall(r"\b[a-z]{4,}\b", gt_lower)
    keywords = [w for w in words if w not in stopwords][:5]

    matched_numbers  = [n for n in numbers  if n.lower() in ans_lower]
    matched_keywords = [k for k in keywords if k in ans_lower]
    total_checks  = len(numbers) + len(keywords)
    total_matched = len(matched_numbers) + len(matched_keywords)

    if total_checks == 0:
        is_correct = gt_lower in ans_lower
        return is_correct, "PASS" if is_correct else "FAIL", "Substring match"
    elif total_matched / total_checks >= 0.6:
        return True, "PASS", f"Found {total_matched}/{total_checks} key facts"
    else:
        missing = [n for n in numbers if n not in matched_numbers]
        return False, "FAIL", f"Only {total_matched}/{total_checks} facts found. Missing: {missing}"


def run_eval(method: str = "llm_judge"):
    """
    Run accuracy evaluation.

    Args:
        method: "llm_judge"  — LLM-as-judge (most accurate, uses Groq API)
                "keyword"    — Keyword/number extraction (fast, free)
                "substring"  — Original exact substring match
    """
    total   = len(dataset)
    correct = 0
    results = []

    print(f"\n{'='*60}")
    print(f"FinVault AI — Accuracy Evaluation")
    print(f"Method   : {method}")
    print(f"Questions: {total}")
    print(f"{'='*60}\n")

    judge_llm = None
    if method == "llm_judge":
        try:
            judge_llm = get_judge_llm()
            print("Judge LLM: Groq llama-3.3-70b-versatile\n")
        except Exception as e:
            print(f"Could not load judge LLM: {e}. Falling back to keyword.\n")
            method = "keyword"

    for i, item in enumerate(dataset, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"[{i}/{total}] {question[:55]}...")

        try:
            if i > 1:
                time.sleep(6)

            answer = agent_query(question)

            if method == "llm_judge" and judge_llm:
                time.sleep(3)
                is_correct, label, reason = llm_judge_accuracy(
                    question, ground_truth, answer, judge_llm
                )
            elif method == "keyword":
                is_correct, label, reason = keyword_accuracy(ground_truth, answer)
            else:
                is_correct = ground_truth.lower() in answer.lower()
                label  = "PASS" if is_correct else "FAIL"
                reason = "Exact substring match"

            if is_correct:
                correct += 1

            print(f"   {label:<12} {reason[:72]}")
            if not is_correct:
                print(f"   Expected : {ground_truth[:65]}")
                print(f"   Got      : {answer[:65]}")

            results.append({
                "question": question, "ground_truth": ground_truth,
                "answer": answer, "correct": is_correct,
                "label": label, "reason": reason,
            })

        except Exception as e:
            print(f"   ERROR        {str(e)[:60]}")
            results.append({
                "question": question, "ground_truth": ground_truth,
                "answer": "", "correct": False, "label": "ERROR",
                "reason": str(e),
            })

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    accuracy = (correct / total * 100) if total > 0 else 0
    borderlines = sum(1 for r in results if r.get("label") == "BORDERLINE")

    print(f"{'='*60}")
    print(f"ACCURACY  : {correct}/{total}  ({accuracy:.1f}%)")
    if borderlines:
        print(f"BORDERLINE: {borderlines} (close but not fully correct)")
    print(f"{'='*60}")

    grade = (
        "GOOD — system is reliable"         if accuracy >= 80 else
        "OK — some gaps in retrieval"        if accuracy >= 60 else
        "LOW — retrieval or grounding issue"
    )
    print(f"Grade  : {grade}")
    print(f"Method : {method}")
    print(f"{'='*60}\n")

    failures = [r for r in results if not r["correct"]]
    if failures:
        print("Failed / borderline questions:")
        for f in failures:
            print(f"  [{f['label']}] {f['question'][:55]}")
            print(f"         {f['reason']}")
        print()

    return accuracy, results


if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "llm_judge"
    run_eval(method=method)
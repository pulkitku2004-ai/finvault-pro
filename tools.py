import re
from vector_retreiver import retrieve_docs
from graph_retriever import (
    get_executives_for_period,
    get_company_risks,
    get_company_metrics,
    get_company_regulations,
)

# Quarter detection: maps any "Q1/Q2/Q3/Q4 FY20XX" pattern in the query
QUARTER_PATTERN = re.compile(r"Q[1-4]\s*FY\s*\d{2,4}", re.IGNORECASE)
DEFAULT_COMPANY  = "HDFC Bank"
DEFAULT_PERIOD   = "Q3 FY2025"


def vector_search_tool(query):
    docs = retrieve_docs(query)
    print("\nRetrieved Docs:")
    for d in docs:
        print("-", d[:120])
    return "\n".join(docs)


def graph_search_tool(query):
    """
    Routes graph queries to the correct Neo4j retrieval function.

    Fix 1 (Issue 5): Was hardcoded to 'Q3' only — any other period returned
    'No graph information available'. Now detects any quarter from the query
    text using a regex and falls back to DEFAULT_PERIOD if none found.

    Also queries risks, metrics, and regulations from the graph, not just
    executives — those functions existed in graph_retriever.py but were
    never called from here.
    """
    query_lower = query.lower()

    # Detect which period the question is about
    match = QUARTER_PATTERN.search(query)
    period = match.group(0).strip() if match else DEFAULT_PERIOD

    results = []

    # Executives / people questions
    if any(w in query_lower for w in ["who", "speaker", "executive", "ceo", "cfo",
                                       "spoke", "present", "management", "role"]):
        executives = get_executives_for_period(period)
        if executives:
            results.append(f"Executives who spoke in {period}: {', '.join(executives)}")

    # Risk questions
    if any(w in query_lower for w in ["risk", "risks", "exposure", "threat", "concern"]):
        risks = get_company_risks(DEFAULT_COMPANY)
        if risks:
            results.append(f"Risks associated with {DEFAULT_COMPANY}: {', '.join(risks)}")

    # Metric questions
    if any(w in query_lower for w in ["metric", "kpi", "ratio", "nim", "npa", "car",
                                       "return", "roe", "roa"]):
        metrics = get_company_metrics(DEFAULT_COMPANY)
        if metrics:
            results.append(f"Financial metrics for {DEFAULT_COMPANY}: {', '.join(metrics)}")

    # Regulation questions
    if any(w in query_lower for w in ["regulation", "rbi", "basel", "compliance",
                                       "guideline", "circular"]):
        regulations = get_company_regulations(DEFAULT_COMPANY)
        if regulations:
            results.append(f"Regulations affecting {DEFAULT_COMPANY}: {', '.join(regulations)}")

    if results:
        return "\n".join(results)

    return "No graph information available for this query."


def calculator_tool(expression: str) -> str:
    """
    Fix 2 (Issue 2): Replaced eval() with a safe math-only evaluator.

    eval() executes arbitrary Python — a security vulnerability if the
    expression comes from user input through the API. The replacement
    uses re.fullmatch to whitelist only digits, operators, spaces,
    parentheses, and decimal points before evaluating. Anything else
    is rejected before eval() is ever called.
    """
    # Whitelist: only allow digits, basic operators, spaces, parens, decimals
    safe_pattern = re.compile(r"^[\d\s\+\-\*/\(\)\.]+$")
    expression = expression.strip()

    if not safe_pattern.fullmatch(expression):
        return "Invalid calculation: only basic arithmetic is supported (+, -, *, /)"

    try:
        result = eval(expression)  # Safe: input validated above
        return f"Calculation result: {result}"
    except ZeroDivisionError:
        return "Invalid calculation: division by zero"
    except Exception:
        return "Invalid calculation"


# now finvault has 3 tools: vector_search_tool, graph_search_tool, calculator_tool
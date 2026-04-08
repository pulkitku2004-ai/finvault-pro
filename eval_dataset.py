
dataset = [

    # Vector: Exact Metric (NIM)
    {
        "question": "What is the Net Interest Margin for Q3 FY2025?",
        "ground_truth": "The Net Interest Margin (NIM) for Q3 FY2025 was 3.43%.",
        "contexts": [],
    },

    # Vector: Exact Metric (PAT) — unit matches what the PDF chart shows
    {
        "question": "What was the Profit After Tax for Q3 FY2025?",
        "ground_truth": "The Profit After Tax (PAT) for Q3 FY2025 was approximately Rs 167 bn on a standalone basis.",
        "contexts": [],
    },

    # Vector: Exact Ratio (NPA) — removed ambiguous date qualifier
    {
        "question": "What is the Gross NPA ratio of HDFC Bank for Q3 FY2025?",
        "ground_truth": "The Gross NPA ratio of HDFC Bank for Q3 FY2025 was approximately 1.42%.",
        "contexts": [],
    },

    # Vector: Exact Ratio (CAR)
    {
        "question": "What was the capital adequacy ratio reported in Q3 FY2025?",
        "ground_truth": "The Capital Adequacy Ratio (CAR) was 20.0% as reported in Q3 FY2025.",
        "contexts": [],
    },

    # Hybrid: Risk factors
    {
        "question": "What specific risks did HDFC Bank mention in the Q3 FY2025 earnings presentation?",
        "ground_truth": "HDFC Bank highlighted credit risk, market risk, regulatory and compliance risk, liquidity risk, and operational risk as key risk factors in Q3 FY2025.",
        "contexts": [],
    },

]
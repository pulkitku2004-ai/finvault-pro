FINVAULT_SCHEMA = {
    "net_interest_margin": {
        "regex": r"\b(\d+\.\d+)\s*%",
        "critical": True,
        "severity": "HIGH",
        "context_patterns": ["NIM", "net interest margin", "margin"],
    },
    "profit_after_tax": {
        "regex": r"₹\s*(\d+(?:\.\d+)?)\s*(?:bn|billion|crore|lakh)?",
        "critical": True,
        "severity": "HIGH",
        "context_patterns": ["PAT", "profit after tax", "profit"],
    },
    "gross_npa": {
        "regex": r"\b(\d+\.\d+)\s*%",
        "critical": True,
        "severity": "HIGH",
        "context_patterns": ["Gross NPA", "GNPA", "NPA ratio"],
    },
    "capital_adequacy_ratio": {
        "regex": r"\b(\d+\.\d+)\s*%",
        "critical": True,
        "severity": "HIGH",
        "context_patterns": ["CAR", "capital adequacy", "CRAR", "Basel"],
    },
    "eps": {
        "regex": r"₹\s*(\d+(?:\.\d+)?)",
        "critical": False,
        "severity": "MEDIUM",
        "context_patterns": ["EPS", "earnings per share"],
    },
    "return_on_assets": {
        "regex": r"\b(\d+\.\d+)\s*%",
        "critical": False,
        "severity": "MEDIUM",
        "context_patterns": ["RoA", "return on assets", "ROA"],
    },
}

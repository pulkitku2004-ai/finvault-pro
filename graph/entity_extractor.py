import spacy
nlp = spacy.load("en_core_web_sm")

RISK_KEYWORDS = [
    "credit risk",
    "liquidity risk",
    "market volatility",
    "regulatory risk",
    "asset quality"
]

METRIC_KEYWORDS = [
    "NPA",
    "ROE",
    "revenue",
    "profit",
    "loan growth",
    "deposit growth"
]

REGULATION_KEYWORDS = [
    "RBI",
    "Basel",
    "capital adequacy",
    "compliance"
]

def extract_entities(text):
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities.append(("Company", ent.text))
        if ent.label_ == "PERSON":
            entities.append(("Executive", ent.text))
    
    for word in RISK_KEYWORDS:
        if word.lower() in text.lower():
            entities.append(("Risk", word))

    for word in METRIC_KEYWORDS:
        if word.lower() in text.lower():
            entities.append(("FinancialMetric", word))

    for word in REGULATION_KEYWORDS:
        if word.lower() in text.lower():
            entities.append(("Regulation", word))

    if "Q1" in text or "Q2" in text or "Q3" in text or "Q4" in text:
        entities.append(("Quarter", "Q3 FY2025"))

    return entities
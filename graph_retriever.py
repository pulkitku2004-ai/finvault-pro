
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize driver
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Test connection
    with driver.session() as session:
        session.run("RETURN 1")
    CONNECTED = True
    logger.info("Neo4j connected")
except Exception as e:
    CONNECTED = False
    driver = None
    logger.warning("Neo4j offline: %s", str(e)[:50])


def safe_run(query_str, params):
    """Run query safely, return results or empty."""
    if not CONNECTED or driver is None:
        return []
    
    try:
        with driver.session() as session:
            result = session.run(query_str, params)
            return result.data()
    except Exception as e:
        logger.warning("Neo4j query failed: %s", e)
        return []


# ============================================================
# RETRIEVAL FUNCTIONS
# ============================================================

def get_executives_for_period(period):
    """Get executives who spoke in a period."""
    query = """
    MATCH (e:Executive)-[:SPOKE_IN]->(p:FIN_PERIOD {name:$period})
    RETURN e.name
    """
    records = safe_run(query, {"period": period})
    return [r.get("e.name") for r in records if r.get("e.name")]


def get_company_periods(company):
    """Get periods for a company."""
    query = """
    MATCH (c:Company {name:$company})-[:REPORTED_IN]->(p)
    RETURN p.name
    """
    records = safe_run(query, {"company": company})
    return [r.get("p.name") for r in records if r.get("p.name")]


def get_company_risks(company):
    """Get risks mentioned by company."""
    query = """
    MATCH (c:Company {name:$company})-[:MENTIONS_RISK]->(r:Risk)
    RETURN r.name
    """
    records = safe_run(query, {"company": company})
    return [r.get("r.name") for r in records if r.get("r.name")]


def get_company_metrics(company):
    """Get financial metrics for company."""
    query = """
    MATCH (c:Company {name:$company})-[:REPORTS_METRIC]->(m:FinancialMetric)
    RETURN m.name
    """
    records = safe_run(query, {"company": company})
    return [r.get("m.name") for r in records if r.get("m.name")]


def get_company_regulations(company):
    """Get regulations affecting company."""
    query = """
    MATCH (c:Company {name:$company})-[:AFFECTED_BY]->(r:Regulation)
    RETURN r.name
    """
    records = safe_run(query, {"company": company})
    return [r.get("r.name") for r in records if r.get("r.name")]


def validate_connection():
    """Check if Neo4j is connected."""
    return CONNECTED
import json
from datetime import datetime


def log_query(query, answer):

    log_entry = {
        "timestamp": str(datetime.now()),
        "query": query,
        "answer": answer
    }

    with open("query_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
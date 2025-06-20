import os
import json
from datetime import datetime

LOG_PATH = os.path.join("data", "rag_query_logs.json")

def log_query_entry(query, answer, sources, model_name, duration):
    os.makedirs("data", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model": model_name,
        "duration_sec": round(duration, 2),
        "answer_preview": answer[:200],
        "sources": sources,
    }
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r+", encoding="utf-8") as f:
                logs = json.load(f)
                logs.append(entry)
                f.seek(0)
                json.dump(logs, f, indent=2)
        else:
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump([entry], f, indent=2)
    except Exception as e:
        print(f"Logging failed: {e}")


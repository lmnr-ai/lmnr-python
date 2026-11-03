import json
import sqlite3
import sys
import time

sys.path.insert(0, "/srv/app")
from app.support_agent import DB_PATH, handle_ticket  # the production code path

case = json.load(open("/case/case.json"))

# Create this case's database from its seed data.
conn = sqlite3.connect(DB_PATH)
conn.executescript(
    """
    CREATE TABLE charges (
        id TEXT PRIMARY KEY, email TEXT, description TEXT, amount REAL, created_at TEXT
    );
    CREATE TABLE refunds (charge_id TEXT, amount REAL);
    """
)
for c in json.load(open("/case/seed.json"))["charges"]:
    conn.execute(
        "INSERT INTO charges VALUES (:id, :email, :description, :amount, :created_at)", c
    )
conn.commit()
conn.close()

started = time.time()
try:
    result = {"outcome": "completed", "reply": handle_ticket(case["ticket"])}
except Exception as e:
    result = {"outcome": "failed", "error": repr(e)}
result["duration_s"] = round(time.time() - started, 2)

json.dump(result, open("/logs/agent/result.json", "w"), indent=2)
print(json.dumps(result))

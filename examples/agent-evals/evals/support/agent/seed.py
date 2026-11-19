import json
import sqlite3
import sys

sys.path.insert(0, "/srv/app")
from app.support_agent import DB_PATH  # the database the production code reads

# The schema every case shares, then this case's rows (environment/seed.json).
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

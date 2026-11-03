import json
import sqlite3

from anthropic import Anthropic

expected = json.load(open("/tests/expected.json"))
result = json.load(open("/logs/agent/result.json"))

if result["outcome"] != "completed":
    print(f"agent run did not complete: {result}")
    json.dump({"refund": 0, "reply": 0}, open("/logs/verifier/reward.json", "w"))
    raise SystemExit(0)

reply = result["reply"]

# 1. Hard check: what actually happened in the database.
conn = sqlite3.connect("/srv/app/app.db")
count, total = conn.execute("SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM refunds").fetchone()
refund_ok = count == expected["refund_count"] and total == expected["refund_amount"]
print(f"refunds: {count} totalling {total}")

# 2. LLM judge: is the reply to the customer any good?
verdict = Anthropic().messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    # Structured output, so the verdict is always valid JSON.
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
                "required": ["pass", "reason"],
                "additionalProperties": False,
            },
        }
    },
    messages=[
        {
            "role": "user",
            "content": (
                f"A support agent wrote this reply to a customer:\n\n{reply}\n\n"
                f"The reply should: {expected['reply_should']}"
            ),
        }
    ],
)
judge = json.loads(verdict.content[0].text)
print(f"judge: {judge}")

json.dump(
    {"refund": int(refund_ok), "reply": int(judge["pass"])},
    open("/logs/verifier/reward.json", "w"),
)

import json
import sys
import time

sys.path.insert(0, "/srv/app")
from app.support_agent import handle_ticket  # the production code path

case = json.load(open("/case/case.json"))

started = time.time()
try:
    result = {"outcome": "completed", "reply": handle_ticket(case["ticket"])}
except Exception as e:
    result = {"outcome": "failed", "error": repr(e)}
result["duration_s"] = round(time.time() - started, 2)

json.dump(result, open("/logs/agent/result.json", "w"), indent=2)
print(json.dumps(result))

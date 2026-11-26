"""The support agent behind a small HTTP API, the way it runs in production.

POST /tickets {"ticket": ...} queues a ticket and answers 202 {"id": ...}; a
worker thread handles it and writes the outcome to <AUDIT_DIR>/<id>.json.
GET /health answers 200 once the server is up.
"""

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lmnr import Laminar

from app.support_agent import handle_ticket

AUDIT_DIR = os.environ.get("AUDIT_DIR", "/srv/app/audit")
PORT = int(os.environ.get("PORT", "8080"))


def work(ticket_id: str, ticket: str) -> None:
    try:
        record = {"status": "done", "reply": handle_ticket(ticket)}
    except Exception as e:
        record = {"status": "failed", "error": repr(e)}
    Laminar.flush()
    path = os.path.join(AUDIT_DIR, f"{ticket_id}.json")
    # Written whole, then renamed, so a reader never sees half a record.
    with open(path + ".tmp", "w") as f:
        json.dump({"id": ticket_id, "ticket": ticket, **record}, f)
    os.replace(path + ".tmp", path)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/tickets":
            self.send_json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ticket_id = uuid.uuid4().hex[:12]
        threading.Thread(target=work, args=(ticket_id, body["ticket"]), daemon=True).start()
        self.send_json(202, {"id": ticket_id})

    def send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    os.makedirs(AUDIT_DIR, exist_ok=True)
    print(f"listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

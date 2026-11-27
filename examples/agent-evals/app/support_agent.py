import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request

from anthropic import Anthropic
from lmnr import Laminar, observe

Laminar.initialize()
client = Anthropic()

DB_PATH = os.environ.get("APP_DB", "/srv/app/app.db")
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")
# Production services the agent talks to over HTTP.
CRM_API_URL = os.environ.get("CRM_API_URL", "https://crm.example.com/api")
HELPDESK_API_URL = os.environ.get("HELPDESK_API_URL", "https://helpdesk.example.com/api")

SYSTEM_PROMPT = """You are a support agent for a subscription software company.
Customers write in about billing problems. Use the tools to look up their charges
and issue refunds when a refund is justified. Never refund more than the customer
is owed. When you are done, write a short, friendly reply to the customer."""

TOOLS = [
    {
        "name": "get_customer",
        "description": "Look up a customer in the CRM: name, plan, account standing.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "list_charges",
        "description": "List the charges on a customer's account.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "refund_charge",
        "description": "Refund a single charge in full.",
        "input_schema": {
            "type": "object",
            "properties": {"charge_id": {"type": "string"}},
            "required": ["charge_id"],
        },
    },
]


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@observe(name="crm.get_customer", span_type="TOOL")
def get_customer(email: str) -> dict:
    url = f"{CRM_API_URL}/customers?{urllib.parse.urlencode({'email': email})}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.load(response)
    except OSError as e:
        return {"error": f"CRM unavailable: {e}"}


@observe(span_type="TOOL")
def list_charges(email: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, description, amount, created_at FROM charges WHERE email = ?",
            (email,),
        ).fetchall()
    return [dict(row) for row in rows]


@observe(span_type="TOOL")
def refund_charge(charge_id: str) -> dict:
    with db() as conn:
        charge = conn.execute(
            "SELECT amount FROM charges WHERE id = ?", (charge_id,)
        ).fetchone()
        if charge is None:
            return {"error": f"No charge {charge_id}"}
        conn.execute(
            "INSERT INTO refunds (charge_id, amount) VALUES (?, ?)",
            (charge_id, charge["amount"]),
        )
    return {"refunded": charge["amount"]}


@observe()
def handle_ticket(ticket: str) -> str:
    messages = [{"role": "user", "content": ticket}]
    for _ in range(8):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = {
                    "get_customer": get_customer,
                    "list_charges": list_charges,
                    "refund_charge": refund_charge,
                }
                output = fn[block.name](**block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output),
                    }
                )
        messages.append({"role": "user", "content": results})
    return "Sorry, I could not finish handling this ticket."


@observe()
def post_reply(ticket_id: str, reply: str) -> None:
    """Send the reply to the customer through the helpdesk."""
    request = urllib.request.Request(
        f"{HELPDESK_API_URL}/tickets/{ticket_id}/replies",
        data=json.dumps({"body": reply}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(request, timeout=10).close()


if __name__ == "__main__":
    # python -m app.support_agent case.json: answer the case's ticket through
    # the helpdesk, and print the reply.
    with open(sys.argv[1]) as f:
        case = json.load(f)
    reply = handle_ticket(case["ticket"])
    post_reply(case["ticket_id"], reply)
    print(reply)

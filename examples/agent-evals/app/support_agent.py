import json
import os
import sqlite3

from anthropic import Anthropic
from lmnr import Laminar, observe

Laminar.initialize()
client = Anthropic()

DB_PATH = os.environ.get("APP_DB", "/srv/app/app.db")
MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a support agent for a subscription software company.
Customers write in about billing problems. Use the tools to look up their charges
and issue refunds when a refund is justified. Never refund more than the customer
is owed. When you are done, write a short, friendly reply to the customer."""

TOOLS = [
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
                fn = {"list_charges": list_charges, "refund_charge": refund_charge}
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

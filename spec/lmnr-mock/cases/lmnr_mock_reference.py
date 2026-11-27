"""Reference handlers for the lmnr-mock conformance cases (`mocks.json` maps
each key to one of these)."""

from __future__ import annotations

from typing import Any

from lmnr.evals import mock


def get_customer(call: mock.Call, ctx: mock.Context) -> Any:
    return {"id": "cus_1", "email": call.args["email"], "plan": "pro"}


def find_order(call: mock.Call, ctx: mock.Context) -> Any:
    return None


def echo_args(call: mock.Call, ctx: mock.Context) -> Any:
    return {"args": call.args, "index": call.index}


def refund(call: mock.Call, ctx: mock.Context) -> Any:
    if ctx.state.setdefault("refunded", False):
        raise mock.Error("DuplicateRefund", "charge already refunded")
    ctx.state["refunded"] = True
    return {"status": "ok"}


def web_search(call: mock.Call, ctx: mock.Context) -> Any:
    return mock.PASSTHROUGH


def check_inventory(call: mock.Call, ctx: mock.Context) -> Any:
    raise ValueError("inventory handler is broken")


async def slow_quote(call: mock.Call, ctx: mock.Context) -> Any:
    return {"quote": 42, "earlier_calls": len(ctx.calls)}

"""`wrap_mcp_server` pieces that don't need an MCP server: redaction, turns, ids."""

from __future__ import annotations

import hashlib

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from lmnr.opentelemetry_lib.tracing.id_generator import (
    LaminarIdGenerator,
    can_force_trace_id,
    forced_trace_id,
)
from lmnr.opentelemetry_lib.tracing.tracer import get_laminar_tracer_provider
from lmnr.sdk.mcp_server.redaction import REDACTED, is_secret_key, normalize_keys, redact
from lmnr.sdk.mcp_server.turns import TurnTracker, turn_trace_id

# shared with the TS SDK: both must agree
SECRET = [
    "password",
    "user_password",
    "apiKey",
    "api_key",
    "X-API-Key",
    "client_secret",
    "access_token",
    "refreshToken",
    "githubToken",
    "Authorization",
    "auth",
    "session_cookie",
    "private_key",
    "aws_access_key_id",
    "credentials",
    "jwt",
]
NOT_SECRET = [
    "max_tokens",
    "tokens",
    "author",
    "query",
    "session_id",
    "token_count",
    "path",
    "keyword",
]


@pytest.mark.parametrize("key", SECRET)
def test_secret_keys(key: str):
    assert is_secret_key(key)


@pytest.mark.parametrize("key", NOT_SECRET)
def test_not_secret_keys(key: str):
    assert not is_secret_key(key)


def test_redact_is_recursive_and_does_not_mutate():
    value = {
        "query": "q",
        "auth": {"nested": "x"},
        "items": [{"apiKey": 1, "keep": [{"password": None}]}],
        "max_tokens": 5,
    }
    redacted = redact(value)
    assert redacted == {
        "query": "q",
        "auth": REDACTED,
        "items": [{"apiKey": REDACTED, "keep": [{"password": REDACTED}]}],
        "max_tokens": 5,
    }
    assert value["auth"] == {"nested": "x"}


def test_extra_keys_match_on_the_normalized_form():
    extra = normalize_keys(["Customer-ID"])
    assert redact({"customer_id": "c", "customerId": "d", "customer": "e"}, extra) == {
        "customer_id": REDACTED,
        "customerId": REDACTED,
        "customer": "e",
    }


def test_turn_trace_id_is_deterministic():
    expected = int(hashlib.sha256(b"sess\nturn").hexdigest()[:32], 16)
    assert turn_trace_id("sess", "turn") == expected
    assert turn_trace_id(None, "turn") == int(hashlib.sha256(b"\nturn").hexdigest()[:32], 16)
    assert turn_trace_id("other", "turn") != expected


def test_turn_tracker_gap_and_in_flight():
    now = [0.0]
    tracker = TurnTracker(20, clock=lambda: now[0])
    first = tracker.begin("s")
    tracker.end("s")
    now[0] = 20
    assert tracker.begin("s") == first  # at most `gap` after the last activity
    # a call still in flight keeps the turn open however long it runs
    now[0] = 100
    assert tracker.begin("s") == first
    tracker.end("s")
    tracker.end("s")
    now[0] = 120
    assert tracker.begin("s") == first  # the end at 100 was activity
    tracker.end("s")
    now[0] = 140.5
    second = tracker.begin("s")
    assert second != first
    # an explicit (turn id / traceparent) trace becomes the session's turn
    assert tracker.begin("s", 0xABC) == 0xABC
    assert tracker.begin("s") == 0xABC


def test_turn_tracker_is_capped():
    now = [0.0]
    tracker = TurnTracker(20, max_sessions=3, clock=lambda: now[0])
    for i in range(3):
        _ = tracker.begin(f"s{i}")
        tracker.end(f"s{i}")
    now[0] = 5
    _ = tracker.begin("busy")  # evicts the oldest idle one
    assert len(tracker) == 3
    now[0] = 100
    _ = tracker.begin("late")  # idle past the gap: evicted lazily
    assert len(tracker) <= 3


def test_id_generator_forces_root_trace_ids_only():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(id_generator=LaminarIdGenerator())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with forced_trace_id(0x1234), tracer.start_as_current_span("root") as root:
        child = tracer.start_span("child")
        child.end()
    other = tracer.start_span("other")
    other.end()
    assert root.get_span_context().trace_id == 0x1234
    assert child.get_span_context().trace_id == 0x1234
    assert other.get_span_context().trace_id != 0x1234
    with forced_trace_id(None):
        unforced = tracer.start_span("unforced")
        unforced.end()
    assert unforced.get_span_context().trace_id not in (0, 0x1234)


def test_can_force_trace_id(span_exporter: InMemorySpanExporter):
    # Laminar's own provider has the generator; a plain provider doesn't
    assert can_force_trace_id(get_laminar_tracer_provider())
    assert not can_force_trace_id(TracerProvider())
    assert not can_force_trace_id(object())

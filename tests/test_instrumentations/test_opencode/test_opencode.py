"""Tests for the opencode-ai instrumentation.

Mirrors the TypeScript ``packages/lmnr/test/opencode.test.ts`` suite with the
Python-specific differences that the Python SDK's ``SessionResource.chat``
takes ``parts`` as a keyword argument and applies Stainless's alias transforms
(``modelID`` / ``providerID``) before sending.
"""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from lmnr import observe


def test_injects_span_context_into_chat_parts_inside_observe(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    client = make_opencode_client()

    @observe(name="test-observe")
    def run():
        client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "Hello, world!"}],
        )

    run()

    assert captured_request.url == "http://localhost:4096/session/test-session/message"
    assert captured_request.body is not None

    sent_parts = captured_request.body["parts"]
    # Original part is preserved and kept first.
    assert sent_parts[0] == {"type": "text", "text": "Hello, world!"}

    # Exactly one synthetic metadata part appended.
    assert len(sent_parts) == 2
    metadata_part = sent_parts[1]
    assert metadata_part["type"] == "text"
    assert metadata_part["text"] == ""
    assert metadata_part["ignored"] is True
    assert metadata_part["synthetic"] is True

    span_context_json = metadata_part["metadata"]["lmnrSpanContext"]
    span_context = json.loads(span_context_json)
    assert span_context["trace_id"]
    assert span_context["span_id"]

    spans = span_exporter.get_finished_spans()
    observe_span = next(span for span in spans if span.name == "test-observe")
    assert observe_span is not None


@pytest.mark.asyncio
async def test_injects_span_context_into_async_chat_parts_inside_observe(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_async_opencode_client,
):
    client = make_async_opencode_client()

    @observe(name="test-observe-async")
    async def run():
        await client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "Async hello!"}],
        )

    await run()

    assert captured_request.body is not None
    sent_parts = captured_request.body["parts"]
    assert len(sent_parts) == 2

    metadata_part = sent_parts[1]
    assert metadata_part["type"] == "text"
    assert metadata_part["text"] == ""
    assert metadata_part["ignored"] is True
    assert metadata_part["synthetic"] is True

    span_context = json.loads(metadata_part["metadata"]["lmnrSpanContext"])
    assert span_context["trace_id"]
    assert span_context["span_id"]


def test_does_not_inject_when_no_active_span(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    client = make_opencode_client()

    # Call without an observe wrapper — no active Laminar span context.
    client.session.chat(
        "test-session",
        model_id="anthropic/claude-sonnet-4-5",
        provider_id="anthropic",
        parts=[{"type": "text", "text": "No context here"}],
    )

    assert captured_request.body is not None
    sent_parts = captured_request.body["parts"]
    assert len(sent_parts) == 1
    assert sent_parts[0] == {"type": "text", "text": "No context here"}


def test_preserves_multiple_existing_parts_when_injecting(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    client = make_opencode_client()

    original_parts = [
        {"type": "text", "text": "First part"},
        {"type": "text", "text": "Second part"},
        {
            "type": "file",
            "mime": "image/png",
            "url": "https://example.com/image.png",
        },
    ]

    @observe(name="test-multiple-parts")
    def run():
        client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=list(original_parts),
        )

    run()

    sent_parts = captured_request.body["parts"]
    # Original 3 parts are preserved in order + 1 injected at the end.
    assert len(sent_parts) == 4
    assert sent_parts[0]["text"] == "First part"
    assert sent_parts[1]["text"] == "Second part"
    assert sent_parts[2]["type"] == "file"
    assert sent_parts[3]["type"] == "text"
    assert sent_parts[3]["ignored"] is True
    assert sent_parts[3]["synthetic"] is True


def test_injection_does_not_mutate_callers_parts_list(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    """The instrumentation must not mutate the caller's ``parts`` list.

    Callers frequently reuse a ``parts`` list across turns; mutating it would
    cause a compounding injection where every subsequent call gains another
    synthetic part.
    """
    client = make_opencode_client()

    parts = [{"type": "text", "text": "Original"}]

    @observe(name="test-no-mutation")
    def run():
        client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=parts,
        )

    run()

    # Sent over the wire: 2 parts (1 original + 1 synthetic).
    assert len(captured_request.body["parts"]) == 2
    # Caller's list: still exactly 1 part.
    assert len(parts) == 1
    assert parts[0] == {"type": "text", "text": "Original"}


def test_span_context_trace_id_matches_observe_span(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    """The injected traceId must match the active observe span's trace."""
    client = make_opencode_client()

    @observe(name="test-trace-match")
    def run():
        client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "Trace matching test"}],
        )

    run()

    spans = span_exporter.get_finished_spans()
    observe_span = next(span for span in spans if span.name == "test-trace-match")

    metadata_part = captured_request.body["parts"][1]
    span_context = json.loads(metadata_part["metadata"]["lmnrSpanContext"])

    # LaminarSpanContext stores traceId as a canonical UUID string
    # (8-4-4-4-12). The OTel trace id is a 32-char hex; the canonical UUID
    # representation is the same 32 hex chars with dashes inserted.
    otel_hex = format(observe_span.context.trace_id, "032x")
    uuid_from_context = span_context["trace_id"].replace("-", "")
    assert uuid_from_context == otel_hex


def test_chat_returns_mocked_response_body(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    """Instrumentation must not swallow or mutate the SDK's response."""
    client = make_opencode_client(
        response_json={
            "info": {
                "id": "msg-42",
                "sessionID": "test-session",
                "role": "assistant",
            },
            "parts": [{"id": "p-1", "type": "text", "text": "Answer"}],
        },
    )

    @observe(name="test-response-passthrough")
    def run():
        return client.session.chat(
            "test-session",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "Q"}],
        )

    result = run()

    # AssistantMessage exposes ``info`` and ``parts`` attributes populated
    # from the response JSON.
    assert getattr(result, "info", None) is not None
    assert result.info["id"] == "msg-42"
    assert result.parts[0]["text"] == "Answer"


def test_nested_observe_uses_inner_span_context(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_opencode_client,
):
    """When ``chat`` is invoked inside a nested ``observe``, the injected
    span id must be that of the innermost span, not the outer one."""
    client = make_opencode_client()

    @observe(name="outer")
    def outer():
        @observe(name="inner")
        def inner():
            client.session.chat(
                "test-session",
                model_id="anthropic/claude-sonnet-4-5",
                provider_id="anthropic",
                parts=[{"type": "text", "text": "nested"}],
            )

        inner()

    outer()

    spans = span_exporter.get_finished_spans()
    inner_span = next(s for s in spans if s.name == "inner")
    outer_span = next(s for s in spans if s.name == "outer")
    assert inner_span.context.span_id != outer_span.context.span_id

    metadata_part = captured_request.body["parts"][1]
    span_context = json.loads(metadata_part["metadata"]["lmnrSpanContext"])

    inner_hex = format(inner_span.context.span_id, "016x")
    outer_hex = format(outer_span.context.span_id, "016x")
    injected_hex = span_context["span_id"].replace("-", "")
    # Injected span id comes from the innermost span. LaminarSpanContext stores
    # span_id as a full 32-hex-digit UUID whose last 16 hex digits are the OTel
    # span_id (see ``LaminarSpan.get_laminar_span_context``).
    assert injected_hex.endswith(inner_hex)
    assert not injected_hex.endswith(outer_hex)


@pytest.mark.asyncio
async def test_async_chat_does_not_inject_without_active_span(
    span_exporter: InMemorySpanExporter,
    captured_request,
    make_async_opencode_client,
):
    client = make_async_opencode_client()

    await client.session.chat(
        "test-session",
        model_id="anthropic/claude-sonnet-4-5",
        provider_id="anthropic",
        parts=[{"type": "text", "text": "no-span"}],
    )

    sent_parts = captured_request.body["parts"]
    assert len(sent_parts) == 1
    assert sent_parts[0]["text"] == "no-span"


@pytest.mark.asyncio
async def test_sync_and_async_can_coexist(
    span_exporter: InMemorySpanExporter,
    make_opencode_client,
    make_async_opencode_client,
):
    """Smoke test: sync and async clients don't stomp on each other's
    wrapping. Run both flavors and confirm each span gets recorded."""
    sync_client = make_opencode_client()
    async_client = make_async_opencode_client()

    @observe(name="sync-observe")
    def sync_call():
        sync_client.session.chat(
            "s1",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "s"}],
        )

    @observe(name="async-observe")
    async def async_call():
        await async_client.session.chat(
            "s2",
            model_id="anthropic/claude-sonnet-4-5",
            provider_id="anthropic",
            parts=[{"type": "text", "text": "a"}],
        )

    sync_call()
    await async_call()

    spans = span_exporter.get_finished_spans()
    names = {span.name for span in spans}
    assert "sync-observe" in names
    assert "async-observe" in names

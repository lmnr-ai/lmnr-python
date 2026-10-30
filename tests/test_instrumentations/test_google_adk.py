"""Tests for the Google ADK instrumentation.

The end-to-end tests drive a real `InMemoryRunner` against
gemini-3.5-flash-lite. The actual key was used during recording and the
requests/responses were saved to the VCR cassettes, so the asserted spans
are the ones ADK produces for real model turns.
"""

import asyncio
import json
import os
from collections.abc import Generator
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext
from opentelemetry.util.types import AttributeValue

pytest.importorskip("google.adk")

from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def get_weather(city: str) -> dict[str, str]:
    """Returns the weather for a city."""
    return {"city": city, "forecast": "sunny"}


def get_time(city: str) -> dict[str, str]:
    """Returns the local time for a city."""
    return {"city": city, "time": "12:00"}


def get_default_city() -> dict[str, str]:
    """Returns the default city."""
    return {"city": "Almaty"}


@pytest.fixture(scope="module")
def vcr_config():
    # Same filters as the shared config in conftest, plus decompression:
    # ADK talks to Gemini through aiohttp, and vcrpy records that path with
    # a decoded body while keeping the Content-Encoding: gzip header, which
    # breaks replay through httpx. Decoding at record time stores plain
    # bodies with matching headers.
    return {
        "filter_headers": [
            "authorization",
            "api-key",
            "x-goog-api-key",
            "x-api-key",
        ],
        "filter_query_parameters": ["key"],
        "decode_compressed_response": True,
    }


@pytest.fixture(autouse=True, params=["legacy", "v2"])
def adk_schema(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    # ADK 2.13 defaults to a semconv-aligned telemetry schema ("v2"): no
    # `call_llm` span (ADK's native `generate_content` span is the LLM span)
    # and tool payloads on `gen_ai.tool.call.*`. The legacy schema is still
    # reachable through an opt-in, and both must keep working.
    schema = cast(str, request.param)
    if schema == "v2":
        pytest.importorskip("google.adk.telemetry._schema_version", reason="ADK < 2.13 has no v2")
    if schema == "legacy":
        monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
    else:
        monkeypatch.delenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", raising=False)
        monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY")
    return schema


@pytest.fixture
def default_cassette_name(request: pytest.FixtureRequest) -> str:
    # One cassette per test, shared by both schemas: the HTTP traffic is identical.
    return cast(str, request.node.originalname)


@pytest.fixture(autouse=True)
def gemini_env(monkeypatch: pytest.MonkeyPatch):
    # Real key from the environment while recording; the placeholder is
    # enough for replay because vcr_config filters the key out of matches.
    monkeypatch.setenv("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "123"))


@pytest.fixture(scope="module", autouse=True)
def adk_instrumentation(span_exporter: InMemorySpanExporter) -> Generator[None]:
    # conftest.py's session fixture blocks GOOGLE_ADK: with `google-adk` a
    # pinned dev dependency, leaving it enabled would auto-remove
    # GOOGLE_GENAI from the session-wide default set (see
    # _GOOGLE_ADK_GENAI_CONFLICTS in tracing/instruments.py) and break every
    # other module's raw google_genai tests. This module instruments ADK and
    # uninstruments google_genai for its own tests, mirroring what a real
    # ADK-only application gets by default in production.
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_adk import (
        GoogleAdkInstrumentor,
    )
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai import (
        GoogleGenAiSdkInstrumentor,
    )

    adk_instrumentor = GoogleAdkInstrumentor()
    genai_instrumentor = GoogleGenAiSdkInstrumentor()
    adk_was_instrumented = adk_instrumentor.is_instrumented_by_opentelemetry
    genai_was_instrumented = genai_instrumentor.is_instrumented_by_opentelemetry

    if genai_was_instrumented:
        genai_instrumentor.uninstrument()
    if not adk_was_instrumented:
        adk_instrumentor.instrument()

    yield

    if not adk_was_instrumented:
        adk_instrumentor.uninstrument()
    if genai_was_instrumented:
        genai_instrumentor.instrument()


def run_agent(
    user_id: str = "test-user",
    instruction: str = (
        "Call get_weather for the city the user asks about, then answer in one short sentence."
    ),
    prompt: str = "Weather in Almaty?",
):
    agent = Agent(  # pyright: ignore[reportEmptyAbstractUsage]
        name="weather_agent",
        model="gemini-3.5-flash-lite",
        instruction=instruction,
        tools=[get_weather, get_time, get_default_city],
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )
    runner = InMemoryRunner(agent=agent, app_name="test-app")
    session = asyncio.run(
        runner.session_service.create_session(app_name="test-app", user_id=user_id)
    )
    message = types.Content(role="user", parts=[types.Part(text=prompt)])
    for _ in runner.run(user_id=session.user_id, session_id=session.id, new_message=message):
        pass
    return session


def llm_spans(span_exporter: InMemorySpanExporter, schema: str) -> list[ReadableSpan]:
    """The per-turn LLM spans: `call_llm` (legacy) or ADK's native `generate_content` (v2)."""
    if schema == "legacy":
        return spans_by_name(span_exporter, "call_llm")
    return [s for s in span_exporter.get_finished_spans() if s.name.startswith("generate_content ")]


def spans_by_name(span_exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in span_exporter.get_finished_spans() if s.name == name]


@pytest.mark.vcr
def test_tool_span_typed_with_input_and_output(
    span_exporter: InMemorySpanExporter, adk_schema: str
):
    _result = run_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    attributes = tool_span.attributes or {}
    assert attributes["lmnr.span.type"] == "TOOL"
    if adk_schema == "v2":
        # The backend reads gen_ai.tool.call.* on TOOL spans; nothing is copied.
        assert "lmnr.span.input" not in attributes
        assert "lmnr.span.output" not in attributes
        input_, output = (
            attributes["gen_ai.tool.call.arguments"],
            attributes["gen_ai.tool.call.result"],
        )
    else:
        input_, output = attributes["lmnr.span.input"], attributes["lmnr.span.output"]
    assert json.loads(cast(str, input_)) == {"city": "Almaty"}
    assert json.loads(cast(str, output)) == {"city": "Almaty", "forecast": "sunny"}


@pytest.mark.vcr
def test_call_llm_span_carries_gen_ai_attributes(
    span_exporter: InMemorySpanExporter, adk_schema: str
):
    if adk_schema == "v2":
        # No call_llm span: ADK's native generate_content span is the sole LLM
        # span per turn (two turns here), typed LLM. The enclosing
        # invoke_agent span must not be mistaken for one.
        _res = run_agent()
        generate_spans = llm_spans(span_exporter, adk_schema)
        assert len(generate_spans) == 2
        for span in generate_spans:
            assert (span.attributes or {})["lmnr.span.type"] == "LLM"
            assert (span.attributes or {})["gen_ai.request.model"] == "gemini-3.5-flash-lite"
        assert spans_by_name(span_exporter, "call_llm") == []
        (agent_span,) = spans_by_name(span_exporter, "invoke_agent weather_agent")
        assert "lmnr.span.type" not in (agent_span.attributes or {})
        return

    # With GOOGLE_GENAI excluded by default (google-adk installed), ADK's own
    # `call_llm` span is the sole LLM span per turn (two turns here: the
    # tool call and the final answer), enriched directly from the real
    # LlmRequest/LlmResponse objects instead of relying on a separate
    # google_genai span or ADK's raw gcp.vertex.agent.llm_request/response
    # JSON blobs.
    _res = run_agent()

    call_llm_spans = spans_by_name(span_exporter, "call_llm")
    assert len(call_llm_spans) == 2
    for span in call_llm_spans:
        assert (span.attributes or {})["lmnr.span.type"] == "LLM"
        assert (span.attributes or {})["gen_ai.request.model"] == "gemini-3.5-flash-lite"
        assert (span.attributes or {})["gen_ai.response.model"]
        assert json.loads(cast(str, (span.attributes or {})["gen_ai.input.messages"]))
        assert json.loads(cast(str, (span.attributes or {})["gen_ai.output.messages"]))
    assert json.loads(cast(str, (call_llm_spans[0].attributes or {})["gen_ai.tool.definitions"]))

    # Neither a separate Laminar google_genai span nor ADK's own native
    # "generate_content <model>" span should exist.
    assert spans_by_name(span_exporter, "gemini.generate_content") == []
    native = [
        s for s in span_exporter.get_finished_spans() if s.name.startswith("generate_content ")
    ]
    assert native == []


@pytest.mark.vcr
def test_tool_span_is_sibling_of_call_llm_not_child(
    span_exporter: InMemorySpanExporter, adk_schema: str
):
    # Regression: ADK's own `call_llm` span stays the ambient "current span"
    # (via its still-open start_as_current_span block) through the tool
    # postprocessing that immediately follows, unless the call_llm wrap
    # detaches it first. Confirm the tool span shares call_llm's parent
    # instead of nesting under call_llm itself.
    _res = run_agent()

    call_llm_spans = llm_spans(span_exporter, adk_schema)
    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")

    call_llm_span_ids = {_ctx(s).span_id for s in call_llm_spans}
    assert tool_span.parent is not None
    assert tool_span.parent.span_id not in call_llm_span_ids
    assert tool_span.parent.span_id in {
        s.parent.span_id for s in call_llm_spans if s.parent is not None
    }


@pytest.mark.vcr
def test_call_llm_span_ends_before_tool_execution(
    span_exporter: InMemorySpanExporter, adk_schema: str
):
    # Regression: call_llm's recorded duration must not stretch across the
    # tool-execution/callback postprocessing that follows inside ADK's
    # still-open start_as_current_span block — it should end as soon as the
    # model's last token arrives, not whenever ADK's own `with` block
    # eventually unwinds.
    _res = run_agent()

    call_llm_spans = llm_spans(span_exporter, adk_schema)
    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")

    # Both call_llm spans (tool-call turn, final-answer turn) share the same
    # invoke_agent parent, so pick the one that produced the function call by
    # timing: it's the turn that ran first.
    call_llm_span = min(call_llm_spans, key=lambda s: s.start_time or 0)
    assert call_llm_span.end_time is not None
    assert call_llm_span.end_time <= (tool_span.start_time or 0)


@pytest.mark.vcr
def test_call_llm_content_respects_adk_content_toggle(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    adk_schema: str,
):
    # With the content knob off, ADK stamps "{}" for the legacy
    # gcp.vertex.agent.llm_request/response attributes; the new gen_ai.*
    # attributes must not leak content through a side door.
    if adk_schema == "v2":
        pytest.skip("v2 has no call_llm span; its content is gated by ADK itself")
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "0")
    _res = run_agent()

    for span in spans_by_name(span_exporter, "call_llm"):
        assert (span.attributes or {})["lmnr.span.type"] == "LLM"
        assert "gen_ai.input.messages" not in (span.attributes or {})
        assert "gen_ai.tool.definitions" not in (span.attributes or {})
        assert "gen_ai.output.messages" not in (span.attributes or {})


@pytest.mark.vcr
def test_agent_span_carries_session_association(span_exporter: InMemorySpanExporter):
    session = run_agent(user_id="user-42")

    (agent_span,) = spans_by_name(span_exporter, "invoke_agent weather_agent")
    attributes = agent_span.attributes or {}
    assert attributes["lmnr.association.properties.session_id"] == session.id
    assert attributes["lmnr.association.properties.user_id"] == "user-42"


@pytest.mark.vcr
def test_tool_content_respects_adk_content_toggle(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    # With the content knob off, ADK stamps "{}" for tool args/response;
    # that must not leak into the Laminar input/output attributes.
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "0")
    _res = run_agent()

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_weather")
    assert (tool_span.attributes or {})["lmnr.span.type"] == "TOOL"
    assert "lmnr.span.input" not in (tool_span.attributes or {})
    assert "lmnr.span.output" not in (tool_span.attributes or {})


def test_adk_skips_native_span_under_legacy_schema(adk_schema: str):
    # ADK's own native `generate_content` span is redundant under the legacy
    # schema (call_llm is the LLM span), so the instrumentor tells ADK not to
    # emit it. Under v2 the native span is the LLM span and is kept.
    if adk_schema == "v2":
        pytest.skip("native span is kept under v2")
    from google.adk.telemetry import tracing

    assert tracing._should_emit_native_telemetry(None, None) is False  # pyright: ignore[reportArgumentType]


@pytest.mark.vcr
def test_merged_parallel_tool_span_typed(span_exporter: InMemorySpanExporter, adk_schema: str):
    # Two function calls in one turn produce an `execute_tool (merged)`
    # span via trace_merged_tool_calls, which flows.llm_flows.functions
    # binds by name at import time. This module imports ADK at collection,
    # before Laminar.initialize() runs, so this fails unless the
    # instrumentor patches the flow module's binding too.
    _res = run_agent(
        instruction=(
            "Call get_weather and get_time together in the same turn, then answer in one short sentence."
        ),
        prompt="Weather and local time in Almaty?",
    )

    (merged_span,) = spans_by_name(span_exporter, "execute_tool (merged)")
    assert (merged_span.attributes or {})["lmnr.span.type"] == "TOOL"
    if adk_schema == "legacy":
        assert "lmnr.span.output" in (merged_span.attributes or {})


@pytest.mark.vcr
def test_empty_args_tool_is_not_mistaken_for_redaction(
    span_exporter: InMemorySpanExporter, adk_schema: str
):
    # A niladic tool's args serialize to "{}", same as ADK's redaction
    # sentinel; the (empty) input must still be recorded.
    _res = run_agent(
        instruction=("Call get_default_city, then answer with just the city name."),
        prompt="Which city am I in?",
    )

    (tool_span,) = spans_by_name(span_exporter, "execute_tool get_default_city")
    attributes = tool_span.attributes or {}
    assert attributes["lmnr.span.type"] == "TOOL"
    if adk_schema == "v2":
        assert attributes["gen_ai.tool.call.arguments"] == "{}"
        assert json.loads(cast(str, attributes["gen_ai.tool.call.result"])) == {"city": "Almaty"}
    else:
        assert attributes["lmnr.span.input"] == "{}"
        assert json.loads(cast(str, attributes["lmnr.span.output"])) == {"city": "Almaty"}


def test_agent_enrichment_keeps_explicit_session_id():
    # An id already on the span was set explicitly through Laminar (the
    # processor stamps context values at span start); the derived ADK id
    # must not overwrite it.
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    class FakeSession:
        id: str = "adk-session-uuid"
        user_id: str = "adk-user"

    class FakeCtx:
        session: FakeSession = FakeSession()

    class FakeSpan:
        def __init__(self, attributes: dict[str, AttributeValue] | Any):
            self.attributes: dict[str, AttributeValue] = dict(attributes)

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key: str, value: AttributeValue):
            self.attributes[key] = value

    span = FakeSpan({"lmnr.association.properties.session_id": "checkout-42"})
    google_adk._wrap_trace_agent_invocation(
        lambda *a, **k: None, None, (span, object(), FakeCtx()), {}
    )
    assert span.attributes["lmnr.association.properties.session_id"] == "checkout-42"
    assert span.attributes["lmnr.association.properties.user_id"] == "adk-user"


def test_partial_call_llm_response_does_not_detach_or_end_span(adk_schema: str):
    if adk_schema == "v2":
        pytest.skip("v2 has no call_llm span")
    # Streaming turns run trace_call_llm once per chunk with partial=True
    # for every fragment but the last. Detaching/ending call_llm on a
    # partial chunk would be premature: the call hasn't actually finished,
    # so a later trace_call_llm call for the same turn would enrich/end (or,
    # via _resolve_span's current-span fallback, attach a tool span onto)
    # call_llm's parent instead of call_llm itself.
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    class FakeSpan:
        def __init__(self):
            self.attributes: dict[str, AttributeValue] = {}
            self.parent: Any = None
            self.ended: bool = False

        def is_recording(self) -> bool:
            return not self.ended

        def set_attribute(self, key: str, value: AttributeValue):
            self.attributes[key] = value

        def end(self):
            self.ended = True

    class FakeConfig:
        system_instruction: str | None = None
        tools: list[Any] | None = None

    class FakeLlmRequest:
        model: str = "gemini-3.5-flash-lite"
        config: FakeConfig = FakeConfig()
        contents: list[Any] | None = None

    class FakeLlmResponse:
        partial: bool = True
        content: Any = None
        model_version: str = "gemini-3.5-flash-lite"

    span = FakeSpan()
    detach_calls = []
    original_detach = google_adk._detach_from_current_context
    google_adk._detach_from_current_context = detach_calls.append
    try:
        google_adk._wrap_trace_call_llm(
            lambda *a, **k: None,
            None,
            (object(), "event-1", FakeLlmRequest(), FakeLlmResponse(), span),
            {},
        )
    finally:
        google_adk._detach_from_current_context = original_detach

    assert detach_calls == []
    assert span.ended is False
    # Enrichment (including the span type) still runs on every chunk.
    assert span.attributes["lmnr.span.type"] == "LLM"


def test_uninstrument_unwraps_lazily_imported_binding():
    # initialize() before any ADK import means the flow module is first
    # imported by the instrumentor itself, mid wrap-loop; its by-name
    # binding of trace_merged_tool_calls must come out single-wrapped, and
    # uninstrument must leave no live layer on any binding.
    import sys

    from _wrapt_compat import is_wrapt_proxy
    from google.adk.telemetry import tracing

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
        google_adk,
    )

    # Whichever flow module binds it in the installed ADK version.
    (module_name,) = [
        name
        for name in (
            google_adk._FLOW_FUNCTIONS_MODULE,
            google_adk._BATCH_TOOL_EXECUTOR_MODULE,
            google_adk._TOOLS_BATCH_EXECUTOR_MODULE,
        )
        if hasattr(sys.modules.get(name), "trace_merged_tool_calls")
    ]
    original_module = sys.modules[module_name]
    parent_name, _, attr_name = module_name.rpartition(".")
    instrumentor = google_adk.GoogleAdkInstrumentor()
    instrumentor.uninstrument()
    _popped_val = sys.modules.pop(module_name)
    try:
        instrumentor.instrument()
        fresh = sys.modules[module_name]
        assert is_wrapt_proxy(fresh.trace_merged_tool_calls)
        assert not is_wrapt_proxy(fresh.trace_merged_tool_calls.__wrapped__)
        instrumentor.uninstrument()
        assert not is_wrapt_proxy(fresh.trace_merged_tool_calls)
        assert not is_wrapt_proxy(tracing.trace_merged_tool_calls)
    finally:
        sys.modules[module_name] = original_module
        setattr(sys.modules[parent_name], attr_name, original_module)
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()

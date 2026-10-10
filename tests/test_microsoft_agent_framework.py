"""Tests for the Microsoft Agent Framework instrumentation.

The end-to-end tests drive real `Agent` runs against gpt-5-mini through the
framework's OpenAI client. The real key was used during recording and the
requests/responses were saved to the VCR cassettes, so the asserted spans are
the ones the framework produces for real model turns.
"""

import asyncio
import json
import os
from collections.abc import Iterable
from importlib.metadata import version
from typing import Annotated, Any, cast
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

pytest.importorskip("agent_framework")

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from opentelemetry import trace
from opentelemetry.trace import SpanContext

from lmnr import Laminar, observe
from lmnr.opentelemetry_lib.opentelemetry.instrumentation import (
    microsoft_agent_framework as maf_instrumentation,
)

MODEL = "gpt-5-mini"


def _ctx(span: ReadableSpan) -> SpanContext:
    ctx = span.get_span_context()
    assert ctx is not None
    return ctx


def _parent(span: ReadableSpan) -> SpanContext:
    assert span.parent is not None
    return span.parent


@pytest.fixture(autouse=True)
def openai_env(monkeypatch: pytest.MonkeyPatch):
    # Real key from the environment while recording; the placeholder is
    # enough for replay because vcr_config filters the key out of matches.
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "test"))


@observe(name="lookup_weather")
def lookup_weather(city: str) -> str:
    return f"Sunny, 22C in {city}"


@tool
def get_weather(city: Annotated[str, "City name"]) -> str:
    """Get the weather for a city."""
    return lookup_weather(city)


def make_agent() -> Agent:
    return Agent(
        client=OpenAIChatClient(model=MODEL),
        name="WeatherAgent",
        instructions="You are a weather assistant. Always use tools. One sentence.",
        tools=[get_weather],
    )


def run_agent(stream: bool = False, session_id: str | None = None) -> str:
    @observe(name="root")
    async def main() -> str:
        if session_id:
            Laminar.set_trace_session_id(session_id)
        agent = make_agent()
        if stream:
            text = ""
            async for update in agent.run("Weather in Paris?", stream=True):
                text += update.text or ""
            return text
        return (await agent.run("Weather in Paris?")).text

    return asyncio.run(main())


def spans_by_name(spans: Iterable[ReadableSpan], name: str) -> list[ReadableSpan]:
    return [s for s in spans if s.name == name]


def assert_agent_tree(spans: Iterable[ReadableSpan]) -> tuple[list[ReadableSpan], ReadableSpan]:
    [root] = spans_by_name(spans, "root")
    [agent] = spans_by_name(spans, "invoke_agent WeatherAgent")
    chats = spans_by_name(spans, f"chat {MODEL}")
    [tool_span] = spans_by_name(spans, "execute_tool get_weather")
    [lookup] = spans_by_name(spans, "lookup_weather")

    assert _parent(agent).span_id == _ctx(root).span_id
    assert len(chats) == 2
    assert all(_parent(c).span_id == _ctx(agent).span_id for c in chats)
    assert _parent(tool_span).span_id == _ctx(agent).span_id
    # An @observe function called from a tool nests under the tool span.
    assert _parent(lookup).span_id == _ctx(tool_span).span_id
    assert len({_ctx(s).trace_id for s in spans}) == 1
    # The framework's chat span is the LLM span: the OpenAI SDK call made
    # underneath it is not traced a second time.
    assert not [s for s in spans if s.name.startswith("openai.")]
    return chats, tool_span


@pytest.mark.vcr
def test_agent_run_span_tree(span_exporter: InMemorySpanExporter):
    assert "Paris" in run_agent()
    spans = span_exporter.get_finished_spans()
    chats, tool_span = assert_agent_tree(spans)

    for chat in chats:
        assert (chat.attributes or {})["gen_ai.operation.name"] == "chat"
        assert (chat.attributes or {})["gen_ai.system"] == "openai"
        assert (chat.attributes or {})["gen_ai.request.model"] == MODEL
        assert cast(int, (chat.attributes or {})["gen_ai.usage.input_tokens"]) > 0
        assert json.loads(cast(str, (chat.attributes or {})["gen_ai.input.messages"]))
        assert json.loads(cast(str, (chat.attributes or {})["gen_ai.output.messages"]))
        [definition] = json.loads(cast(str, (chat.attributes or {})["gen_ai.tool.definitions"]))
        assert definition["name"] == "get_weather"

    assert json.loads(cast(str, (tool_span.attributes or {})["gen_ai.tool.call.arguments"])) == {
        "city": "Paris"
    }
    assert "Sunny" in cast(str, (tool_span.attributes or {})["gen_ai.tool.call.result"])

    framework_spans = [
        s for s in spans if s.instrumentation_scope is not None and s.instrumentation_scope.name == "agent_framework"
    ]
    assert framework_spans
    for span in framework_spans:
        attributes = span.attributes or {}
        assert attributes["lmnr.span.instrumentation_scope.name"] == "agent-framework"
        assert attributes["lmnr.span.instrumentation_scope.version"] == version(
            "agent-framework-core"
        )


@pytest.mark.vcr
def test_streaming_agent_run_span_tree(span_exporter: InMemorySpanExporter):
    assert "Paris" in run_agent(stream=True)
    chats, _ = assert_agent_tree(span_exporter.get_finished_spans())
    assert all(json.loads(cast(str, (c.attributes or {})["gen_ai.output.messages"])) for c in chats)


@pytest.mark.vcr
def test_session_id_propagates_to_framework_spans(span_exporter: InMemorySpanExporter):
    run_agent(session_id="maf-session")
    framework_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.instrumentation_scope is not None and s.instrumentation_scope.name == "agent_framework"
    ]
    assert framework_spans
    for span in framework_spans:
        assert (
            (span.attributes or {})["lmnr.association.properties.session_id"]
            == "maf-session"
        )


@pytest.mark.vcr
def test_direct_openai_call_outside_agent_is_traced(span_exporter: InMemorySpanExporter):
    from openai import OpenAI

    _response = OpenAI().responses.create(model=MODEL, input="Say hi in one word.")
    assert spans_by_name(span_exporter.get_finished_spans(), "openai.response")


def test_framework_spans_use_laminar_provider_without_global_provider():
    # With `set_global_tracer_provider=False` the framework's own lookup
    # would land on a no-op provider.
    from agent_framework.observability import get_tracer

    with patch(
        "opentelemetry.trace.get_tracer_provider",
        return_value=trace.NoOpTracerProvider(),
    ):
        span = get_tracer().start_span("probe")
    assert span.is_recording()
    span.end()


def test_uninstrument_restores_every_hook():
    import agent_framework._mcp as mcp
    import agent_framework.observability as observability
    from wrapt import ObjectProxy

    def wrapped_hooks():
        return {
            name
            for name, value in (
                *(
                    (name, getattr(observability, name))
                    for name in (
                        "get_tracer",
                        "_get_span",
                        "_activate_span",
                        "_get_span_attributes",
                        "_get_response_attributes",
                        "create_mcp_client_span",
                    )
                ),
                (
                    "ChatTelemetryLayer.get_response",
                    observability.ChatTelemetryLayer.__dict__["get_response"],
                ),
                ("_mcp.create_mcp_client_span", mcp.create_mcp_client_span),
            )
            if isinstance(value, ObjectProxy)
        }

    instrumentor = maf_instrumentation.MicrosoftAgentFrameworkInstrumentor()
    tracer_provider = instrumentor._tracer_provider
    assert len(wrapped_hooks()) == 8
    instrumentor.uninstrument()
    try:
        assert wrapped_hooks() == set()
    finally:
        instrumentor.instrument(tracer_provider=tracer_provider)
    assert len(wrapped_hooks()) == 8


def test_content_capture_respects_opt_outs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENABLE_SENSITIVE_DATA", raising=False)
    monkeypatch.delenv("LMNR_TRACE_CONTENT", raising=False)
    assert maf_instrumentation._should_enable_sensitive_data()

    monkeypatch.setenv("LMNR_TRACE_CONTENT", "false")
    assert not maf_instrumentation._should_enable_sensitive_data()

    monkeypatch.delenv("LMNR_TRACE_CONTENT")
    monkeypatch.setenv("ENABLE_SENSITIVE_DATA", "false")
    assert not maf_instrumentation._should_enable_sensitive_data()


def test_instrumentation_respects_explicit_setting(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENABLE_INSTRUMENTATION", raising=False)
    assert maf_instrumentation._should_enable_instrumentation()

    monkeypatch.setenv("ENABLE_INSTRUMENTATION", "false")
    assert not maf_instrumentation._should_enable_instrumentation()


def test_instrumentation_enables_content_capture():
    from agent_framework.observability import OBSERVABILITY_SETTINGS

    assert OBSERVABILITY_SETTINGS.ENABLED
    assert OBSERVABILITY_SETTINGS.SENSITIVE_DATA_ENABLED


def test_workflow_spans_keep_span_path(span_exporter: InMemorySpanExporter):
    # Workflow spans are named with the framework's `OtelAttr` str enum. Left
    # as is, OTel drops `lmnr.span.path` for them and every descendant.
    from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
    from typing_extensions import Never

    class Upper(Executor):
        @handler
        async def process(self, text: str, ctx: WorkflowContext[str]) -> None:
            await ctx.send_message(text.upper())

    class Reverse(Executor):
        @handler
        async def process(self, text: str, ctx: WorkflowContext[Never, str]) -> None:
            await ctx.yield_output(text[::-1])

    @observe(name="root")
    async def main():
        upper, reverse = Upper(id="upper"), Reverse(id="reverse")
        workflow = WorkflowBuilder(start_executor=upper).add_edge(upper, reverse).build()
        return (await workflow.run("hello")).get_outputs()

    assert asyncio.run(main()) == ["OLLEH"]

    spans = span_exporter.get_finished_spans()
    by_id = {_ctx(s).span_id: s for s in spans}
    framework_spans = [
        s for s in spans if s.instrumentation_scope is not None and s.instrumentation_scope.name == "agent_framework"
    ]
    assert {"workflow.build", "workflow.run", "message.send"} <= {
        s.name for s in framework_spans
    }
    for span in framework_spans:
        assert type(span.name) is str
        path = list(cast(tuple[str, ...], (span.attributes or {})["lmnr.span.path"]))
        assert path[0] == "root"
        assert path[-1] == span.name
        parent_path = cast(
            tuple[str, ...],
            (by_id[_parent(span).span_id].attributes or {})["lmnr.span.path"],
        )
        assert path[:-1] == list(parent_path)


@pytest.mark.parametrize(
    "provider, kwargs, expected_output, expected_reasoning",
    [
        # Gemini bills thinking as output but reports it separately.
        ("gcp.gemini", {}, 40, 30),
        # OpenAI's output tokens already include reasoning tokens.
        ("openai", {}, 10, None),
        ("gcp.gemini", {"capture_usage": False}, None, None),
    ],
)
def test_gemini_thinking_tokens_count_as_output(
    provider: Any,
    kwargs: dict[str, Any],
    expected_output: int,
    expected_reasoning: int,
):
    from agent_framework import ChatResponse
    from agent_framework.observability import _get_response_attributes

    response = ChatResponse(
        messages=[],
        usage_details={"output_token_count": 10, "reasoning_output_token_count": 30},
    )
    attributes = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider}
    _ = _get_response_attributes(attributes, response, **kwargs)
    # A second call on the same attributes must not add the tokens again.
    _ = _get_response_attributes(attributes, response, **kwargs)

    assert attributes.get("gen_ai.usage.output_tokens") == expected_output
    assert attributes.get("gen_ai.usage.reasoning_tokens") == expected_reasoning


def test_mcp_tool_call_span_is_not_a_second_tool_span(span_exporter: InMemorySpanExporter):
    # `tools/call` carries `gen_ai.operation.name = execute_tool`, which made
    # Laminar show it as a duplicate tool span inside the real `execute_tool`.
    from agent_framework import _mcp

    with _mcp.create_mcp_client_span(
        "tools/call",
        target="add",
        attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "add"},
    ):
        pass

    [span] = spans_by_name(span_exporter.get_finished_spans(), "tools/call add")
    assert "gen_ai.operation.name" not in (span.attributes or {})
    assert (span.attributes or {})["mcp.method.name"] == "tools/call"
    assert (span.attributes or {})["gen_ai.tool.name"] == "add"

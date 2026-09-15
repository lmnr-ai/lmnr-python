"""Regression tests for the shape of the trace tree the Agents processor builds.

Drives `LaminarAgentsTraceProcessor` directly with stand-in traces and spans, so
no agent run, network or cassette is involved.
"""

import itertools
from types import SimpleNamespace

import pytest
from agents.tracing import span_data as agents_span_data

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.processor import (
    LaminarAgentsTraceProcessor,
)

TRACE_ID = "trace_shape_tests"

_ids = itertools.count(1)


def _trace(name="Agent workflow"):
    return SimpleNamespace(
        trace_id=TRACE_ID, name=name, metadata=None, group_id=None
    )


def _span(span_data, parent_id=None):
    return SimpleNamespace(
        trace_id=TRACE_ID,
        span_id=f"span_{next(_ids)}",
        parent_id=parent_id,
        span_data=span_data,
        error=None,
    )


@pytest.fixture
def processor():
    processor = LaminarAgentsTraceProcessor()
    yield processor
    processor.shutdown()


def _spans_by_name(span_exporter):
    return {span.name: span for span in span_exporter.get_finished_spans()}


def test_root_span_path_uses_the_trace_name(processor, span_exporter):
    """`lmnr.span.path` is captured at span start, so a placeholder root name
    leaks into the path of every span in the trace."""
    trace = _trace()
    processor.on_trace_start(trace)
    child = _span(agents_span_data.CustomSpanData(name="step", data={}))
    processor.on_span_start(child)
    processor.on_span_end(child)
    processor.on_trace_end(trace)

    spans = _spans_by_name(span_exporter)
    assert list(spans["Agent workflow"].attributes["lmnr.span.path"]) == [
        "Agent workflow"
    ]
    assert list(spans["step"].attributes["lmnr.span.path"]) == [
        "Agent workflow",
        "step",
    ]


def test_root_span_path_is_repaired_when_a_span_arrives_first(
    processor, span_exporter
):
    """The trace name is unknown until on_trace_start, so the root is renamed -
    the path captured at start has to follow."""
    child = _span(agents_span_data.CustomSpanData(name="step", data={}))
    processor.on_span_start(child)
    trace = _trace()
    processor.on_trace_start(trace)
    processor.on_span_end(child)
    processor.on_trace_end(trace)

    spans = _spans_by_name(span_exporter)
    assert "agents.trace" not in spans
    assert list(spans["Agent workflow"].attributes["lmnr.span.path"]) == [
        "Agent workflow"
    ]


def test_handoff_destination_is_not_parented_onto_a_closed_span(
    processor, span_exporter
):
    """The SDK finishes the source agent's span before starting the
    destination's, so reparenting the destination under the handoff's parent
    makes a child that starts after its parent ended."""
    trace = _trace()
    processor.on_trace_start(trace)

    source = _span(agents_span_data.AgentSpanData(name="Agent A"))
    processor.on_span_start(source)
    handoff = _span(
        agents_span_data.HandoffSpanData(from_agent="Agent A", to_agent="Agent B"),
        parent_id=source.span_id,
    )
    processor.on_span_start(handoff)
    processor.on_span_end(handoff)
    processor.on_span_end(source)

    destination = _span(agents_span_data.AgentSpanData(name="Agent B"))
    processor.on_span_start(destination)
    processor.on_span_end(destination)
    processor.on_trace_end(trace)

    spans = _spans_by_name(span_exporter)
    root, agent_a, agent_b = spans["Agent workflow"], spans["Agent A"], spans["Agent B"]
    # Siblings under the root, which is still open - not nested under the
    # already-ended source agent.
    assert agent_b.parent.span_id == root.context.span_id
    assert agent_b.start_time >= agent_a.end_time
    assert list(agent_b.attributes["lmnr.span.path"]) == ["Agent workflow", "Agent B"]


def test_a_span_never_outlives_its_parent(processor, span_exporter):
    trace = _trace()
    processor.on_trace_start(trace)
    agent = _span(agents_span_data.AgentSpanData(name="Agent A"))
    processor.on_span_start(agent)
    handoff = _span(
        agents_span_data.HandoffSpanData(from_agent="Agent A", to_agent="Agent B"),
        parent_id=agent.span_id,
    )
    processor.on_span_start(handoff)
    processor.on_span_end(handoff)
    processor.on_span_end(agent)
    destination = _span(agents_span_data.AgentSpanData(name="Agent B"))
    processor.on_span_start(destination)
    processor.on_span_end(destination)
    processor.on_trace_end(trace)

    spans = span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    for span in spans:
        parent = by_id.get(span.parent.span_id) if span.parent else None
        if parent is not None:
            assert span.start_time >= parent.start_time, span.name
            assert span.end_time <= parent.end_time, span.name

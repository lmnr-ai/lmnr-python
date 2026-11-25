"""Recorded LangSmith runs, replayed through the bridge's recorder.

`fixtures/python_sdk_payloads.json` holds what langsmith 0.16.2 handed to
`Client._create_run` / `_update_run` (the bridge's tap point) for each
integration; see `bridges/langsmith/AGENTS.md` for how to re-record it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan

from lmnr.opentelemetry_lib.bridges.langsmith.runs import (
    LangSmithRunRecorder,
    span_id_for,
    trace_id_for,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_BRIDGE_NAME,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_PATH,
    SPAN_TYPE,
)

from .conftest import Laminar, ids, parent_id

FIXTURES: dict[str, list[dict[str, Any]]] = json.loads(
    (Path(__file__).parent / "fixtures" / "python_sdk_payloads.json").read_text()
)


def replay(laminar: Laminar, scenario: str) -> dict[str, list[ReadableSpan]]:
    recorder = LangSmithRunRecorder(laminar.processor, laminar.provider, bridge_version="0.16.2")
    for record in FIXTURES[scenario]:
        run = json.loads(json.dumps(record["run"]))
        if record["op"] == "create":
            recorder.on_create(run, Context())
        else:
            recorder.on_update(run)
    spans: dict[str, list[ReadableSpan]] = {}
    for span in laminar.spans():
        spans.setdefault(span.name, []).append(span)
    return spans


def attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def messages(span: ReadableSpan, key: str) -> list[Any]:
    return json.loads(str(attrs(span)[key]))


def test_traceable(laminar: Laminar) -> None:
    spans = replay(laminar, "traceable")
    agent, search, llm = spans["agent"][0], spans["search"][0], spans["llm"][0]

    root_create = FIXTURES["traceable"][0]["run"]
    assert ids(agent).trace_id == trace_id_for(root_create["trace_id"])
    assert ids(agent).span_id == span_id_for(root_create["id"])
    assert agent.parent is None
    assert parent_id(search) == ids(agent).span_id
    assert parent_id(llm) == ids(agent).span_id
    assert attrs(llm)[SPAN_PATH] == ("agent", "llm")

    assert attrs(agent)[SPAN_TYPE] == "DEFAULT"
    assert attrs(search)[SPAN_TYPE] == "TOOL"
    assert attrs(llm)[SPAN_TYPE] == "LLM"
    assert attrs(agent)[SPAN_BRIDGE_NAME] == "langsmith"
    assert json.loads(attrs(search)[SPAN_INPUT]) == {"query": "hello"}
    assert json.loads(attrs(search)[SPAN_OUTPUT]) == {"output": ["a", "b"]}

    llm_attrs = attrs(llm)
    assert messages(llm, "gen_ai.input.messages") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]
    assert messages(llm, "gen_ai.output.messages") == [{"role": "assistant", "content": "hi"}]
    assert llm_attrs["gen_ai.request.model"] == "gpt-4o"
    assert llm_attrs["gen_ai.system"] == "openai"
    assert llm_attrs["gen_ai.usage.input_tokens"] == 3
    assert llm_attrs["gen_ai.usage.output_tokens"] == 2
    assert llm_attrs["llm.usage.total_tokens"] == 5

    # Session / user from metadata on every span, trace metadata on the root
    # only, without LangSmith's own keys.
    for span in (agent, search, llm):
        assert attrs(span)[f"{ASSOCIATION_PROPERTIES}.session_id"] == "s-1"
        assert attrs(span)[f"{ASSOCIATION_PROPERTIES}.user_id"] == "u-1"
        assert attrs(span)[f"{ASSOCIATION_PROPERTIES}.tags"] == ("t1",)
    metadata = {
        k.removeprefix(f"{ASSOCIATION_PROPERTIES}.metadata."): v
        for k, v in attrs(agent).items()
        if k.startswith(f"{ASSOCIATION_PROPERTIES}.metadata.")
    }
    assert metadata == {"env": "test"}
    assert not any(k.startswith(f"{ASSOCIATION_PROPERTIES}.metadata.") for k in attrs(llm))


def test_error(laminar: Laminar) -> None:
    span = replay(laminar, "traceable_error")["boom"][0]
    [event] = span.events
    assert event.name == "exception"
    assert dict(event.attributes or {})["exception.type"] == "ValueError"
    assert dict(event.attributes or {})["exception.message"] == "bad input"
    assert "Traceback" in str(dict(event.attributes or {})["exception.stacktrace"])


def test_wrap_openai_tool_loop(laminar: Laminar) -> None:
    first, second = replay(laminar, "wrap_openai_tools")["ChatOpenAI"]
    a = attrs(first)
    assert a["gen_ai.request.model"] == "gpt-4o-mini"
    assert a["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert a["gen_ai.request.temperature"] == 0.2
    assert a["gen_ai.usage.input_tokens"] == 20
    assert a["gen_ai.usage.cache_read.input_tokens"] == 4
    assert a["gen_ai.usage.reasoning.output_tokens"] == 2
    tools = json.loads(str(a["gen_ai.tool.definitions"]))
    assert tools[0]["function"]["name"] == "get_weather"
    output = messages(first, "gen_ai.output.messages")
    assert output == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                }
            ],
        }
    ]
    # The tool-call turn is identical in call 2's history, so Laminar's
    # transcript dedups it.
    history = messages(second, "gen_ai.input.messages")
    assert history[1] == output[0]
    assert history[2] == {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}


def test_wrap_openai_stream(laminar: Laminar) -> None:
    span = replay(laminar, "wrap_openai_stream")["ChatOpenAI"][0]
    assert messages(span, "gen_ai.output.messages") == [{"role": "assistant", "content": "Hello"}]
    assert attrs(span)["gen_ai.usage.input_tokens"] == 7
    assert attrs(span)["gen_ai.usage.output_tokens"] == 2


def test_wrap_openai_responses(laminar: Laminar) -> None:
    span = replay(laminar, "wrap_openai_responses")["ChatOpenAI"][0]
    assert messages(span, "gen_ai.input.messages") == [
        {"role": "system", "content": "Be nice"},
        {"role": "user", "content": "Hi"},
    ]
    assert messages(span, "gen_ai.output.messages") == [{"role": "assistant", "content": "Hi!"}]


def test_wrap_anthropic_tool_loop(laminar: Laminar) -> None:
    first, second = replay(laminar, "wrap_anthropic_tools")["ChatAnthropic"]
    a = attrs(first)
    assert a["gen_ai.system"] == "anthropic"
    assert a["gen_ai.request.max_tokens"] == 100
    # LangSmith's input tokens already include cache reads and writes.
    assert a["gen_ai.usage.input_tokens"] == 37
    assert a["gen_ai.usage.cache_read.input_tokens"] == 5
    assert a["gen_ai.usage.cache_write.input_tokens"] == 2
    assert json.loads(str(a["gen_ai.tool.definitions"]))[0]["name"] == "get_weather"
    assert messages(first, "gen_ai.input.messages")[0] == {"role": "system", "content": "Be brief"}
    output = messages(first, "gen_ai.output.messages")
    assert output[0]["content"][1] == {
        "type": "tool_call",
        "id": "toolu_1",
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
    }
    history = messages(second, "gen_ai.input.messages")
    assert history[-1] == {"role": "tool", "content": "sunny", "tool_call_id": "toolu_1"}


def test_langchain(laminar: Laminar) -> None:
    spans = replay(laminar, "langchain_openai_tools")
    # The nested unnamed lambda (`RunnableLambda`) is glue and gets no span;
    # the root `RunnableSequence` stays, as in the LangChain instrumentation.
    assert "RunnableLambda" not in spans
    root = spans["RunnableSequence"][0]
    agent = spans["weather_agent"][0]
    assert parent_id(agent) == ids(root).span_id
    first, second = spans["ChatOpenAI"]
    tool = spans["get_weather"][0]
    assert parent_id(first) == ids(agent).span_id
    assert attrs(first)[SPAN_PATH] == ("RunnableSequence", "weather_agent", "ChatOpenAI")

    # Chain payloads keep only small known keys, so these chains have none.
    assert SPAN_INPUT not in attrs(root) and SPAN_OUTPUT not in attrs(root)
    assert SPAN_INPUT not in attrs(agent)
    # Tool payloads are kept.
    assert json.loads(attrs(tool)[SPAN_INPUT]) == {"city": "Paris"}

    a = attrs(first)
    assert a[SPAN_TYPE] == "LLM"
    assert a["gen_ai.system"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o-mini"
    assert a["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert a["gen_ai.request.temperature"] == 0.1
    assert a["gen_ai.usage.input_tokens"] == 20
    assert a["gen_ai.usage.cache_read.input_tokens"] == 4
    assert json.loads(str(a["gen_ai.tool.definitions"]))[0]["function"]["name"] == "get_weather"
    assert messages(first, "gen_ai.input.messages") == [
        {"role": "user", "content": "Weather in Paris?"}
    ]
    output = messages(first, "gen_ai.output.messages")
    assert output == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"city": "Paris"},
                }
            ],
        }
    ]
    history = messages(second, "gen_ai.input.messages")
    assert history[1] == output[0]
    assert history[2] == {"role": "tool", "content": "sunny", "tool_call_id": "call_1"}

    # LangChain's thread id is the session; step tags are dropped.
    assert a[f"{ASSOCIATION_PROPERTIES}.session_id"] == "th-1"
    assert attrs(agent)[f"{ASSOCIATION_PROPERTIES}.tags"] == ("lc-tag",)
    # The root's metadata is only internal keys (and the thread id).
    assert not any(k.startswith(f"{ASSOCIATION_PROPERTIES}.metadata.") for k in attrs(root))


def test_langgraph(laminar: Laminar) -> None:
    spans = replay(laminar, "langgraph")
    graph, plan, act = spans["LangGraph"][0], spans["plan"][0], spans["act"][0]
    assert parent_id(plan) == ids(graph).span_id
    assert parent_id(act) == ids(graph).span_id
    # Graph state: only the small keys stay, `messages` is dropped.
    assert json.loads(attrs(plan)[SPAN_INPUT]) == {"steps_taken": 0}
    assert json.loads(attrs(act)[SPAN_OUTPUT]) == {"steps_taken": 2}
    for span in (graph, plan, act):
        assert attrs(span)[f"{ASSOCIATION_PROPERTIES}.session_id"] == "thread-42"
        assert f"{ASSOCIATION_PROPERTIES}.tags" not in attrs(span)


@pytest.mark.parametrize("scenario", sorted(FIXTURES))
def test_every_run_becomes_one_span(laminar: Laminar, scenario: str) -> None:
    spans = replay(laminar, scenario)
    created = {r["run"]["id"] for r in FIXTURES[scenario] if r["op"] == "create"}
    glue = {
        r["run"]["id"]
        for r in FIXTURES[scenario]
        if r["op"] == "create" and r["run"]["name"] == "RunnableLambda"
    }
    emitted = [attrs(s)["langsmith.run.id"] for group in spans.values() for s in group]
    assert sorted(emitted) == sorted(created - glue)

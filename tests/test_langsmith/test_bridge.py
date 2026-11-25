"""The LangSmith bridge against a real LangSmith client.

Each test gets its own Laminar span processor (`laminar`), the installed
bridge (`bridge`) and a LangSmith client whose HTTP traffic is recorded
instead of sent (`langsmith`).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from langsmith import Client, traceable, tracing_context
from langsmith.run_trees import RunTree
from opentelemetry import trace
from opentelemetry.context import set_value

from lmnr.opentelemetry_lib.bridges.langsmith import LangSmithBridge
from lmnr.opentelemetry_lib.bridges.langsmith.runs import LangSmithRunRecorder, span_id_for
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    LANGGRAPH_TOPOLOGY,
)
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_IDS_PATH,
    SPAN_PATH,
)
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_SESSION_ID_KEY,
    attach_context,
    detach_context,
    get_current_context,
)

from .conftest import Laminar, LangSmith, ids, parent_id


def _scenario(client: Client) -> None:
    @traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model_name": "gpt-4o"})
    def llm(messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage_metadata": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }

    @traceable(run_type="tool")
    def lookup(x: int) -> dict[str, int]:
        return {"result": x}

    @traceable
    def fail() -> None:
        raise ValueError("kaboom")

    @traceable(name="agent", tags=["t1"], metadata={"session_id": "s-1"})
    def agent(question: str) -> dict[str, Any]:
        lookup(1)
        try:
            fail()
        except ValueError:
            pass
        return llm([{"role": "user", "content": question}])

    with tracing_context(enabled=True, client=client):
        _ = agent("hello")


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIME = re.compile(r"\d{8}T\d{6}\d*Z|\d{4}-\d\d-\d\dT[\d:.]+(\+00:00|Z)?")


def _normalized(bodies: list[tuple[str, str, Any]]) -> str:
    """Run ids and times differ between two runs; everything else must not."""
    text = json.dumps(bodies, sort_keys=True)
    ids: dict[str, str] = {}
    text = _UUID.sub(lambda m: ids.setdefault(m.group(0), f"<id{len(ids)}>"), text)
    text = re.sub(r"\\n  File [^\\]*", "<frame>", text)
    return _TIME.sub("<time>", text)


def test_langsmith_traffic_is_unchanged_by_the_bridge(laminar: Laminar) -> None:
    without = LangSmith()
    _scenario(without.client)

    bridge = LangSmithBridge()
    assert bridge.install(laminar.context)
    try:
        with_bridge = LangSmith()
        _scenario(with_bridge.client)
    finally:
        bridge.uninstall()

    assert laminar.spans(), "the bridge recorded nothing"
    assert _normalized(with_bridge.bodies()) == _normalized(without.bodies())


def test_runs_become_spans(laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith) -> None:
    _scenario(langsmith.client)
    agent, lookup, fail, llm = (laminar.span(n) for n in ("agent", "lookup", "fail", "llm"))
    for child in (lookup, fail, llm):
        assert parent_id(child) == ids(agent).span_id
        assert ids(child).trace_id == ids(agent).trace_id
    assert dict(llm.attributes or {})["lmnr.span.type"] == "LLM"
    assert dict(lookup.attributes or {})["lmnr.span.type"] == "TOOL"
    assert dict(llm.attributes or {})["gen_ai.usage.input_tokens"] == 3
    assert dict(llm.attributes or {})[f"{ASSOCIATION_PROPERTIES}.session_id"] == "s-1"
    assert [e.name for e in fail.events] == ["exception"]
    assert dict(fail.events[0].attributes or {})["exception.message"] == "kaboom"
    # The span times are LangSmith's run times.
    assert agent.start_time is not None and llm.start_time is not None
    assert agent.start_time <= llm.start_time


def test_nests_under_the_active_laminar_span(
    laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith
) -> None:
    tracer = laminar.provider.get_tracer("lmnr.tracer")
    with tracer.start_as_current_span("outer") as outer:
        context = trace.set_span_in_context(outer, get_current_context())
        token = attach_context(set_value(CONTEXT_SESSION_ID_KEY, "laminar-session", context))
        try:
            _scenario(langsmith.client)
        finally:
            detach_context(token)
    outer_span, agent, llm = laminar.span("outer"), laminar.span("agent"), laminar.span("llm")
    assert parent_id(agent) == ids(outer_span).span_id
    assert ids(agent).trace_id == ids(outer_span).trace_id
    assert ids(llm).trace_id == ids(outer_span).trace_id
    assert dict(llm.attributes or {})[SPAN_PATH] == ("outer", "agent", "llm")
    assert dict(llm.attributes or {})[SPAN_IDS_PATH] == tuple(
        str(uuid.UUID(int=ids(s).span_id)) for s in (outer_span, agent, llm)
    )
    # Laminar's own session wins over the run's metadata.
    assert dict(agent.attributes or {})[f"{ASSOCIATION_PROPERTIES}.session_id"] == (
        "laminar-session"
    )


def test_redaction_reaches_laminar(laminar: Laminar, bridge: LangSmithBridge) -> None:
    langsmith = LangSmith(hide_inputs=True, hide_outputs=lambda outputs: {"redacted": True})
    _scenario(langsmith.client)
    llm = dict(laminar.span("llm").attributes or {})
    assert "gen_ai.input.messages" not in llm
    assert json.loads(str(llm["lmnr.span.output"])) == {"redacted": True}
    assert "lmnr.span.input" not in dict(laminar.span("lookup").attributes or {})


def test_sampled_out_runs_are_not_recorded(laminar: Laminar, bridge: LangSmithBridge) -> None:
    langsmith = LangSmith(tracing_sampling_rate=0.0)
    _scenario(langsmith.client)
    assert laminar.spans() == []
    assert langsmith.bodies() == []


def test_replicas_are_recorded_once(
    laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith
) -> None:
    @traceable
    def child() -> str:
        return "x"

    @traceable(name="root")
    def root() -> str:
        return child()

    with tracing_context(
        enabled=True,
        client=langsmith.client,
        replicas=[{"project_name": "first"}, {"project_name": "second"}],
    ):
        root()
    # LangSmith wrote each run twice, with ids remapped per project.
    assert len([b for b in langsmith.bodies() if b[0] == "POST"]) == 4
    root_span, child_span = laminar.span("root"), laminar.span("child")
    assert parent_id(child_span) == ids(root_span).span_id


def test_primary_replica_keeps_the_run_ids(
    laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith
) -> None:
    run_id = uuid.uuid4()

    @traceable(name="root")
    def root() -> str:
        return "x"

    with tracing_context(
        enabled=True,
        client=langsmith.client,
        replicas=[{"project_name": "other"}, {"project_name": "main", "primary": True}],
    ):
        root(langsmith_extra={"run_id": run_id})
    span = laminar.span("root")
    assert ids(span).span_id == span_id_for(str(run_id))


def test_complete_run_tree_is_recorded_once(
    laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith
) -> None:
    run = RunTree(name="manual", run_type="chain", inputs={"q": 1}, ls_client=langsmith.client)
    run.end(outputs={"a": 2})
    run.post()
    run.patch()
    span = laminar.span("manual")
    assert json.loads(str(dict(span.attributes or {})["lmnr.span.output"])) == {"a": 2}


def test_langchain_runs(laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith) -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tracers.context import tracing_v2_enabled

    model = GenericFakeChatModel(messages=iter([AIMessage("pong")]))
    chain = RunnableLambda(lambda x: x, name="prepare") | model

    with tracing_v2_enabled(client=langsmith.client):
        chain.invoke("ping", config={"metadata": {"thread_id": "t-9"}})

    root = laminar.span("RunnableSequence")
    chat = laminar.span("GenericFakeChatModel")
    assert parent_id(chat) == ids(root).span_id
    a = dict(chat.attributes or {})
    assert a["lmnr.span.type"] == "LLM"
    assert json.loads(str(a["gen_ai.input.messages"])) == [{"role": "user", "content": "ping"}]
    assert json.loads(str(a["gen_ai.output.messages"])) == [
        {"role": "assistant", "content": "pong"}
    ]
    assert a[f"{ASSOCIATION_PROPERTIES}.session_id"] == "t-9"


def test_langgraph_topology_lands_on_the_graph_run(
    laminar: Laminar,
    bridge: LangSmithBridge,
    langsmith: LangSmith,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import TypedDict

    from langchain_core.tracers.context import tracing_v2_enabled
    from langgraph.graph import END, START, StateGraph

    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langchain.callback_handler import (
        LaminarCallbackHandler,
    )
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langgraph import _topology

    # The test session's Laminar has the LangChain instrumentation on, whose
    # handler would consume the topology first; with the bridge it is off.
    monkeypatch.setattr(LaminarCallbackHandler, "on_chain_start", lambda *a, **k: None)

    class State(TypedDict):
        steps_taken: int

    def step(state: State) -> State:
        return {"steps_taken": state["steps_taken"] + 1}

    graph = StateGraph(State)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_edge("step", END)
    app = graph.compile()

    # What Laminar's langgraph instrumentation publishes around each step (it
    # is also on in the test session, which publishes the same).
    topology = _topology(app.get_graph())

    def steps() -> Iterator[Any]:
        iterator = iter(app.stream({"steps_taken": 0}))
        while True:
            token = LANGGRAPH_TOPOLOGY.set(dict(topology))
            try:
                yield next(iterator)
            except StopIteration:
                return
            finally:
                LANGGRAPH_TOPOLOGY.reset(token)

    with tracing_v2_enabled(client=langsmith.client):
        _ = list(steps())
    a = dict(laminar.span("LangGraph").attributes or {})
    assert a[f"{ASSOCIATION_PROPERTIES}.langgraph.nodes"] == topology["langgraph.nodes"]
    assert '"step"' in topology["langgraph.nodes"]
    assert f"{ASSOCIATION_PROPERTIES}.langgraph.nodes" not in dict(
        laminar.span("step").attributes or {}
    )


def test_a_failing_bridge_never_breaks_langsmith(
    laminar: Laminar, bridge: LangSmithBridge, langsmith: LangSmith, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("bridge bug")

    monkeypatch.setattr(LangSmithRunRecorder, "on_create", boom)
    monkeypatch.setattr(LangSmithRunRecorder, "on_update", boom)
    _scenario(langsmith.client)
    assert len(langsmith.bodies()) == 8


def test_uninstall_restores_langsmith(laminar: Laminar) -> None:
    originals = (Client._create_run, Client._update_run, RunTree.post, RunTree.patch)
    bridge = LangSmithBridge()
    assert bridge.install(laminar.context)
    assert Client._create_run is not originals[0]
    bridge.uninstall()
    assert (Client._create_run, Client._update_run, RunTree.post, RunTree.patch) == originals

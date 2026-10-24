import json
from collections.abc import Iterator
from typing import Annotated, Any, TypedDict, cast

import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from lmnr import Laminar, observe
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langchain import callback_handler
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    LANGGRAPH_TOPOLOGY,
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState) -> AgentState:
    return {"messages": [AIMessage(content="Hello, how can I help you?")]}


def get_current_weather(location: str) -> str:
    return f"The weather in {location} is sunny."


tools = [
    Tool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        func=get_current_weather,
    )
]

EXPECTED_NODES = [
    {"id": "__start__", "name": "__start__", "metadata": None},
    {"id": "assistant", "name": "assistant", "metadata": None},
    {"id": "tools", "name": "tools", "metadata": None},
    {"id": "__end__", "name": "__end__", "metadata": None},
]
EXPECTED_EDGES = [
    {"source": "__start__", "target": "assistant", "conditional": False},
    {"source": "assistant", "target": "tools", "conditional": True},
    {"source": "assistant", "target": "__end__", "conditional": True},
    {"source": "tools", "target": "assistant", "conditional": False},
]
NODES_KEY = "lmnr.association.properties.langgraph.nodes"
EDGES_KEY = "lmnr.association.properties.langgraph.edges"


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _span(spans: tuple[ReadableSpan, ...] | list[ReadableSpan], name: str) -> ReadableSpan:
    return next(span for span in spans if span.name == name)


def _span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def _parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent is not None else None


def _build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    graph_builder = StateGraph(AgentState)
    _ = graph_builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_edge(START, "assistant")
    _ = graph_builder.add_conditional_edges("assistant", tools_condition)
    _ = graph_builder.add_edge("tools", "assistant")
    return graph_builder.compile()  # pyright: ignore[reportUnknownMemberType]


def _assert_graph_spans(span_exporter: InMemorySpanExporter) -> None:
    spans = span_exporter.get_finished_spans()
    # The graph run, the `assistant` node and the `tools_condition` branch
    assert len(spans) == 3
    workflow_span = _span(spans, "LangGraph.workflow")
    assistant_span = _span(spans, "execute_task assistant")
    condition_span = _span(spans, "execute_task tools_condition")

    assert workflow_span.parent is None
    assert _parent_id(assistant_span) == _span_id(workflow_span)
    assert _parent_id(condition_span) == _span_id(assistant_span)
    assert len({_span_id(span) for span in spans}) == 3
    assert len({cast(Any, span.context).trace_id for span in spans}) == 1

    workflow_attrs = _attrs(workflow_span)
    assert workflow_attrs["lmnr.span.type"] == "DEFAULT"
    assert workflow_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert workflow_attrs["gen_ai.agent.name"] == "LangGraph"
    assert workflow_attrs["lmnr.span.instrumentation_scope.name"] == "langchain"
    assert workflow_attrs["lmnr.span.path"] == ("LangGraph.workflow",)
    assert json.loads(cast(str, workflow_attrs["lmnr.span.input"])) == {
        "messages": [
            {
                "content": "What is the weather in Tokyo?",
                "additional_kwargs": {},
                "response_metadata": {},
                "type": "human",
                "name": None,
                "id": None,
            }
        ]
    }
    output = json.loads(cast(str, workflow_attrs["lmnr.span.output"]))
    assert [m["content"] for m in output["messages"]] == [
        "What is the weather in Tokyo?",
        "Hello, how can I help you?",
    ]
    assert json.loads(cast(str, workflow_attrs[NODES_KEY])) == EXPECTED_NODES
    workflow_edges = json.loads(cast(str, workflow_attrs[EDGES_KEY]))
    assert len(workflow_edges) == 4
    assert all(edge in workflow_edges for edge in EXPECTED_EDGES)

    assert _attrs(assistant_span)["gen_ai.operation.name"] == "execute_task"
    assert _attrs(assistant_span)["lmnr.span.path"] == (
        "LangGraph.workflow",
        "execute_task assistant",
    )
    assert json.loads(cast(str, _attrs(condition_span)["lmnr.span.output"])) == "__end__"

    for span in (assistant_span, condition_span):
        assert NODES_KEY not in _attrs(span)
        assert EDGES_KEY not in _attrs(span)

    # The topology is only published while the graph runs
    assert LANGGRAPH_TOPOLOGY.get() is None


def test_langchain_langgraph(span_exporter: InMemorySpanExporter):
    graph = _build_graph()
    _res = graph.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})  # pyright: ignore[reportUnknownMemberType]
    _assert_graph_spans(span_exporter)


@pytest.mark.asyncio
async def test_langchain_langgraph_async(span_exporter: InMemorySpanExporter):
    graph = _build_graph()
    _res = await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    _assert_graph_spans(span_exporter)


def test_langchain_langgraph_stream(span_exporter: InMemorySpanExporter):
    graph = _build_graph()
    for _chunk in graph.stream(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    ):
        # Caller code between steps does not see the topology
        assert LANGGRAPH_TOPOLOGY.get() is None
    _assert_graph_spans(span_exporter)


@pytest.mark.asyncio
async def test_langchain_langgraph_async_stream(span_exporter: InMemorySpanExporter):
    graph = _build_graph()
    async for _chunk in graph.astream(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    ):
        assert LANGGRAPH_TOPOLOGY.get() is None
    _assert_graph_spans(span_exporter)


def test_langgraph_inside_observe(span_exporter: InMemorySpanExporter):
    graph = _build_graph()

    @observe(name="outer")
    def run() -> None:
        _ = graph.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})  # pyright: ignore[reportUnknownMemberType]

    run()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    outer = _span(spans, "outer")
    workflow_span = _span(spans, "LangGraph.workflow")
    assert _parent_id(workflow_span) == _span_id(outer)
    assert _attrs(workflow_span)["lmnr.span.path"] == ("outer", "LangGraph.workflow")
    assert json.loads(cast(str, _attrs(workflow_span)[NODES_KEY])) == EXPECTED_NODES


def test_nested_graph_topology_only_on_root(span_exporter: InMemorySpanExporter):
    inner = _build_graph()

    def call_inner(state: AgentState) -> AgentState:
        return cast(AgentState, inner.invoke(state))  # pyright: ignore[reportUnknownMemberType]

    outer_builder = StateGraph(AgentState)
    _ = outer_builder.add_node("call_inner", call_inner)  # pyright: ignore[reportUnknownMemberType]
    _ = outer_builder.add_edge(START, "call_inner")
    outer = outer_builder.compile()  # pyright: ignore[reportUnknownMemberType]

    _ = outer.invoke({"messages": [HumanMessage(content="hi")]})  # pyright: ignore[reportUnknownMemberType]

    spans = span_exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert len(roots) == 1
    root_nodes = json.loads(cast(str, _attrs(roots[0])[NODES_KEY]))
    assert [node["id"] for node in root_nodes] == ["__start__", "call_inner", "__end__"]
    for span in spans:
        if span is not roots[0]:
            assert NODES_KEY not in _attrs(span)


def test_chain(span_exporter: InMemorySpanExporter):
    def increment(x: int) -> int:
        return x + 1

    def times_two(x: int) -> int:
        return x * 2

    chain = RunnableLambda(increment).with_config(run_name="add_one") | RunnableLambda(
        times_two
    ).with_config(run_name="double")

    assert chain.invoke(1) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    root = _span(spans, "RunnableSequence.workflow")
    add_one = _span(spans, "execute_task add_one")
    double = _span(spans, "execute_task double")
    assert root.parent is None
    assert _parent_id(add_one) == _span_id(root)
    assert _parent_id(double) == _span_id(root)
    assert json.loads(cast(str, _attrs(root)["lmnr.span.input"])) == 1
    assert json.loads(cast(str, _attrs(root)["lmnr.span.output"])) == 4
    assert json.loads(cast(str, _attrs(double)["lmnr.span.input"])) == 2
    assert _attrs(double)["langchain.tags"] == ("seq:step:2",)


def test_chain_error(span_exporter: InMemorySpanExporter):
    def fail(_: int) -> int:
        raise ValueError("boom")

    chain = RunnableLambda(fail).with_config(run_name="fail")
    with pytest.raises(ValueError, match="boom"):
        _ = chain.invoke(1)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "fail.workflow"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    assert _attrs(span)["error.type"] == "ValueError"
    assert [event.name for event in span.events] == ["exception"]
    # The failed run's context does not leak into the caller
    assert context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None
    assert Laminar.get_current_span() is None or not Laminar.get_current_span().is_recording()  # pyright: ignore[reportOptionalMemberAccess]


def _fake_chat_model(*responses: AIMessage) -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=list(responses))


def _assert_chat_span(span: ReadableSpan) -> None:
    attrs = _attrs(span)
    assert span.name == "FakeMessagesListChatModel.chat"
    assert attrs["lmnr.span.type"] == "LLM"
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "langchain"
    assert json.loads(cast(str, attrs["gen_ai.input.messages"])) == [
        {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
        {"role": "user", "parts": [{"type": "text", "content": "What is 2+2?"}]},
    ]
    assert json.loads(cast(str, attrs["gen_ai.output.messages"])) == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "4"},
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "calculator",
                    "arguments": {"expression": "2+2"},
                },
            ],
            "finish_reason": "",
        }
    ]
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 3
    assert attrs["llm.usage.total_tokens"] == 13
    assert attrs["gen_ai.usage.cache_read_input_tokens"] == 4
    assert attrs["gen_ai.response.model"] == "fake-model-1"
    assert attrs["gen_ai.request.model"] == "fake-model-1"
    assert attrs["gen_ai.response.id"] == "msg_1"


def _chat_response() -> AIMessage:
    return AIMessage(
        content="4",
        id="msg_1",
        tool_calls=[{"id": "call_1", "name": "calculator", "args": {"expression": "2+2"}}],
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "input_token_details": {"cache_read": 4},
        },
        response_metadata={"model_name": "fake-model-1"},
    )


CHAT_INPUT = [SystemMessage(content="Be brief."), HumanMessage(content="What is 2+2?")]


def test_chat_model(span_exporter: InMemorySpanExporter):
    model = _fake_chat_model(_chat_response())
    _ = model.invoke(CHAT_INPUT)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].parent is None
    _assert_chat_span(spans[0])
    assert context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_chat_model_async(span_exporter: InMemorySpanExporter):
    model = _fake_chat_model(_chat_response())
    _ = await model.ainvoke(CHAT_INPUT)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_chat_span(spans[0])
    # `agenerate` ends the run from another task; the suppression it set
    # for the provider SDK must not keep suppressing the caller afterwards
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)

    with Laminar.start_as_current_span("after"):
        pass
    assert len(span_exporter.get_finished_spans()) == 2


@pytest.mark.asyncio
async def test_chat_model_async_stream(span_exporter: InMemorySpanExporter):
    model = _fake_chat_model(AIMessage(content="four"))
    chunks = [chunk async for chunk in model.astream(CHAT_INPUT)]
    assert "".join(cast(str, chunk.content) for chunk in chunks) == "four"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "FakeMessagesListChatModel.chat"
    output = json.loads(cast(str, _attrs(spans[0])["gen_ai.output.messages"]))
    assert output[0]["parts"] == [{"type": "text", "content": "four"}]
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)

    with Laminar.start_as_current_span("after"):
        pass
    assert len(span_exporter.get_finished_spans()) == 2


def test_chat_model_suppresses_provider_instrumentation(span_exporter: InMemorySpanExporter):
    seen: list[tuple[object, bool]] = []

    class ProbeModel(FakeMessagesListChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            flag = context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
            seen.append((flag, bool(flag)))
            return super()._generate(*args, **kwargs)

    _ = ProbeModel(responses=[AIMessage(content="ok")]).invoke("hi")

    assert len(seen) == 1
    flag, active_during_run = seen[0]
    assert active_during_run
    assert not flag  # off once the run has ended, wherever it is still attached
    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["ProbeModel.chat"]


def test_runs_inside_llm_run_are_recorded(span_exporter: InMemorySpanExporter):
    suppressed_in_tool: list[bool] = []

    @tool
    def lookup(query: str) -> str:
        """Look something up."""
        suppressed_in_tool.append(bool(context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)))
        return query

    class ToolUsingModel(FakeMessagesListChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            _ = lookup.invoke({"query": "x"})
            return super()._generate(*args, **kwargs)

    _ = ToolUsingModel(responses=[AIMessage(content="ok")]).invoke("hi")

    spans = span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == ["ToolUsingModel.chat", "execute_tool lookup"]
    # The flag only silences the provider call of the LLM run itself, not
    # provider calls made by a tool running under it
    assert suppressed_in_tool == [False]


def test_runs_while_streaming_are_recorded(span_exporter: InMemorySpanExporter):
    echo = Tool(name="echo", func=lambda x: x, description="Echo.")
    model = GenericFakeChatModel(messages=iter([AIMessage(content="one two")]))

    for chunk in model.stream("hi"):
        # The LLM run is still open here, its flag attached in this context
        _ = echo.invoke(cast(str, chunk.content))

    spans = span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == [
        "GenericFakeChatModel.chat",
        "execute_tool echo",
        "execute_tool echo",
        "execute_tool echo",
    ]
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)


def test_chat_model_error(span_exporter: InMemorySpanExporter):
    class FailingModel(FakeMessagesListChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        _ = FailingModel(responses=[]).invoke("hi")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert _attrs(spans[0])["error.type"] == "RuntimeError"
    assert context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None


@pytest.mark.parametrize(
    "run",
    [
        lambda: _fake_chat_model(_chat_response()).invoke(CHAT_INPUT),
        lambda: RunnableLambda(lambda x: x).invoke("hi"),
        lambda: Tool(name="echo", func=lambda x: x, description="Echo.").invoke("hi"),
        lambda: _StaticRetriever().invoke("cats"),
    ],
    ids=["llm", "chain", "tool", "retriever"],
)
def test_run_ends_when_recording_output_fails(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, run: Any
):
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("cannot record")

    monkeypatch.setattr(callback_handler, "set_response_attributes", fail)
    monkeypatch.setattr(callback_handler.LaminarCallbackHandler, "_set_output", fail)
    _ = run()

    # The run still ends, and an LLM run's suppression with it
    assert len(span_exporter.get_finished_spans()) == 1
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
    with Laminar.start_as_current_span("after"):
        pass
    assert len(span_exporter.get_finished_spans()) == 2


def test_tool(span_exporter: InMemorySpanExporter):
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        with Laminar.start_as_current_span("inside_tool"):
            pass
        return a * b

    assert multiply.invoke({"a": 3, "b": 4}) == 12

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    tool_span = _span(spans, "execute_tool multiply")
    inner = _span(spans, "inside_tool")
    attrs = _attrs(tool_span)
    assert attrs["lmnr.span.type"] == "TOOL"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "multiply"
    assert attrs["gen_ai.tool.description"] == "Multiply two numbers."
    assert json.loads(cast(str, attrs["lmnr.span.input"])) == {"a": 3, "b": 4}
    assert json.loads(cast(str, attrs["lmnr.span.output"])) == 12
    # Spans opened inside the tool nest under it
    assert _parent_id(inner) == _span_id(tool_span)
    assert _attrs(inner)["lmnr.span.path"] == ("execute_tool multiply", "inside_tool")


def test_tool_call_id(span_exporter: InMemorySpanExporter):
    @tool
    def echo(text: str) -> str:
        """Echo the text."""
        return text

    _ = echo.invoke({"type": "tool_call", "id": "call_42", "name": "echo", "args": {"text": "x"}})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert _attrs(spans[0])["gen_ai.tool.call.id"] == "call_42"


class _StaticRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [Document(page_content=f"about {query}", metadata={"source": "kb"})]


def test_retriever(span_exporter: InMemorySpanExporter):
    _ = _StaticRetriever().invoke("cats")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = _attrs(spans[0])
    assert spans[0].name == "vector_db_retrieve _StaticRetriever"
    assert attrs["gen_ai.operation.name"] == "vector_db_retrieve"
    assert json.loads(cast(str, attrs["lmnr.span.input"])) == "cats"
    assert json.loads(cast(str, attrs["lmnr.span.output"])) == [
        {"page_content": "about cats", "metadata": {"source": "kb"}}
    ]


def test_metadata_is_sanitized(span_exporter: InMemorySpanExporter):
    class Secretive:
        def __str__(self) -> str:
            return "api_key=sk-secret"

    chain = RunnableLambda(lambda x: x).with_config(
        run_name="echo", metadata={"user": "u1", "client": Secretive(), "n": 1}
    )
    _ = chain.invoke("hi")

    spans = span_exporter.get_finished_spans()
    assert json.loads(cast(str, _attrs(spans[0])["langchain.metadata"])) == {"user": "u1", "n": 1}
    assert not any(key.startswith("lmnr.association.properties.") for key in _attrs(spans[0]))


def test_thread_id_is_conversation_id(span_exporter: InMemorySpanExporter):
    graph = _build_graph()
    _ = graph.invoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "thread-1"}},
    )

    workflow_span = _span(span_exporter.get_finished_spans(), "LangGraph.workflow")
    assert _attrs(workflow_span)["gen_ai.conversation.id"] == "thread-1"


def _openai_sse(*chunks: dict[str, Any]) -> bytes:
    events = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return "".join([*events, "data: [DONE]\n\n"]).encode()


def _openai_stream_chunk(delta: dict[str, Any], finish_reason: str | None = None):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _openai_transport(captured: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body.get("stream"):
            usage = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_openai_sse(
                    _openai_stream_chunk({"role": "assistant", "content": "Paris"}),
                    _openai_stream_chunk({}, "stop"),
                    {**_openai_stream_chunk({}), "choices": [], "usage": usage},
                ),
            )
        content = json.dumps({"city": "Paris"}) if "response_format" in body else "Paris"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
            },
        )

    return httpx.MockTransport(handler)


def _chat_openai(captured: list[httpx.Request], **kwargs: Any) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=SecretStr("test-api-key"),
        http_client=httpx.Client(transport=_openai_transport(captured)),
        http_async_client=httpx.AsyncClient(transport=_openai_transport(captured)),
        **kwargs,
    )


def _assert_traceparent(request: httpx.Request, span: ReadableSpan) -> None:
    assert span.context is not None
    assert request.headers["traceparent"].startswith(
        f"00-{span.context.trace_id:032x}-{span.context.span_id:016x}-"
    )


def _assert_openai_spans(
    span_exporter: InMemorySpanExporter, captured: list[httpx.Request], streamed: bool = False
):
    spans = span_exporter.get_finished_spans()
    # The OpenAI SDK instrumentation records the call; LangChain opens no
    # span of its own for it
    assert [span.name for span in spans] == ["openai.chat"]
    span = spans[0]
    attrs = _attrs(span)
    assert attrs["lmnr.span.type"] == "LLM"
    assert attrs["lmnr.span.instrumentation_scope.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert attrs["gen_ai.response.id"] == "chatcmpl-1"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 1
    assert span.parent is None
    assert len(captured) == 1
    _assert_traceparent(captured[0], span)
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)


def test_openai_handed_off_to_provider(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    _ = _chat_openai(captured).invoke("What is the capital of France?")
    _assert_openai_spans(span_exporter, captured)


@pytest.mark.asyncio
async def test_openai_handed_off_to_provider_async(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    _ = await _chat_openai(captured).ainvoke("What is the capital of France?")
    _assert_openai_spans(span_exporter, captured)


def test_openai_handed_off_to_provider_stream(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    model = _chat_openai(captured, stream_usage=True)
    chunks = list(model.stream("What is the capital of France?"))
    assert "".join(cast(str, chunk.content) for chunk in chunks) == "Paris"
    _assert_openai_spans(span_exporter, captured, streamed=True)


@pytest.mark.asyncio
async def test_openai_handed_off_to_provider_async_stream(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    model = _chat_openai(captured, stream_usage=True)
    chunks = [chunk async for chunk in model.astream("What is the capital of France?")]
    assert "".join(cast(str, chunk.content) for chunk in chunks) == "Paris"
    _assert_openai_spans(span_exporter, captured, streamed=True)


def test_openai_handed_off_span_nests_under_chain(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    chain = RunnableLambda(lambda x: x) | _chat_openai(captured)
    _ = chain.invoke("What is the capital of France?")

    spans = span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == [
        "RunnableSequence.workflow",
        "execute_task RunnableLambda",
        "openai.chat",
    ]
    assert _parent_id(_span(spans, "openai.chat")) == _span_id(
        _span(spans, "RunnableSequence.workflow")
    )


@pytest.fixture
def openai_uninstrumented(monkeypatch: pytest.MonkeyPatch) -> None:
    from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import OpenAIInstrumentor

    monkeypatch.setattr(OpenAIInstrumentor(), "_is_instrumented_by_opentelemetry", False)


def _assert_langchain_openai_span(
    span_exporter: InMemorySpanExporter, captured: list[httpx.Request], streamed: bool = False
):
    spans = span_exporter.get_finished_spans()
    # The OpenAI SDK instrumentation underneath is suppressed
    assert [span.name for span in spans] == ["ChatOpenAI.chat"]
    span = spans[0]
    attrs = _attrs(span)
    assert attrs["lmnr.span.instrumentation_scope.name"] == "langchain"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    if not streamed:
        # Merged stream chunks carry LangChain's own `lc_run--` message id
        assert attrs["gen_ai.response.id"] == "chatcmpl-1"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 1
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert len(captured) == 1
    _assert_traceparent(captured[0], span)
    assert not context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)


@pytest.mark.usefixtures("openai_uninstrumented")
def test_openai_traceparent_header(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    _ = _chat_openai(captured).invoke("What is the capital of France?")
    _assert_langchain_openai_span(span_exporter, captured)


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_uninstrumented")
async def test_openai_traceparent_header_async(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    _ = await _chat_openai(captured).ainvoke("What is the capital of France?")
    _assert_langchain_openai_span(span_exporter, captured)


@pytest.mark.usefixtures("openai_uninstrumented")
def test_openai_traceparent_header_stream(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    model = _chat_openai(captured, stream_usage=True)
    # `stream` calls `_stream` without a `run_manager`
    chunks = list(model.stream("What is the capital of France?"))
    assert "".join(cast(str, chunk.content) for chunk in chunks) == "Paris"
    _assert_langchain_openai_span(span_exporter, captured, streamed=True)


@pytest.mark.asyncio
@pytest.mark.usefixtures("openai_uninstrumented")
async def test_openai_traceparent_header_async_stream(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    model = _chat_openai(captured, stream_usage=True)
    chunks = [chunk async for chunk in model.astream("What is the capital of France?")]
    assert "".join(cast(str, chunk.content) for chunk in chunks) == "Paris"
    _assert_langchain_openai_span(span_exporter, captured, streamed=True)


def test_openai_inside_llm_run_keeps_langchain_span(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    inner = _chat_openai(captured)

    class DelegatingModel(FakeMessagesListChatModel):
        def _generate(self, *args: Any, **kwargs: Any) -> Any:
            _ = inner.invoke("What is the capital of France?")
            return super()._generate(*args, **kwargs)

    _ = DelegatingModel(responses=[AIMessage(content="ok")]).invoke("hi")

    # The outer LLM run suppresses the provider instrumentation, so the inner
    # run cannot be handed off to it
    spans = span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == ["ChatOpenAI.chat", "DelegatingModel.chat"]


class _City(TypedDict):
    city: str


def test_openai_structured_output_keeps_langchain_span(span_exporter: InMemorySpanExporter):
    captured: list[httpx.Request] = []
    model = _chat_openai(captured).with_structured_output(_City, method="json_schema")
    assert model.invoke("What is the capital of France?") == {"city": "Paris"}

    # It may go through `responses.parse`, which the OpenAI instrumentation
    # does not record, so the LangChain span stays
    spans = span_exporter.get_finished_spans()
    assert "ChatOpenAI.chat" in [span.name for span in spans]
    assert "openai.chat" not in [span.name for span in spans]


def test_agent_loop_with_tools(span_exporter: InMemorySpanExporter):
    responses: Iterator[AIMessage] = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "get_current_weather", "args": {"__arg1": "Tokyo"}}
                ],
            ),
            AIMessage(content="It is sunny in Tokyo."),
        ]
    )
    model = _fake_chat_model(*responses)

    def call_model(state: AgentState) -> AgentState:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    _ = builder.add_node("assistant", call_model)  # pyright: ignore[reportUnknownMemberType]
    _ = builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = builder.add_edge(START, "assistant")
    _ = builder.add_conditional_edges("assistant", tools_condition)
    _ = builder.add_edge("tools", "assistant")
    graph = builder.compile()  # pyright: ignore[reportUnknownMemberType]

    _ = graph.invoke({"messages": [HumanMessage(content="Weather in Tokyo?")]})  # pyright: ignore[reportUnknownMemberType]

    spans = span_exporter.get_finished_spans()
    root = _span(spans, "LangGraph.workflow")
    by_id = {_span_id(span): span for span in spans}
    llm_spans = [span for span in spans if span.name == "FakeMessagesListChatModel.chat"]
    tool_span = _span(spans, "execute_tool get_current_weather")
    assert len(llm_spans) == 2
    assert _attrs(tool_span)["gen_ai.tool.call.id"] == "call_1"
    assert json.loads(cast(str, _attrs(tool_span)["lmnr.span.output"]))["content"] == (
        "The weather in Tokyo is sunny."
    )
    # Everything hangs off the one graph root
    assert [span for span in spans if span.parent is None] == [root]
    for span in [*llm_spans, tool_span]:
        ancestor = span
        while ancestor.parent is not None:
            ancestor = by_id[ancestor.parent.span_id]
        assert ancestor is root
    assert context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) is None

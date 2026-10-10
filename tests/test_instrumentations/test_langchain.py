import json
from typing import Annotated, TypedDict, cast

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import Tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


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


def test_langchain_langgraph(span_exporter: InMemorySpanExporter):
    graph_builder = StateGraph(AgentState)
    _ = graph_builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_edge(START, "assistant")
    _ = graph_builder.add_conditional_edges("assistant", tools_condition)
    _ = graph_builder.add_edge("tools", "assistant")
    graph = graph_builder.compile()  # pyright: ignore[reportUnknownMemberType]

    _res = graph.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})  # pyright: ignore[reportUnknownMemberType]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = next(span for span in spans if span.name == "LangGraph.workflow")
    other_spans = [span for span in spans if span.name != "LangGraph.workflow"]
    assert json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.nodes"])
    ) == [
        {
            "id": "__start__",
            "name": "__start__",
            "metadata": None,
        },
        {
            "id": "assistant",
            "name": "assistant",
            "metadata": None,
        },
        {
            "id": "tools",
            "name": "tools",
            "metadata": None,
        },
        {
            "id": "__end__",
            "name": "__end__",
            "metadata": None,
        },
    ]

    workflow_edges = json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.edges"])
    )
    assert all(
        edge in workflow_edges
        for edge in [
            {"source": "__start__", "target": "assistant", "conditional": False},
            {"source": "assistant", "target": "tools", "conditional": True},
            {"source": "assistant", "target": "__end__", "conditional": True},
            {"source": "tools", "target": "assistant", "conditional": False},
        ]
    )
    assert len(workflow_edges) == 4

    for other_span in other_spans:
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.nodes")
            is None
        )
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.edges")
            is None
        )


@pytest.mark.asyncio
async def test_langchain_langgraph_async(span_exporter: InMemorySpanExporter):
    graph_builder = StateGraph(AgentState)
    _ = graph_builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_edge(START, "assistant")
    _ = graph_builder.add_conditional_edges("assistant", tools_condition)
    _ = graph_builder.add_edge("tools", "assistant")
    graph = graph_builder.compile()  # pyright: ignore[reportUnknownMemberType]

    _res = await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = next(span for span in spans if span.name == "LangGraph.workflow")
    other_spans = [span for span in spans if span.name != "LangGraph.workflow"]
    assert json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.nodes"]
    )) == [
        {
            "id": "__start__",
            "name": "__start__",
            "metadata": None,
        },
        {
            "id": "assistant",
            "name": "assistant",
            "metadata": None,
        },
        {
            "id": "tools",
            "name": "tools",
            "metadata": None,
        },
        {
            "id": "__end__",
            "name": "__end__",
            "metadata": None,
        },
    ]

    workflow_edges = json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.edges"])
    )
    assert all(
        edge in workflow_edges
        for edge in [
            {"source": "__start__", "target": "assistant", "conditional": False},
            {"source": "assistant", "target": "tools", "conditional": True},
            {"source": "assistant", "target": "__end__", "conditional": True},
            {"source": "tools", "target": "assistant", "conditional": False},
        ]
    )
    assert len(workflow_edges) == 4

    for other_span in other_spans:
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.nodes")
            is None
        )
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.edges")
            is None
        )


def test_langchain_langgraph_stream(span_exporter: InMemorySpanExporter):
    graph_builder = StateGraph(AgentState)
    _ = graph_builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_edge(START, "assistant")
    _ = graph_builder.add_conditional_edges("assistant", tools_condition)
    _ = graph_builder.add_edge("tools", "assistant")
    graph = graph_builder.compile()  # pyright: ignore[reportUnknownMemberType]

    for _chunk in graph.stream(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = next(span for span in spans if span.name == "LangGraph.workflow")
    other_spans = [span for span in spans if span.name != "LangGraph.workflow"]
    assert json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.nodes"])
    ) == [
        {
            "id": "__start__",
            "name": "__start__",
            "metadata": None,
        },
        {
            "id": "assistant",
            "name": "assistant",
            "metadata": None,
        },
        {
            "id": "tools",
            "name": "tools",
            "metadata": None,
        },
        {
            "id": "__end__",
            "name": "__end__",
            "metadata": None,
        },
    ]

    workflow_edges = json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.edges"])
    )
    assert all(
        edge in workflow_edges
        for edge in [
            {"source": "__start__", "target": "assistant", "conditional": False},
            {"source": "assistant", "target": "tools", "conditional": True},
            {"source": "assistant", "target": "__end__", "conditional": True},
            {"source": "tools", "target": "assistant", "conditional": False},
        ]
    )
    assert len(workflow_edges) == 4

    for other_span in other_spans:
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.nodes")
            is None
        )
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.edges")
            is None
        )


@pytest.mark.asyncio
async def test_langchain_langgraph_async_stream(span_exporter: InMemorySpanExporter):
    graph_builder = StateGraph(AgentState)
    _ = graph_builder.add_node("assistant", assistant)  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_node("tools", ToolNode(tools))  # pyright: ignore[reportUnknownMemberType]
    _ = graph_builder.add_edge(START, "assistant")
    _ = graph_builder.add_conditional_edges("assistant", tools_condition)
    _ = graph_builder.add_edge("tools", "assistant")
    graph = graph_builder.compile()  # pyright: ignore[reportUnknownMemberType]

    async for _chunk in graph.astream(  # pyright: ignore[reportUnknownMemberType]
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = next(span for span in spans if span.name == "LangGraph.workflow")
    other_spans = [span for span in spans if span.name != "LangGraph.workflow"]
    assert json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.nodes"])
    ) == [
        {
            "id": "__start__",
            "name": "__start__",
            "metadata": None,
        },
        {
            "id": "assistant",
            "name": "assistant",
            "metadata": None,
        },
        {
            "id": "tools",
            "name": "tools",
            "metadata": None,
        },
        {
            "id": "__end__",
            "name": "__end__",
            "metadata": None,
        },
    ]

    workflow_edges = json.loads(
        cast(str, (workflow_span.attributes or {})["lmnr.association.properties.langgraph.edges"])
    )
    assert all(
        edge in workflow_edges
        for edge in [
            {"source": "__start__", "target": "assistant", "conditional": False},
            {"source": "assistant", "target": "tools", "conditional": True},
            {"source": "assistant", "target": "__end__", "conditional": True},
            {"source": "tools", "target": "assistant", "conditional": False},
        ]
    )
    assert len(workflow_edges) == 4

    for other_span in other_spans:
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.nodes")
            is None
        )
        assert (
            (other_span.attributes or {}).get("lmnr.association.properties.langgraph.edges")
            is None
        )

import json
import pytest

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolCall
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing import Annotated, TypedDict


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


def test_langchain_langgraph(exporter: InMemorySpanExporter):
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "assistant")
    graph_builder.add_conditional_edges("assistant", tools_condition)
    graph_builder.add_edge("tools", "assistant")
    graph = graph_builder.compile()

    graph.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})

    spans = exporter.get_finished_spans()
    assert len(spans) == 3
    workflow_span = [span for span in spans if span.name == "LangGraph.workflow"][0]
    other_spans = [span for span in spans if span.name != "LangGraph.workflow"]
    assert json.loads(
        workflow_span.attributes["lmnr.association.properties.langgraph.nodes"]
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
        workflow_span.attributes["lmnr.association.properties.langgraph.edges"]
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
            other_span.attributes.get("lmnr.association.properties.langgraph.nodes")
            is None
        )
        assert (
            other_span.attributes.get("lmnr.association.properties.langgraph.edges")
            is None
        )

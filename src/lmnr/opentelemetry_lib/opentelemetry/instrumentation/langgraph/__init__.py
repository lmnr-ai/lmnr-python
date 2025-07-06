"""OpenTelemetry Langgraph instrumentation"""

import json
import logging
from typing import Collection

from .utils import (
    with_tracer_wrapper,
)

from langchain_core.runnables.graph import Graph
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer
from opentelemetry.context import get_value, attach, set_value

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap


logger = logging.getLogger(__name__)

_instruments = ("langgraph >= 0.1.0",)


@with_tracer_wrapper
def wrap_pregel_stream(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    graph: Graph = instance.get_graph()
    nodes = [
        {
            "id": node.id,
            "name": node.name,
            "metadata": node.metadata,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "conditional": edge.conditional,
        }
        for edge in graph.edges
    ]
    d = {
        "langgraph.edges": json.dumps(edges),
        "langgraph.nodes": json.dumps(nodes),
    }
    association_properties = get_value("lmnr.langgraph.graph") or {}
    association_properties.update(d)
    attach(set_value("lmnr.langgraph.graph", association_properties))
    return wrapped(*args, **kwargs)


@with_tracer_wrapper
async def async_wrap_pregel_stream(
    tracer: Tracer, to_wrap, wrapped, instance, args, kwargs
):
    graph: Graph = await instance.aget_graph()
    nodes = [
        {
            "id": node.id,
            "name": node.name,
            "metadata": node.metadata,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "conditional": edge.conditional,
        }
        for edge in graph.edges
    ]

    d = {
        "langgraph.edges": json.dumps(edges),
        "langgraph.nodes": json.dumps(nodes),
    }
    association_properties = get_value("lmnr.langgraph.graph") or {}
    association_properties.update(d)
    attach(set_value("lmnr.langgraph.graph", association_properties))

    async for item in wrapped(*args, **kwargs):
        yield item


class LanggraphInstrumentor(BaseInstrumentor):
    """An instrumentor for Langgraph."""

    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "0.0.1a0", tracer_provider)

        wrap_function_wrapper(
            module="langgraph.pregel",
            name="Pregel.stream",
            wrapper=wrap_pregel_stream(tracer, "Pregel.stream"),
        )
        wrap_function_wrapper(
            module="langgraph.pregel",
            name="Pregel.astream",
            wrapper=async_wrap_pregel_stream(tracer, "Pregel.astream"),
        )

    def _uninstrument(self, **kwargs):
        unwrap(
            module="langgraph.pregel",
            name="Pregel.stream",
        )
        unwrap(
            module="langgraph.pregel",
            name="Pregel.astream",
        )

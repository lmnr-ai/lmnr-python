"""OpenTelemetry Langgraph instrumentation"""

import json
import logging
from typing import Collection

from langchain_core.runnables.graph import Graph
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper
from opentelemetry.trace import get_tracer
from opentelemetry.context import get_value, attach, set_value

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from lmnr.sdk.utils import with_tracer_wrapper


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
            "langgraph.pregel",
            "Pregel.stream",
            wrap_pregel_stream(tracer, "Pregel.stream"),
        )
        wrap_function_wrapper(
            "langgraph.pregel",
            "Pregel.astream",
            async_wrap_pregel_stream(tracer, "Pregel.astream"),
        )

    def _uninstrument(self, **kwargs):
        # `unwrap` takes (holder, "attr") — NOT wrapt's (module, "Object.method")
        # split used in `_instrument`. With the wrapt split it looks for an
        # attribute literally named "Pregel.stream", finds nothing, and returns
        # silently, leaving both methods wrapped forever.
        unwrap("langgraph.pregel.Pregel", "stream")
        unwrap("langgraph.pregel.Pregel", "astream")

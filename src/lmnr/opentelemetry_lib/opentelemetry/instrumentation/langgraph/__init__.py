"""OpenTelemetry Langgraph instrumentation"""

import json
import logging
from typing import Collection

from importlib.metadata import version
from typing import Any, Sequence

from langchain_core.runnables.graph import Graph
from opentelemetry.context import get_value, attach, set_value

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)


logger = logging.getLogger(__name__)

_instruments = ("langgraph >= 0.1.0",)


def wrap_pregel_stream(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
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


async def async_wrap_pregel_stream(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
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


class LanggraphInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Langgraph."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                langgraph_version = version("langgraph")
            except Exception as e:
                logger.debug(f"Failed to get langgraph version {e}")
                langgraph_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="langgraph",
                version=langgraph_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name="langgraph.pregel",
                    object_name="Pregel",
                    method_name="stream",
                    is_async=False,
                    is_streaming=True,
                    # These wrappers only attach graph topology onto the OTel
                    # context for downstream spans to pick up; they open no span
                    # of their own, so there is no span_name/span_type to read.
                    span_name=None,
                    span_type=None,
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=wrap_pregel_stream,
                ),
                WrappedFunctionSpec(
                    package_name="langgraph.pregel",
                    object_name="Pregel",
                    method_name="astream",
                    is_async=True,
                    is_streaming=True,
                    span_name=None,
                    span_type=None,
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=async_wrap_pregel_stream,
                ),
            ]
        )

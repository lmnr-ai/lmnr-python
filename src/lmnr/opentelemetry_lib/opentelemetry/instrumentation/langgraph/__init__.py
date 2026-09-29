"""OpenTelemetry Langgraph instrumentation"""

import json
import logging
from collections.abc import AsyncIterable, Callable, Collection, Iterable, Sequence
from importlib.metadata import version
from typing import Any, cast

from langchain_core.runnables.graph import Graph
from opentelemetry.context import attach, get_value, set_value
from typing_extensions import TypeVar, override

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

T = TypeVar("T")


def wrap_pregel_stream(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Iterable[T]],
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
)-> Iterable[T]:
    graph = cast(Graph, instance.get_graph())  # pyright: ignore[reportAny]
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
    association_properties = cast(dict[str, str], get_value("lmnr.langgraph.graph") or {})
    association_properties.update(d)
    _attach_token = attach(set_value("lmnr.langgraph.graph", association_properties))
    return wrapped(*args, **kwargs)


async def async_wrap_pregel_stream(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., AsyncIterable[T]],
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
)-> AsyncIterable[T]:
    graph = cast(Graph, await instance.aget_graph())  # pyright: ignore[reportAny]
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
    association_properties = cast(dict[str, str], get_value("lmnr.langgraph.graph") or {})
    association_properties.update(d)
    _attach_token = attach(set_value("lmnr.langgraph.graph", association_properties))

    async for item in wrapped(*args, **kwargs):
        yield item


class LanggraphInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Langgraph."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                langgraph_version = version("langgraph")
            except Exception:
                logger.debug("Failed to get langgraph version", exc_info=True)
                langgraph_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="langgraph",
                version=langgraph_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
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

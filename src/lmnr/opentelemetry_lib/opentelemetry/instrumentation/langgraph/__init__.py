"""OpenTelemetry Langgraph instrumentation"""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Collection,
    Iterator,
    Sequence,
)
from importlib.metadata import version
from typing import Any

from langchain_core.runnables.graph import Graph
from typing_extensions import TypeVar, override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    LANGGRAPH_TOPOLOGY,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.tracing.instrument_options import get_instrument_option
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

_instruments = ("langgraph >= 0.1.0",)

T = TypeVar("T")


def _topology(graph: Graph) -> dict[str, str]:
    nodes = [
        {"id": node.id, "name": node.name, "metadata": node.metadata}
        for node in graph.nodes.values()
    ]
    edges = [
        {"source": edge.source, "target": edge.target, "conditional": edge.conditional}
        for edge in graph.edges
    ]
    return {"langgraph.nodes": json_dumps(nodes), "langgraph.edges": json_dumps(edges)}


def _steps(iterator: Iterator[T], topology: dict[str, str]) -> Iterator[T]:
    # Publish the topology only while the graph itself runs, i.e. inside each
    # `next()`, so it never leaks into whatever the caller does between steps
    try:
        while True:
            token = LANGGRAPH_TOPOLOGY.set(topology)
            try:
                item = next(iterator)
            except StopIteration:
                return
            finally:
                LANGGRAPH_TOPOLOGY.reset(token)
            yield item
    finally:
        # Propagate an early close (e.g. `break` in the caller) to the graph
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def wrap_pregel_stream(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Iterator[T]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Iterator[T]:
    if not get_instrument_option("langgraph", "record_graph"):
        return wrapped(*args, **kwargs)
    try:
        topology = _topology(instance.get_graph())
    except Exception:
        logger.debug("Failed to read the LangGraph topology", exc_info=True)
        return wrapped(*args, **kwargs)
    return _steps(iter(wrapped(*args, **kwargs)), topology)


async def async_wrap_pregel_stream(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., AsyncIterator[T]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> AsyncGenerator[T]:
    topology: dict[str, str] | None = None
    if get_instrument_option("langgraph", "record_graph"):
        try:
            topology = _topology(await instance.aget_graph())
        except Exception:
            logger.debug("Failed to read the LangGraph topology", exc_info=True)
    # `async def` with `yield` is a generator, so unlike the sync wrapper it can't
    # return the graph's iterator as is; it forwards it, with or without the
    # topology. Steps are inlined rather than delegated to a nested generator so
    # closing this one (`break` + `aclose()`, `aclosing(...)`, cancellation)
    # closes the graph right away instead of leaving it to the GC finalizer.
    iterator = aiter(wrapped(*args, **kwargs))
    try:
        while True:
            token = LANGGRAPH_TOPOLOGY.set(topology) if topology is not None else None
            try:
                item = await anext(iterator)
            except StopAsyncIteration:
                return
            finally:
                if token is not None:
                    LANGGRAPH_TOPOLOGY.reset(token)
            yield item
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


class LanggraphInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Langgraph.

    Spans for graph runs come from the LangChain callback handler; this only
    publishes each graph's topology so the handler can record it on the
    graph's root span.
    """

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
                    # These wrappers only publish graph topology for the
                    # LangChain callback handler to pick up; they open no span
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

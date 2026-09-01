"""OpenTelemetry CUA instrumentation"""

import logging
from importlib.metadata import version
from typing import Any, AsyncGenerator, Collection, Sequence

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.sdk.utils import json_dumps

from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode

logger = logging.getLogger(__name__)

_instruments = ("cua-agent >= 0.4.0",)


def _wrap_run(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    parent_span = Laminar.start_span(to_wrap.get("span_name") or "ComputerAgent.run")
    stamp_instrumentation_scope(parent_span, to_wrap)
    instance._lmnr_parent_span = parent_span

    try:
        result: AsyncGenerator[dict[str, Any], None] = wrapped(*args, **kwargs)
        return _abuild_from_streaming_response(to_wrap, parent_span, result)
    except Exception as e:
        if parent_span.is_recording():
            parent_span.set_status(Status(StatusCode.ERROR))
            parent_span.record_exception(e)
            parent_span.end()
        raise


async def _abuild_from_streaming_response(
    to_wrap: WrappedFunctionSpec,
    parent_span: Span,
    response: AsyncGenerator[dict[str, Any], None],
) -> AsyncGenerator[dict[str, Any], None]:
    with Laminar.use_span(parent_span, end_on_exit=True):
        response_iter = aiter(response)
        while True:
            step = None
            step_span = Laminar.start_span("ComputerAgent.step")
            stamp_instrumentation_scope(step_span, to_wrap)
            with Laminar.use_span(step_span):
                try:
                    step = await anext(response_iter)
                    step_span.set_attribute("lmnr.span.output", json_dumps(step))
                    try:
                        # When processing tool calls, each output item is processed separately,
                        # if the output is message, agent.step returns an empty array
                        # https://github.com/trycua/cua/blob/17d670962970a1d1774daaec029ebf92f1f9235e/libs/python/agent/agent/agent.py#L459
                        if len(step.get("output", [])) == 0:
                            continue
                    except Exception:
                        pass
                    if step_span.is_recording():
                        step_span.end()
                except StopAsyncIteration:
                    # don't end on purpose, there is no iteration step here.
                    break

            if step is not None:
                yield step


class CuaAgentInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                cua_version = version("cua-agent")
            except Exception as e:
                logger.debug(f"Failed to get cua-agent version {e}")
                cua_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="cua-agent",
                version=cua_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name="agent.agent",
                    object_name="ComputerAgent",
                    method_name="run",
                    # `run` returns an async generator rather than a coroutine,
                    # so the wrapper is a plain function that returns it.
                    is_async=False,
                    is_streaming=True,
                    span_name="ComputerAgent.run",
                    span_type="DEFAULT",
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_wrap_run,
                ),
            ]
        )

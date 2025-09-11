"""OpenTelemetry CUA instrumentation"""

import logging
from typing import Any, AsyncGenerator, Collection

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr import Laminar
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_instruments = ("cua-agent >= 0.4.0",)


def _wrap_run(
    wrapped,
    instance,
    args,
    kwargs,
):
    parent_span = Laminar.start_span("ComputerAgent.run")
    instance._lmnr_parent_span = parent_span

    try:
        result: AsyncGenerator[dict[str, Any], None] = wrapped(*args, **kwargs)
        return _abuild_from_streaming_response(parent_span, result)
    except Exception as e:
        if parent_span.is_recording():
            parent_span.set_status(Status(StatusCode.ERROR))
            parent_span.record_exception(e)
            parent_span.end()
        raise


async def _abuild_from_streaming_response(
    parent_span: Span, response: AsyncGenerator[dict[str, Any], None]
) -> AsyncGenerator[dict[str, Any], None]:
    with Laminar.use_span(parent_span, end_on_exit=True):
        response_iter = aiter(response)
        while True:
            step = None
            step_span = Laminar.start_span("ComputerAgent.step")
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


class CuaAgentInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        wrap_package = "agent.agent"
        wrap_object = "ComputerAgent"
        wrap_method = "run"
        try:
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                _wrap_run,
            )
        except ModuleNotFoundError:
            pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs):
        wrap_package = "agent.agent"
        wrap_object = "ComputerAgent"
        wrap_method = "run"
        try:
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrap_method,
            )
        except ModuleNotFoundError:
            pass  # that's ok, we don't want to fail if some methods do not exist

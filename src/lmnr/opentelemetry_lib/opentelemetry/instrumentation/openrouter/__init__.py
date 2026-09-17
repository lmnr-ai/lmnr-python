"""OpenTelemetry OpenRouter instrumentation"""

import logging
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from openrouter.utils.eventstreaming import EventStream, EventStreamAsync
from wrapt import wrap_function_wrapper

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    safe_start_span,
    to_dict,
)

from .span_utils import (
    aggregate_chat_chunks,
    response_from_stream_events,
    set_chat_request_attributes,
    set_chat_response_attributes,
    set_responses_request_attributes,
    set_responses_response_attributes,
)

logger = logging.getLogger(__name__)

_instruments = ("openrouter >= 1.0.0",)

WRAPPED_METHODS = [
    {
        "package": "openrouter.chat",
        "object": "Chat",
        "method": "send",
        "span_name": "openrouter.chat",
        "kind": "chat",
    },
    {
        "package": "openrouter.responses",
        "object": "Responses",
        "method": "send",
        "span_name": "openrouter.responses",
        "kind": "responses",
    },
]
WRAPPED_AMETHODS = [
    {**method, "method": "send_async"} for method in WRAPPED_METHODS
]


@dont_throw
def _set_request_attributes(span: Span, kind: str, kwargs: dict):
    if kind == "chat":
        set_chat_request_attributes(span, kwargs)
    else:
        set_responses_request_attributes(span, kwargs)


@dont_throw
def _set_response_attributes(span: Span, kind: str, response: dict | None):
    if not response:
        return
    if kind == "chat":
        set_chat_response_attributes(span, response)
    else:
        set_responses_response_attributes(span, response)


def _record_error(span: Span, error: Exception):
    span.set_attribute("error.type", type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def _finish_stream(span: Span, kind: str, chunks: list[dict]):
    if kind == "chat":
        response = aggregate_chat_chunks(chunks)
    else:
        response = response_from_stream_events(chunks)
    _set_response_attributes(span, kind, response)
    span.end()


def _wrap_stream(stream: EventStream, span: Span, kind: str) -> EventStream:
    def generator(source):
        chunks = []
        try:
            for chunk in source:
                chunks.append(to_dict(chunk))
                yield chunk
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    stream.generator = generator(stream.generator)
    return stream


def _wrap_async_stream(
    stream: EventStreamAsync, span: Span, kind: str
) -> EventStreamAsync:
    async def generator(source):
        chunks = []
        try:
            async for chunk in source:
                chunks.append(to_dict(chunk))
                yield chunk
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    stream.generator = generator(stream.generator)
    return stream


def _wrap(to_wrap: dict):
    def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        kind = to_wrap["kind"]
        span = safe_start_span(
            name=to_wrap["span_name"],
            attributes={"gen_ai.system": "openrouter"},
            span_type="LLM",
        )
        if not span:
            return wrapped(*args, **kwargs)

        _set_request_attributes(span, kind, kwargs)
        try:
            response = wrapped(*args, **kwargs)
        except Exception as e:
            _record_error(span, e)
            span.end()
            raise

        if isinstance(response, EventStream):
            return _wrap_stream(response, span, kind)

        _set_response_attributes(span, kind, to_dict(response))
        span.end()
        return response

    return wrapper


def _awrap(to_wrap: dict):
    async def wrapper(wrapped, instance, args, kwargs):
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        kind = to_wrap["kind"]
        span = safe_start_span(
            name=to_wrap["span_name"],
            attributes={"gen_ai.system": "openrouter"},
            span_type="LLM",
        )
        if not span:
            return await wrapped(*args, **kwargs)

        _set_request_attributes(span, kind, kwargs)
        try:
            response = await wrapped(*args, **kwargs)
        except Exception as e:
            _record_error(span, e)
            span.end()
            raise

        if isinstance(response, EventStreamAsync):
            return _wrap_async_stream(response, span, kind)

        _set_response_attributes(span, kind, to_dict(response))
        span.end()
        return response

    return wrapper


class OpenRouterInstrumentor(BaseInstrumentor):
    """An instrumentor for the OpenRouter Python SDK."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method["package"],
                f"{wrapped_method['object']}.{wrapped_method['method']}",
                _wrap(wrapped_method),
            )
        for wrapped_method in WRAPPED_AMETHODS:
            wrap_function_wrapper(
                wrapped_method["package"],
                f"{wrapped_method['object']}.{wrapped_method['method']}",
                _awrap(wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS + WRAPPED_AMETHODS:
            unwrap(
                f"{wrapped_method['package']}.{wrapped_method['object']}",
                wrapped_method["method"],
            )

"""OpenTelemetry OpenRouter instrumentation"""

from importlib.metadata import version
from typing import Collection

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from openrouter.utils.eventstreaming import EventStream, EventStreamAsync

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    safe_start_span,
    to_dict,
)

from .span_utils import (
    aggregate_chat_chunks,
    response_from_stream_events,
    responses_error_message,
    set_chat_request_attributes,
    set_chat_response_attributes,
    set_embeddings_request_attributes,
    set_embeddings_response_attributes,
    set_responses_request_attributes,
    set_responses_response_attributes,
)

_instruments = ("openrouter >= 1.0.0",)

# (kind, method name, is_async). `kind` doubles as the module and the span suffix.
_WRAPPED_METHODS = (
    ("chat", "send", False),
    ("chat", "send_async", True),
    ("responses", "send", False),
    ("responses", "send_async", True),
    ("embeddings", "generate", False),
    ("embeddings", "generate_async", True),
)

_REQUEST_ATTRIBUTE_SETTERS = {
    "chat": set_chat_request_attributes,
    "responses": set_responses_request_attributes,
    "embeddings": set_embeddings_request_attributes,
}

_RESPONSE_ATTRIBUTE_SETTERS = {
    "chat": set_chat_response_attributes,
    "responses": set_responses_response_attributes,
    "embeddings": set_embeddings_response_attributes,
}


def _kind(to_wrap: WrappedFunctionSpec) -> str:
    """`chat`, `responses` or `embeddings`, derived from the span name."""
    return to_wrap["span_name"].split(".")[-1]


def _start_span(to_wrap: WrappedFunctionSpec) -> Span | None:
    scope = to_wrap.get("instrumentation_scope", {})
    return safe_start_span(
        name=to_wrap["span_name"],
        attributes={
            "gen_ai.system": "openrouter",
            "lmnr.span.instrumentation_scope.name": scope.get("name"),
            "lmnr.span.instrumentation_scope.version": scope.get("version"),
        },
        span_type=to_wrap["span_type"],
    )


@dont_throw
def _set_request_attributes(span: Span, kind: str, kwargs: dict):
    _REQUEST_ATTRIBUTE_SETTERS[kind](span, kwargs)


@dont_throw
def _set_response_attributes(span: Span, kind: str, response: dict | None):
    if not response:
        return
    _RESPONSE_ATTRIBUTE_SETTERS[kind](span, response)
    if kind == "responses":
        error = responses_error_message(response)
        if error:
            span.set_attribute("error.type", response["status"])
            span.set_status(Status(StatusCode.ERROR, error))


def _record_error(span: Span, error: Exception):
    span.set_attribute("error.type", type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def _finish_stream(span: Span, kind: str, chunks: list[dict]):
    # Called from both the wrapping generator and `close()`; only the first ends.
    if not span.is_recording():
        return
    if kind == "chat":
        response = aggregate_chat_chunks(chunks)
    else:
        response = response_from_stream_events(chunks)
    _set_response_attributes(span, kind, response)
    span.end()


def _wrap_stream(stream: EventStream, span: Span, kind: str) -> EventStream:
    chunks: list[dict] = []

    def generator(source):
        try:
            for chunk in source:
                chunks.append(to_dict(chunk))
                yield chunk
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    original_close = stream.close

    def close():
        try:
            original_close()
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    stream.generator = generator(stream.generator)
    stream.close = close
    return stream


def _wrap_async_stream(
    stream: EventStreamAsync, span: Span, kind: str
) -> EventStreamAsync:
    chunks: list[dict] = []

    async def generator(source):
        try:
            async for chunk in source:
                chunks.append(to_dict(chunk))
                yield chunk
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    original_close = stream.close

    async def close():
        try:
            await original_close()
        except Exception as e:
            _record_error(span, e)
            raise
        finally:
            _finish_stream(span, kind, chunks)

    stream.generator = generator(stream.generator)
    stream.close = close
    return stream


def _wrap(to_wrap: WrappedFunctionSpec, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap)
    if not span:
        return wrapped(*args, **kwargs)

    kind = _kind(to_wrap)
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


async def _awrap(to_wrap: WrappedFunctionSpec, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(to_wrap)
    if not span:
        return await wrapped(*args, **kwargs)

    kind = _kind(to_wrap)
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


class OpenRouterInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the OpenRouter Python SDK."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                openrouter_version = version("openrouter")
            except Exception:
                openrouter_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="openrouter", version=openrouter_version
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name=f"openrouter.{kind}",
                    object_name=kind.capitalize(),
                    method_name=method_name,
                    is_async=is_async,
                    is_streaming=kind != "embeddings",
                    span_name=f"openrouter.{kind}",
                    span_type="LLM",
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_awrap if is_async else _wrap,
                )
                for kind, method_name, is_async in _WRAPPED_METHODS
            ]
        )

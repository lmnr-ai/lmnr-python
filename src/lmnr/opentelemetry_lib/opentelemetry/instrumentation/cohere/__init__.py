"""OpenTelemetry instrumentation for the Cohere SDK."""

from __future__ import annotations

from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from contextlib import AbstractContextManager
from importlib.metadata import version
from typing import Any, Protocol

from opentelemetry import context as context_api
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.cohere.utils import (
    RequestType,
    V1ChatStreamAccumulator,
    V2ChatStreamAccumulator,
    set_request_attributes,
    set_response_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import safe_start_span
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_instruments = ("cohere >= 4.2.7",)

_V1_MODULE = "cohere.base_client"
_V1_CLIENT_MODULE = "cohere.client"
_V2_MODULE = "cohere.v2.client"


class CohereSpec(WrappedFunctionSpec, total=False):
    request_type: RequestType
    # `co.v2` / `ClientV2`: different message, response and stream-event shapes
    is_v2: bool


class _StreamAccumulator(Protocol):
    def add(self, event: Any) -> None: ...

    def set_attributes(self, span: Span) -> None: ...


def _request_type(to_wrap: CohereSpec) -> RequestType:
    return to_wrap.get("request_type") or "chat"


def _start_span(to_wrap: CohereSpec, kwargs: dict[str, Any]) -> Span | None:
    span = safe_start_span(
        name=to_wrap.get("span_name") or "cohere.chat",
        attributes={"gen_ai.system": "cohere"},
        span_type="LLM",
    )
    if span is None:
        return None
    stamp_instrumentation_scope(span, to_wrap)
    # every Cohere SDK method takes keyword-only arguments
    set_request_attributes(
        span,
        kwargs,
        _request_type(to_wrap),
        is_v2=bool(to_wrap.get("is_v2")),
        is_streaming=bool(to_wrap.get("is_streaming") or kwargs.get("stream") is True),
    )
    return span


def _set_response_attributes(to_wrap: CohereSpec, span: Span, response: Any):
    set_response_attributes(span, response, _request_type(to_wrap), bool(to_wrap.get("is_v2")))
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


def _new_accumulator(to_wrap: CohereSpec) -> _StreamAccumulator:
    if to_wrap.get("is_v2"):
        return V2ChatStreamAccumulator()
    return V1ChatStreamAccumulator()


def _record_error(span: Span, error: Exception):
    # the `exception` event (not the status) is what marks the span failed in Laminar
    span.set_attribute(ERROR_TYPE, error.__class__.__name__)
    span.record_exception(error, attributes=get_event_attributes_from_context())
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.end()


def _finish_stream(span: Span, accumulator: _StreamAccumulator):
    # Runs on exhaustion and when the consumer stops early (GeneratorExit), so
    # the span always ends; a no-op if `_record_error` already ended it.
    if not span.is_recording():
        return
    accumulator.set_attributes(span)
    span.set_status(Status(StatusCode.OK))
    span.end()


def _use_llm_span(span: Span) -> AbstractContextManager[Any]:
    # exceptions are recorded by `_record_error`, not by `use_span`
    return Laminar.use_span(span, record_exception=False, set_status_on_exception=False)


# `chat_stream` is a generator function: the HTTP request only starts on the
# first `next()`, after the wrapped method has returned, so the span is made
# current around every pull. Not across `yield`: the consumer's code between
# events is not part of the call.
def _stream(span: Span, response: Iterable[Any], accumulator: _StreamAccumulator) -> Generator[Any]:
    iterator = iter(response)
    try:
        while True:
            with _use_llm_span(span):
                try:
                    event = next(iterator)
                except StopIteration:
                    break
            accumulator.add(event)
            yield event
    except Exception as e:
        _record_error(span, e)
        raise
    finally:
        _finish_stream(span, accumulator)


async def _astream(
    span: Span, response: AsyncIterable[Any], accumulator: _StreamAccumulator
) -> AsyncGenerator[Any]:
    iterator = aiter(response)
    try:
        while True:
            with _use_llm_span(span):
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
            accumulator.add(event)
            yield event
    except Exception as e:
        _record_error(span, e)
        raise
    finally:
        _finish_stream(span, accumulator)


def _wrap(
    to_wrap: CohereSpec,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Sync methods; `chat_stream` returns a sync generator."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap, kwargs)
    if span is None:
        return wrapped(*args, **kwargs)

    try:
        with _use_llm_span(span):
            response = wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        raise

    if isinstance(response, Iterator):
        return _stream(span, response, _new_accumulator(to_wrap))

    _set_response_attributes(to_wrap, span, response)
    span.end()
    return response


async def _awrap(
    to_wrap: CohereSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Coroutine methods."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(to_wrap, kwargs)
    if span is None:
        return await wrapped(*args, **kwargs)

    try:
        with _use_llm_span(span):
            response = await wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        raise

    if isinstance(response, AsyncIterator):
        return _astream(span, response, _new_accumulator(to_wrap))

    _set_response_attributes(to_wrap, span, response)
    span.end()
    return response


def _wrap_async_generator(
    to_wrap: CohereSpec,
    wrapped: Callable[..., AsyncIterable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> AsyncIterable[Any]:
    """Async generator functions (async `chat_stream`): calling them does no I/O."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap, kwargs)
    if span is None:
        return wrapped(*args, **kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        raise
    return _astream(span, response, _new_accumulator(to_wrap))


_SPAN_NAMES: dict[RequestType, str] = {
    "chat": "cohere.chat",
    "completion": "cohere.completion",
    "rerank": "cohere.rerank",
    "embedding": "cohere.embed",
}


def _specs(
    package_name: str,
    object_name: str,
    methods: list[tuple[str, RequestType, bool]],
    is_async: bool,
    is_v2: bool,
) -> list[CohereSpec]:
    specs: list[CohereSpec] = []
    for method_name, request_type, is_streaming in methods:
        if not is_async:
            wrapper_function = _wrap
        elif is_streaming:
            wrapper_function = _wrap_async_generator
        else:
            wrapper_function = _awrap
        specs.append(
            CohereSpec(
                package_name=package_name,
                object_name=object_name,
                method_name=method_name,
                span_name=_SPAN_NAMES[request_type],
                is_async=is_async,
                is_streaming=is_streaming,
                request_type=request_type,
                is_v2=is_v2,
                wrapper_function=wrapper_function,
            )
        )
    return specs


_V1_METHODS: list[tuple[str, RequestType, bool]] = [
    ("chat", "chat", False),
    ("chat_stream", "chat", True),
    ("generate", "completion", False),
    ("rerank", "rerank", False),
]
_V2_METHODS: list[tuple[str, RequestType, bool]] = [
    ("chat", "chat", False),
    ("chat_stream", "chat", True),
    ("embed", "embedding", False),
    ("rerank", "rerank", False),
]

# Methods are wrapped on the classes that DEFINE them, so every client that
# inherits them (`ClientV2`, Bedrock, SageMaker, OCI, ...) is covered by one
# wrapper. v1 `embed` is the exception: `Client.embed` overrides
# `BaseCohere.embed` to split large inputs into batches and calls the base method
# once per batch, so wrapping the base would emit one span per batch.
# `generate_stream` is not instrumented (neither was it by OpenLLMetry).
WRAPPED_FUNCTIONS: list[CohereSpec] = [
    *_specs(_V1_MODULE, "BaseCohere", _V1_METHODS, is_async=False, is_v2=False),
    *_specs(_V1_MODULE, "AsyncBaseCohere", _V1_METHODS, is_async=True, is_v2=False),
    *_specs(_V1_CLIENT_MODULE, "Client", [("embed", "embedding", False)], False, False),
    *_specs(_V1_CLIENT_MODULE, "AsyncClient", [("embed", "embedding", False)], True, False),
    *_specs(_V2_MODULE, "V2Client", _V2_METHODS, is_async=False, is_v2=True),
    *_specs(_V2_MODULE, "AsyncV2Client", _V2_METHODS, is_async=True, is_v2=True),
]


class CohereInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the Cohere SDK."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                cohere_version = version("cohere")
            except Exception:
                logger.debug("Failed to get cohere version", exc_info=True)
                cohere_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="cohere",
                version=cohere_version,
            )
        return self._scope

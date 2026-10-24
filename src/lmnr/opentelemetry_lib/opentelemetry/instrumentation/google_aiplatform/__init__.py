"""OpenTelemetry instrumentation for the google-cloud-aiplatform (`vertexai`) SDK."""

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

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_aiplatform.utils import (
    GenerativeStreamAccumulator,
    LanguageStreamAccumulator,
    set_generative_request_attributes,
    set_generative_response_attributes,
    set_language_model_request_attributes,
    set_language_model_response_attributes,
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

_instruments = ("google-cloud-aiplatform >= 1.38.1",)

_GENERATIVE_MODULE = "vertexai.generative_models._generative_models"
_LANGUAGE_MODULE = "vertexai.language_models._language_models"


class GoogleAIPlatformSpec(WrappedFunctionSpec, total=False):
    # the method lives on a ChatSession rather than on the model itself
    is_chat: bool
    # legacy PaLM `vertexai.language_models` rather than Gemini `generative_models`
    is_language_model: bool


class _StreamAccumulator(Protocol):
    def add(self, chunk: Any) -> None: ...

    def set_attributes(self, span: Span) -> None: ...


def _start_span(
    to_wrap: GoogleAIPlatformSpec, instance: Any, args: Sequence[Any], kwargs: dict[str, Any]
) -> Span | None:
    span = safe_start_span(
        name=to_wrap.get("span_name") or "vertexai.generate_content",
        attributes={"gen_ai.system": "gemini"},
        span_type="LLM",
    )
    if span is None:
        return None
    stamp_instrumentation_scope(span, to_wrap)
    set_request_attributes = (
        set_language_model_request_attributes
        if to_wrap.get("is_language_model")
        else set_generative_request_attributes
    )
    set_request_attributes(
        span,
        instance,
        args,
        kwargs,
        is_chat=bool(to_wrap.get("is_chat")),
        is_streaming=bool(to_wrap.get("is_streaming") or kwargs.get("stream")),
    )
    return span


def _set_response_attributes(to_wrap: GoogleAIPlatformSpec, span: Span, response: Any):
    if to_wrap.get("is_language_model"):
        set_language_model_response_attributes(span, response)
    else:
        set_generative_response_attributes(span, response)
    if span.is_recording():
        span.set_status(Status(StatusCode.OK))


def _new_accumulator(to_wrap: GoogleAIPlatformSpec) -> _StreamAccumulator:
    if to_wrap.get("is_language_model"):
        return LanguageStreamAccumulator()
    return GenerativeStreamAccumulator()


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


# The SDK's streams are lazy (the gRPC call runs on the first `next()`, after the
# wrapped method has returned), so the span is made current around every pull.
# Not across `yield`: the consumer's code between chunks is not part of the call.
def _stream(span: Span, response: Iterable[Any], accumulator: _StreamAccumulator) -> Generator[Any]:
    iterator = iter(response)
    try:
        while True:
            with _use_llm_span(span):
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
            accumulator.add(chunk)
            yield chunk
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
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
            accumulator.add(chunk)
            yield chunk
    except Exception as e:
        _record_error(span, e)
        raise
    finally:
        _finish_stream(span, accumulator)


def _wrap(
    to_wrap: GoogleAIPlatformSpec,
    wrapped: Callable[..., Any],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Sync methods; `stream=True` / `*_streaming` return a sync generator."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap, instance, args, kwargs)
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
    to_wrap: GoogleAIPlatformSpec,
    wrapped: Callable[..., Awaitable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Methods returning an awaitable; with `stream=True` it resolves to an async iterator."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(to_wrap, instance, args, kwargs)
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
    to_wrap: GoogleAIPlatformSpec,
    wrapped: Callable[..., AsyncIterable[Any]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> AsyncIterable[Any]:
    """Async generator functions (`*_streaming_async`): calling them does no I/O."""
    if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap, instance, args, kwargs)
    if span is None:
        return wrapped(*args, **kwargs)

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        raise
    return _astream(span, response, _new_accumulator(to_wrap))


# Generative methods are wrapped on the classes that DEFINE them, not on the
# public `GenerativeModel` / `ChatSession`: the preview classes inherit from the
# GA ones (preview `ChatSession` subclasses GA `ChatSession`), so wrapping both
# public paths would stack two wrappers on preview `send_message` and emit two
# spans per call. One wrapper on the defining class covers GA and preview alike.
# Same for the legacy language models: GA and preview `TextGenerationModel`
# share `_TextGenerationModel`, and GA / preview / code chat sessions share
# `_ChatSessionBase` (`CodeChatSession` overrides but delegates via `super()`).
WRAPPED_FUNCTIONS: list[GoogleAIPlatformSpec] = [
    GoogleAIPlatformSpec(
        package_name=_GENERATIVE_MODULE,
        object_name="_GenerativeModel",
        method_name="generate_content",
        span_name="vertexai.generate_content",
        is_async=False,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_GENERATIVE_MODULE,
        object_name="_GenerativeModel",
        method_name="generate_content_async",
        span_name="vertexai.generate_content_async",
        is_async=True,
        wrapper_function=_awrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_GENERATIVE_MODULE,
        object_name="ChatSession",
        method_name="send_message",
        span_name="vertexai.send_message",
        is_async=False,
        is_chat=True,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_GENERATIVE_MODULE,
        object_name="ChatSession",
        method_name="send_message_async",
        span_name="vertexai.send_message_async",
        is_async=True,
        is_chat=True,
        wrapper_function=_awrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_TextGenerationModel",
        method_name="predict",
        span_name="vertexai.predict",
        is_async=False,
        is_language_model=True,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_TextGenerationModel",
        method_name="predict_async",
        span_name="vertexai.predict_async",
        is_async=True,
        is_language_model=True,
        wrapper_function=_awrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_TextGenerationModel",
        method_name="predict_streaming",
        span_name="vertexai.predict_streaming",
        is_async=False,
        is_streaming=True,
        is_language_model=True,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_TextGenerationModel",
        method_name="predict_streaming_async",
        span_name="vertexai.predict_streaming_async",
        is_async=True,
        is_streaming=True,
        is_language_model=True,
        wrapper_function=_wrap_async_generator,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_ChatSessionBase",
        method_name="send_message",
        span_name="vertexai.send_message",
        is_async=False,
        is_chat=True,
        is_language_model=True,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_ChatSessionBase",
        method_name="send_message_async",
        span_name="vertexai.send_message_async",
        is_async=True,
        is_chat=True,
        is_language_model=True,
        wrapper_function=_awrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_ChatSessionBase",
        method_name="send_message_streaming",
        span_name="vertexai.send_message_streaming",
        is_async=False,
        is_streaming=True,
        is_chat=True,
        is_language_model=True,
        wrapper_function=_wrap,
    ),
    GoogleAIPlatformSpec(
        package_name=_LANGUAGE_MODULE,
        object_name="_ChatSessionBase",
        method_name="send_message_streaming_async",
        span_name="vertexai.send_message_streaming_async",
        is_async=True,
        is_streaming=True,
        is_chat=True,
        is_language_model=True,
        wrapper_function=_wrap_async_generator,
    ),
]


class GoogleAIPlatformInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the google-cloud-aiplatform (`vertexai`) SDK."""

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
                aiplatform_version = version("google-cloud-aiplatform")
            except Exception:
                logger.debug("Failed to get google-cloud-aiplatform version", exc_info=True)
                aiplatform_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="google-aiplatform",
                version=aiplatform_version,
            )
        return self._scope

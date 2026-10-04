from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.span import Span
from typing_extensions import TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_response_attributes,
    propagate_trace_context,
    set_client_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import model_as_dict
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_openai_v1,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import dont_throw
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
    set_span_attribute,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

SPAN_NAME = "openai.embeddings"
logger = get_default_logger(__name__)
T = TypeVar("T")


def embeddings_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )
    if span is None:
        return wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_request(span, kwargs, instance)

    try:
        response = wrapped(*args, **kwargs)
        _handle_response(response, span)
        return response
    except Exception as e:
        attributes = {"error.type": e.__class__.__name__}

        span.set_attribute("error.type", e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


async def aembeddings_wrapper(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )
    if span is None:
        return await wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _handle_request(span, kwargs, instance)

    try:
        response = await wrapped(*args, **kwargs)
        _handle_response(response, span)
        return response
    except Exception as e:
        attributes = {
            "error.type": e.__class__.__name__,
        }

        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


@dont_throw
def _handle_request(
    span: Span,
    kwargs: dict[str, Any], # pyright: ignore[reportExplicitAny]
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
):
    _set_request_attributes(span, kwargs, instance)

    if should_send_prompts():
        _set_prompts(span, cast(str|list[str], kwargs.get("input")))

    set_client_attributes(span, instance)

    propagate_trace_context(span, kwargs)


@dont_throw
def _handle_response(
    response: Any,  # pyright: ignore[reportExplicitAny, reportAny],
    span: Span,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response
    # span attributes
    _set_response_attributes(span, response_dict)


def _set_prompts(
    span: Span,
    prompt: str | list[str]
):
    if not span.is_recording() or not prompt:
        return

    if isinstance(prompt, list):
        messages = [{"content": p} for p in prompt]
    else:
        messages = [{"content": prompt}]
    set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))

import logging

from opentelemetry import context as context_api

from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from ..shared import (
    _set_client_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    model_as_dict,
    propagate_trace_context,
)
from ..shared.config import Config
from ..utils import (
    _with_embeddings_telemetry_wrapper,
    dont_throw,
    is_openai_v1,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from lmnr.sdk.utils import json_dumps
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Status, StatusCode

SPAN_NAME = "openai.embeddings"
logger = logging.getLogger(__name__)


@_with_embeddings_telemetry_wrapper
def embeddings_wrapper(
    tracer,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = safe_start_span(
        name=SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )
    if span is None:
        return wrapped(*args, **kwargs)

    _handle_request(span, kwargs, instance)

    try:
        response = wrapped(*args, **kwargs)
        _handle_response(response, span)
        return response
    except Exception as e:  # pylint: disable=broad-except
        attributes = {"error.type": e.__class__.__name__}

        span.set_attribute("error.type", e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
    finally:
        span.end()


@_with_embeddings_telemetry_wrapper
async def aembeddings_wrapper(
    tracer,
    wrapped,
    instance,
    args,
    kwargs,
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = safe_start_span(
        name=SPAN_NAME,
        attributes={"gen_ai.system": "openai"},
        span_type="LLM",
    )
    if span is None:
        return await wrapped(*args, **kwargs)

    _handle_request(span, kwargs, instance)

    try:
        response = await wrapped(*args, **kwargs)
        _handle_response(response, span)
        return response
    except Exception as e:  # pylint: disable=broad-except
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
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)

    if should_send_prompts():
        _set_prompts(span, kwargs.get("input"))

    _set_client_attributes(span, instance)

    if Config.enable_trace_context_propagation:
        propagate_trace_context(span, kwargs)


@dont_throw
def _handle_response(
    response,
    span,
):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response
    # span attributes
    _set_response_attributes(span, response_dict)


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    if isinstance(prompt, list):
        messages = [{"content": p} for p in prompt]
    else:
        messages = [{"content": prompt}]
    _set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))

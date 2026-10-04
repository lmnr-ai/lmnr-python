import logging

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace.status import Status, StatusCode

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
from lmnr.opentelemetry_lib.tracing.context import (
    get_event_attributes_from_context,
)
from lmnr.sdk.utils import json_dumps

from ..shared import (
    _set_request_attributes,
    _set_response_attributes,
    is_streaming_response,
    propagate_trace_context,
    set_client_attributes,
    set_tools_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import model_as_dict
from ..utils import (
    is_openai_v1,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import dont_throw

SPAN_NAME = "openai.completion"

logger = logging.getLogger(__name__)


def completion_wrapper(to_wrap: WrappedFunctionSpec, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
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
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(span, kwargs, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


async def acompletion_wrapper(
    to_wrap: WrappedFunctionSpec, wrapped, instance, args, kwargs
):
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
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        attributes = get_event_attributes_from_context()
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, kwargs, response)
    else:
        _handle_response(response, span)

    span.end()
    return response


@dont_throw
def _handle_request(span, kwargs, instance):
    _set_request_attributes(span, kwargs, instance)
    if should_send_prompts():
        _set_prompts(span, kwargs.get("prompt"))
        set_tools_attributes(span, kwargs.get("functions"))
    set_client_attributes(span, instance)
    propagate_trace_context(span, kwargs)


@dont_throw
def _handle_response(response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(span, response_dict)
    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    if isinstance(prompt, list):
        messages = [{"role": "user", "content": p} for p in prompt]
    else:
        messages = [{"role": "user", "content": prompt}]
    set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))


@dont_throw
def _set_completions(span, choices):
    if not span.is_recording() or not choices:
        return

    set_span_attribute(span, "gen_ai.output.messages", json_dumps(choices))


@dont_throw
def _build_from_streaming_response(span, request_kwargs, response):
    complete_response = {"choices": [], "model": "", "id": ""}
    for item in response:
        yield item
        _accumulate_streaming_response(complete_response, item)

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
async def _abuild_from_streaming_response(span, request_kwargs, response):
    complete_response = {"choices": [], "model": "", "id": ""}
    async for item in response:
        yield item
        _accumulate_streaming_response(complete_response, item)

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


@dont_throw
def _accumulate_streaming_response(complete_response, item):
    if is_openai_v1():
        item = model_as_dict(item)

    complete_response["model"] = item.get("model")
    complete_response["id"] = item.get("id")
    for choice in item.get("choices"):
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append({"index": index, "text": ""})
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")

        if choice.get("text"):
            complete_choice["text"] += choice.get("text")

    return complete_response

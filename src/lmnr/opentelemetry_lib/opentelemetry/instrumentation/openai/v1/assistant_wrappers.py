import logging
import time

from opentelemetry import context as context_api
from ..shared import (
    _set_span_attribute,
    model_as_dict,
)
from ..shared.config import Config
from ..utils import (
    _with_tracer_wrapper,
    dont_throw,
)
from lmnr.sdk.utils import json_dumps
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import LLMRequestTypeValues, SpanAttributes
from opentelemetry.trace import SpanKind, Status, StatusCode

from openai._legacy_response import LegacyAPIResponse
from openai.types.beta.threads.run import Run

logger = logging.getLogger(__name__)

assistants = {}
runs = {}


@_with_tracer_wrapper
def assistants_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)

    assistants[response.id] = {
        "model": kwargs.get("model"),
        "instructions": kwargs.get("instructions"),
    }

    return response


@_with_tracer_wrapper
def runs_create_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    thread_id = kwargs.get("thread_id")
    instructions = kwargs.get("instructions")

    try:
        response = wrapped(*args, **kwargs)
        response_dict = model_as_dict(response)

        runs[thread_id] = {
            "start_time": time.time_ns(),
            "assistant_id": kwargs.get("assistant_id"),
            "instructions": instructions,
            "run_id": response_dict.get("id"),
        }

        return response
    except Exception as e:
        runs[thread_id] = {
            "exception": e,
            "end_time": time.time_ns(),
        }
        raise


@_with_tracer_wrapper
def runs_retrieve_wrapper(tracer, wrapped, instance, args, kwargs):
    @dont_throw
    def process_response(response):
        if type(response) is LegacyAPIResponse:
            parsed_response = response.parse()
        else:
            parsed_response = response
        assert type(parsed_response) is Run

        if parsed_response.thread_id in runs:
            thread_id = parsed_response.thread_id
            runs[thread_id]["end_time"] = time.time_ns()
            if parsed_response.usage:
                runs[thread_id]["usage"] = parsed_response.usage

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    try:
        response = wrapped(*args, **kwargs)
        process_response(response)
        return response
    except Exception as e:
        thread_id = kwargs.get("thread_id")
        if thread_id in runs:
            runs[thread_id]["exception"] = e
            runs[thread_id]["end_time"] = time.time_ns()
        raise


@_with_tracer_wrapper
def messages_list_wgrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    id = kwargs.get("thread_id")

    response = wrapped(*args, **kwargs)

    response_dict = model_as_dict(response)
    if id not in runs:
        return response

    run = runs[id]
    messages = sorted(response_dict["data"], key=lambda x: x["created_at"])

    span = tracer.start_span(
        "openai.assistant.run",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value},
        start_time=run.get("start_time"),
        context=get_current_context(),
    )

    if exception := run.get("exception"):
        span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
        span.record_exception(exception, attributes=get_event_attributes_from_context())
        span.set_status(Status(StatusCode.ERROR, str(exception)))
        span.end(run.get("end_time"))

    input_messages = []
    output_messages = []

    if assistants.get(run["assistant_id"]) is not None or Config.enrich_assistant:
        if Config.enrich_assistant:
            assistant = model_as_dict(
                instance._client.beta.assistants.retrieve(run["assistant_id"])
            )
            assistants[run["assistant_id"]] = assistant
        else:
            assistant = assistants[run["assistant_id"]]

        _set_span_attribute(
            span,
            SpanAttributes.LLM_SYSTEM,
            "openai",
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_REQUEST_MODEL,
            assistant["model"],
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_RESPONSE_MODEL,
            assistant["model"],
        )
        input_messages.append(
            {"role": "system", "content": assistant["instructions"]}
        )

    input_messages.append({"role": "system", "content": run["instructions"]})

    for msg in messages:
        content = msg.get("content")
        message_content = content[0].get("text").get("value")
        message_role = msg.get("role")
        if message_role in ["user", "system"]:
            input_messages.append({"role": message_role, "content": message_content})
        else:
            output_messages.append(
                {
                    "role": msg.get("role"),
                    "content": message_content,
                    "id": msg.get("id"),
                }
            )

    _set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))
    _set_span_attribute(span, "gen_ai.output.messages", json_dumps(output_messages))

    if run.get("usage"):
        usage_dict = model_as_dict(run.get("usage"))
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            usage_dict.get("completion_tokens"),
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            usage_dict.get("prompt_tokens"),
        )

    span.end(run.get("end_time"))

    return response


@_with_tracer_wrapper
def runs_create_and_stream_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    assistant_id = kwargs.get("assistant_id")
    instructions = kwargs.get("instructions")

    span = tracer.start_span(
        "openai.assistant.run_stream",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.CHAT.value},
        context=get_current_context(),
    )

    input_messages = []
    if assistants.get(assistant_id) is not None or Config.enrich_assistant:
        if Config.enrich_assistant:
            assistant = model_as_dict(
                instance._client.beta.assistants.retrieve(assistant_id)
            )
            assistants[assistant_id] = assistant
        else:
            assistant = assistants[assistant_id]

        _set_span_attribute(
            span, SpanAttributes.LLM_REQUEST_MODEL, assistants[assistant_id]["model"]
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_SYSTEM,
            "openai",
        )
        _set_span_attribute(
            span,
            SpanAttributes.LLM_RESPONSE_MODEL,
            assistants[assistant_id]["model"],
        )
        input_messages.append(
            {"role": "system", "content": assistants[assistant_id]["instructions"]}
        )

    input_messages.append({"role": "system", "content": instructions})
    _set_span_attribute(span, "gen_ai.input.messages", json_dumps(input_messages))

    from ..v1.event_handler_wrapper import (
        EventHandlerWrapper,
    )

    kwargs["event_handler"] = EventHandlerWrapper(
        original_handler=kwargs["event_handler"],
        span=span,
    )

    try:
        response = wrapped(*args, **kwargs)
        return response
    except Exception as e:
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=get_event_attributes_from_context())
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

import logging
import time

from opentelemetry import context as context_api
from opentelemetry.trace import Span, Tracer
from ..shared import (
    _set_span_attribute,
    model_as_dict,
)
from ..utils import (
    _with_tracer_wrapper,
    dont_throw,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.trace import Status, StatusCode

from openai._legacy_response import LegacyAPIResponse
from openai.types.beta.threads.run import Run

logger = logging.getLogger(__name__)

assistants = {}
runs = {}


# We are not reusing the safe_start_span from shared utils because we need to pass the
# start_time parameter, which we don't want to expose on Laminar.start_span.
# Assistants API is deprecated anyway, so it's not risky to leave old `tracer` path here.
def _safe_start_span(
    tracer: Tracer,
    name: str,
    start_time: int | None = None,
) -> Span | None:
    try:
        return tracer.start_span(
            name,
            start_time=start_time,
            context=get_current_context(),
            attributes={
                "gen_ai.system": "openai",
                "lmnr.span.type": "LLM",
            },
        )
    except Exception:
        logger.debug(f"[openai assistants] Failed to start span: {name}", exc_info=True)
        return None


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
def messages_list_wrapper(tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    id = kwargs.get("thread_id")

    response = wrapped(*args, **kwargs)

    response_dict = model_as_dict(response)
    if id not in runs:
        return response

    run = runs[id]
    messages = sorted(
        response_dict.get("data", []), key=lambda x: x.get("created_at", 0)
    )

    span = _safe_start_span(
        tracer,
        "openai.assistant.run",
        start_time=run.get("start_time"),
    )
    if span is None:
        return response

    if exception := run.get("exception"):
        span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
        span.record_exception(exception, attributes=get_event_attributes_from_context())
        span.set_status(Status(StatusCode.ERROR, str(exception)))
        span.end(run.get("end_time"))

    prompt_index = 0
    if assistants.get(run["assistant_id"]) is not None:
        assistant = assistants[run["assistant_id"]]

        _set_span_attribute(
            span,
            "gen_ai.system",
            "openai",
        )
        _set_span_attribute(
            span,
            GEN_AI_REQUEST_MODEL,
            assistant["model"],
        )
        _set_span_attribute(
            span,
            GEN_AI_RESPONSE_MODEL,
            assistant["model"],
        )
        _set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "system")
        _set_span_attribute(
            span,
            f"gen_ai.prompt.{prompt_index}.content",
            assistant["instructions"],
        )
        prompt_index += 1
    _set_span_attribute(span, f"gen_ai.prompt.{prompt_index}.role", "system")
    _set_span_attribute(
        span,
        f"gen_ai.prompt.{prompt_index}.content",
        run["instructions"],
    )
    prompt_index += 1

    completion_index = 0
    for msg in messages:
        prefix = f"gen_ai.completion.{completion_index}"
        content = msg.get("content")

        message_content = content[0].get("text").get("value")
        message_role = msg.get("role")
        if message_role in ["user", "system"]:
            _set_span_attribute(
                span,
                f"gen_ai.prompt.{prompt_index}.role",
                message_role,
            )
            _set_span_attribute(
                span,
                f"gen_ai.prompt.{prompt_index}.content",
                message_content,
            )
            prompt_index += 1
        else:

            _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
            _set_span_attribute(span, f"{prefix}.content", message_content)
            _set_span_attribute(
                span, f"gen_ai.response.{completion_index}.id", msg.get("id")
            )
            completion_index += 1

    if run.get("usage"):
        usage_dict = model_as_dict(run.get("usage"))
        _set_span_attribute(
            span,
            GEN_AI_USAGE_OUTPUT_TOKENS,
            usage_dict.get("completion_tokens"),
        )
        _set_span_attribute(
            span,
            GEN_AI_USAGE_INPUT_TOKENS,
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

    span = _safe_start_span(
        tracer,
        "openai.assistant.run_stream",
    )
    if span is None:
        return wrapped(*args, **kwargs)

    i = 0
    if assistants.get(assistant_id) is not None:
        _set_span_attribute(
            span, GEN_AI_REQUEST_MODEL, assistants[assistant_id]["model"]
        )
        _set_span_attribute(
            span,
            "gen_ai.system",
            "openai",
        )
        _set_span_attribute(
            span,
            GEN_AI_RESPONSE_MODEL,
            assistants[assistant_id]["model"],
        )
        _set_span_attribute(span, f"gen_ai.prompt.{i}.role", "system")
        _set_span_attribute(
            span,
            f"gen_ai.prompt.{i}.content",
            assistants[assistant_id]["instructions"],
        )
        i += 1
    _set_span_attribute(span, f"gen_ai.prompt.{i}.role", "system")
    _set_span_attribute(span, f"gen_ai.prompt.{i}.content", instructions)

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

"""OpenTelemetry Google Generative AI API instrumentation"""

from collections import defaultdict
import json
import logging
import os
from typing import AsyncGenerator, Callable, Collection, Generator

from google.genai import types

from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)

from .config import (
    Config,
)
from .schema_utils import SchemaJSONEncoder, process_schema
from .utils import (
    dont_throw,
    get_content,
    merge_text_parts,
    process_content_union,
    process_stream_chunk,
    role_from_content_union,
    set_span_attribute,
    to_dict,
    with_tracer_wrapper,
)
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Span, Status, StatusCode
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap

from opentelemetry.semconv_ai import (
    SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY,
    SpanAttributes,
    LLMRequestTypeValues,
)

logger = logging.getLogger(__name__)

_instruments = ("google-genai >= 1.0.0",)

WRAPPED_METHODS = [
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
        "is_streaming": False,
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content",
        "span_name": "gemini.generate_content",
        "is_streaming": False,
        "is_async": True,
    },
    {
        "package": "google.genai.models",
        "object": "Models",
        "method": "generate_content_stream",
        "span_name": "gemini.generate_content_stream",
        "is_streaming": True,
        "is_async": False,
    },
    {
        "package": "google.genai.models",
        "object": "AsyncModels",
        "method": "generate_content_stream",
        "span_name": "gemini.generate_content_stream",
        "is_streaming": True,
        "is_async": True,
    },
]


def should_send_prompts():
    return (
        os.getenv("LAMINAR_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


@dont_throw
def _set_request_attributes(span, args, kwargs):
    config_dict = to_dict(kwargs.get("config", {}))
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_MODEL, kwargs.get("model")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE,
        config_dict.get("temperature"),
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_TOP_P, config_dict.get("top_p")
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_TOP_K, config_dict.get("top_k")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT,
        config_dict.get("candidate_count"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS,
        config_dict.get("max_output_tokens"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES,
        config_dict.get("stop_sequences"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        config_dict.get("frequency_penalty"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        config_dict.get("presence_penalty"),
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_SEED, config_dict.get("seed")
    )

    if schema := config_dict.get("response_schema"):
        try:
            set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json.dumps(process_schema(schema), cls=SchemaJSONEncoder),
            )
        except Exception:
            pass
    elif json_schema := config_dict.get("response_json_schema"):
        try:
            set_span_attribute(
                span,
                SpanAttributes.LLM_REQUEST_STRUCTURED_OUTPUT_SCHEMA,
                json_dumps(json_schema),
            )
        except Exception:
            pass

    tools: list[types.FunctionDeclaration] = []
    arg_tools = config_dict.get("tools", kwargs.get("tools"))
    if arg_tools:
        for tool in arg_tools:
            if isinstance(tool, types.Tool):
                tools += tool.function_declarations or []
            elif isinstance(tool, Callable):
                tools.append(types.FunctionDeclaration.from_callable(tool))

    for tool_num, tool in enumerate(tools):
        tool_dict = to_dict(tool)
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.name",
            tool_dict.get("name"),
        )
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.description",
            tool_dict.get("description"),
        )
        set_span_attribute(
            span,
            f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{tool_num}.parameters",
            json_dumps(tool_dict.get("parameters")),
        )

    if should_send_prompts():
        i = 0
        system_instruction: types.ContentUnion | None = config_dict.get(
            "system_instruction"
        )
        if system_instruction:
            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.content",
                (get_content(process_content_union(system_instruction)) or {}).get(
                    "text", ""
                ),
            )
            set_span_attribute(
                span, f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.role", "system"
            )
            i += 1
        contents = kwargs.get("contents", [])
        if not isinstance(contents, list):
            contents = [contents]
        for content in contents:
            processed_content = process_content_union(content)
            content_payload = get_content(processed_content)
            if isinstance(content_payload, dict):
                content_payload = [content_payload]

            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.content",
                (
                    content_payload
                    if isinstance(content_payload, str)
                    else json_dumps(content_payload)
                ),
            )
            blocks = (
                processed_content
                if isinstance(processed_content, list)
                else [processed_content]
            )
            tool_call_index = 0
            for block in blocks:
                block_dict = to_dict(block)

                if not block_dict.get("function_call"):
                    continue
                function_call = to_dict(block_dict.get("function_call", {}))

                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.tool_calls.{tool_call_index}.name",
                    function_call.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.tool_calls.{tool_call_index}.id",
                    (
                        function_call.get("id")
                        if function_call.get("id") is not None
                        else function_call.get("name")
                    ),  # google genai doesn't support tool call ids
                )
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.tool_calls.{tool_call_index}.arguments",
                    json_dumps(function_call.get("arguments")),
                )
                tool_call_index += 1

            set_span_attribute(
                span,
                f"{gen_ai_attributes.GEN_AI_PROMPT}.{i}.role",
                role_from_content_union(content) or "user",
            )
            i += 1


@dont_throw
def _set_response_attributes(span, response: types.GenerateContentResponse):
    candidates = response.candidates or []
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_RESPONSE_ID, to_dict(response).get("response_id")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_RESPONSE_MODEL,
        to_dict(response).get("model_version"),
    )

    if response.usage_metadata:
        usage_dict = to_dict(response.usage_metadata)
        candidates_token_count = usage_dict.get("candidates_token_count")
        # unlike OpenAI, and unlike input cached tokens, thinking tokens are
        # not counted as part of candidates token count, so we need to add them
        # separately for consistency with other instrumentations
        thoughts_token_count = usage_dict.get("thoughts_token_count")
        output_token_count = (
            (candidates_token_count or 0) + (thoughts_token_count or 0)
            if candidates_token_count is not None or thoughts_token_count is not None
            else None
        )
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
            usage_dict.get("prompt_token_count"),
        )
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            output_token_count,
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_TOTAL_TOKENS,
            usage_dict.get("total_token_count"),
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
            usage_dict.get("cached_content_token_count"),
        )
        set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_REASONING_TOKENS,
            thoughts_token_count,
        )

    if should_send_prompts():
        set_span_attribute(
            span, f"{gen_ai_attributes.GEN_AI_COMPLETION}.0.role", "model"
        )
        candidates_list = candidates if isinstance(candidates, list) else [candidates]
        i = 0
        for candidate in candidates_list:
            has_content = False
            processed_content = process_content_union(candidate.content)
            content_payload = get_content(processed_content)
            if isinstance(content_payload, dict):
                content_payload = [content_payload]

            set_span_attribute(
                span, f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.role", "model"
            )
            if content_payload:
                has_content = True
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.content",
                    (
                        content_payload
                        if isinstance(content_payload, str)
                        else json_dumps(content_payload)
                    ),
                )
            blocks = (
                processed_content
                if isinstance(processed_content, list)
                else [processed_content]
            )

            tool_call_index = 0
            for block in blocks:
                block_dict = to_dict(block)
                if not block_dict.get("function_call"):
                    continue
                function_call = to_dict(block_dict.get("function_call", {}))
                has_content = True
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.tool_calls.{tool_call_index}.name",
                    function_call.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.tool_calls.{tool_call_index}.id",
                    (
                        function_call.get("id")
                        if function_call.get("id") is not None
                        else function_call.get("name")
                    ),  # google genai doesn't support tool call ids
                )
                set_span_attribute(
                    span,
                    f"{gen_ai_attributes.GEN_AI_COMPLETION}.{i}.tool_calls.{tool_call_index}.arguments",
                    json_dumps(function_call.get("arguments")),
                )
                tool_call_index += 1
            if has_content:
                i += 1


@dont_throw
def _build_from_streaming_response(
    span: Span, response: Generator[types.GenerateContentResponse, None, None]
) -> Generator[types.GenerateContentResponse, None, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    for chunk in response:
        try:
            span.add_event("llm.content.completion.chunk")
        except Exception:
            pass
        # Important: do all processing in a separate sync function, that is
        # wrapped in @dont_throw. If we did it here, the @dont_throw on top of
        # this function would not be able to catch the errors, as they are
        # raised later, after the generator is returned, and when it is being
        # consumed.
        chunk_result = process_stream_chunk(
            chunk,
            role,
            model_version,
            aggregated_usage_metadata,
            final_parts,
        )
        # even though process_stream_chunk can't return None, the result can be
        # None, if the processing throws an error (see @dont_throw)
        if chunk_result:
            role = chunk_result["role"]
            model_version = chunk_result["model_version"]
        yield chunk

    try:
        compound_response = types.GenerateContentResponse(
            candidates=[
                {
                    "content": {
                        "parts": merge_text_parts(final_parts),
                        "role": role,
                    },
                }
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadataDict(
                **aggregated_usage_metadata
            ),
            model_version=model_version,
        )
        if span.is_recording():
            _set_response_attributes(span, compound_response)
    finally:
        if span.is_recording():
            span.end()


@dont_throw
async def _abuild_from_streaming_response(
    span: Span, response: AsyncGenerator[types.GenerateContentResponse, None]
) -> AsyncGenerator[types.GenerateContentResponse, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    async for chunk in response:
        try:
            span.add_event("llm.content.completion.chunk")
        except Exception:
            pass
        # Important: do all processing in a separate sync function, that is
        # wrapped in @dont_throw. If we did it here, the @dont_throw on top of
        # this function would not be able to catch the errors, as they are
        # raised later, after the generator is returned, and when it is being
        # consumed.
        chunk_result = process_stream_chunk(
            chunk,
            role,
            model_version,
            aggregated_usage_metadata,
            final_parts,
        )
        # even though process_stream_chunk can't return None, the result can be
        # None, if the processing throws an error (see @dont_throw)
        if chunk_result:
            role = chunk_result["role"]
            model_version = chunk_result["model_version"]
        yield chunk

    try:
        compound_response = types.GenerateContentResponse(
            candidates=[
                {
                    "content": {
                        "parts": merge_text_parts(final_parts),
                        "role": role,
                    },
                }
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadataDict(
                **aggregated_usage_metadata
            ),
            model_version=model_version,
        )
        if span.is_recording():
            _set_response_attributes(span, compound_response)
    finally:
        if span.is_recording():
            span.end()


@with_tracer_wrapper
def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
        context=get_current_context(),
    )

    if span.is_recording():
        _set_request_attributes(span, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
        if to_wrap.get("is_streaming"):
            return _build_from_streaming_response(span, response)
        if span.is_recording():
            _set_response_attributes(span, response)
        span.end()
        return response
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


@with_tracer_wrapper
async def _awrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY) or context_api.get_value(
        SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY
    ):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            SpanAttributes.LLM_SYSTEM: "gemini",
            SpanAttributes.LLM_REQUEST_TYPE: LLMRequestTypeValues.COMPLETION.value,
        },
        context=get_current_context(),
    )

    if span.is_recording():
        _set_request_attributes(span, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
        if to_wrap.get("is_streaming"):
            return _abuild_from_streaming_response(span, response)
        else:
            if span.is_recording():
                _set_response_attributes(span, response)

            span.end()
            return response
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


class GoogleGenAiSdkInstrumentor(BaseInstrumentor):
    """An instrumentor for Google GenAI's client library."""

    def __init__(
        self,
        exception_logger=None,
        upload_base64_image=None,
        convert_image_to_openai_format=True,
    ):
        super().__init__()
        Config.exception_logger = exception_logger
        Config.upload_base64_image = upload_base64_image
        Config.convert_image_to_openai_format = convert_image_to_openai_format

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "0.0.1a1", tracer_provider)

        for wrapped_method in WRAPPED_METHODS:
            wrap_function_wrapper(
                wrapped_method.get("package"),
                f"{wrapped_method.get('object')}.{wrapped_method.get('method')}",
                (
                    _awrap(tracer, wrapped_method)
                    if wrapped_method.get("is_async")
                    else _wrap(tracer, wrapped_method)
                ),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            unwrap(
                f"{wrapped_method.get('package')}.{wrapped_method.get('object')}",
                wrapped_method.get("method"),
            )

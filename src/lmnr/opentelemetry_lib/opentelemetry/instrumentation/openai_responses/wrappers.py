import json
from typing import Any, Optional, Union
import pydantic
import time

from openai._legacy_response import LegacyAPIResponse
from openai.types.responses import (
    Response,
    ResponseInputParam,
    ResponseOutputItem,
    ResponseUsage,
    ToolParam,
)
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_COMPLETION,
    GEN_AI_PROMPT,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
)
from opentelemetry.trace import SpanKind, Span, StatusCode, Tracer
from typing_extensions import TypeAlias

from .utils import (
    set_span_attribute,
    with_tracer_wrapper,
    dont_throw,
    is_validator_iterator,
    model_as_dict,
    should_send_prompts,
)


StringAlias: TypeAlias = str
InputClass: TypeAlias = StringAlias | ResponseInputParam


class TracedData(pydantic.BaseModel):
    start_time: float  # time.time_ns()
    response_id: str
    input: InputClass
    # system message
    instructions: Optional[str] = pydantic.Field(default=None)
    tools: Optional[dict[str, ToolParam]] = pydantic.Field(default=None)
    output_blocks: Optional[dict[str, ResponseOutputItem]] = pydantic.Field(
        default=None
    )
    usage: Optional[ResponseUsage] = pydantic.Field(default=None)
    output_text: Optional[str] = pydantic.Field(default=None)
    request_model: Optional[str] = pydantic.Field(default=None)
    response_model: Optional[str] = pydantic.Field(default=None)


responses: dict[str, TracedData] = {}


def parse_response(response: Union[LegacyAPIResponse, Response]) -> Response:
    if isinstance(response, LegacyAPIResponse):
        return response.parse()
    return response


def get_tools_from_kwargs(kwargs: dict) -> dict[str, ToolParam]:
    tools_input = kwargs.get("tools", [])
    return {
        tool.get("id"): ToolParam(**tool)
        for tool in tools_input
        if tool.get("type") != "computer-preview" and tool.get("id")
    }


def process_content_block(
    block: dict[str, Any],
) -> dict[str, Any]:
    # TODO: keep the original type once backend supports it
    if block.get("type") in ["text", "input_text", "output_text"]:
        return {"type": "text", "text": block.get("text")}
    elif block.get("type") in ["image", "input_image", "output_image"]:
        return {
            "type": "image",
            "image_url": block.get("image_url"),
            "detail": block.get("detail"),
            "file_id": block.get("file_id"),
        }
    elif block.get("type") in ["file", "input_file", "output_file"]:
        return {
            "type": "file",
            "file_id": block.get("file_id"),
            "filename": block.get("filename"),
            "file_data": block.get("file_data"),
        }
    return block


@dont_throw
def set_data_attributes(traced_response: TracedData, span: Span):
    set_span_attribute(span, GEN_AI_REQUEST_MODEL, traced_response.request_model)
    set_span_attribute(span, GEN_AI_RESPONSE_ID, traced_response.response_id)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, traced_response.response_model)
    if usage := traced_response.usage:
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
        set_span_attribute(
            span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.total_tokens
        )
        if usage.input_tokens_details:
            set_span_attribute(
                span,
                SpanAttributes.LLM_USAGE_CACHE_READ_INPUT_TOKENS,
                usage.input_tokens_details.cached_tokens,
            )
        # TODO: add reasoning tokens in output token details

    if should_send_prompts():
        prompt_index = 0
        if traced_response.instructions:
            set_span_attribute(
                span,
                f"{GEN_AI_PROMPT}.{prompt_index}.content",
                traced_response.instructions,
            )
            set_span_attribute(span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "system")
            prompt_index += 1

        if isinstance(traced_response.input, str):
            set_span_attribute(
                span, f"{GEN_AI_PROMPT}.{prompt_index}.content", traced_response.input
            )
            set_span_attribute(span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "user")
            prompt_index += 1
        else:
            for block in traced_response.input:
                block_dict = model_as_dict(block)
                if block_dict.get("type", "message") == "message":
                    content = block_dict.get("content")
                    if is_validator_iterator(content):
                        # we're after the actual call here, so we can consume the iterator
                        content = [process_content_block(block) for block in content]
                    try:
                        stringified_content = (
                            content if isinstance(content, str) else json.dumps(content)
                        )
                    except Exception:
                        stringified_content = (
                            str(content) if content is not None else ""
                        )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.content",
                        stringified_content,
                    )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.role",
                        block_dict.get("role"),
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call_output":
                    set_span_attribute(
                        span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "computer-call"
                    )
                    output_image_url = block_dict.get("output", {}).get("image_url")
                    if output_image_url:
                        set_span_attribute(
                            span,
                            f"{GEN_AI_PROMPT}.{prompt_index}.content",
                            json.dumps(
                                [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": output_image_url},
                                    }
                                ]
                            ),
                        )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call":
                    set_span_attribute(
                        span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "assistant"
                    )
                    call_content = {}
                    if block_dict.get("id"):
                        call_content["id"] = block_dict.get("id")
                    if block_dict.get("call_id"):
                        call_content["call_id"] = block_dict.get("call_id")
                    if block_dict.get("action"):
                        call_content["action"] = block_dict.get("action")
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.content",
                        json.dumps(call_content),
                    )
                    prompt_index += 1
                # TODO: handle other block types

        set_span_attribute(span, f"{GEN_AI_COMPLETION}.0.role", "assistant")
        if traced_response.output_text:
            set_span_attribute(
                span, f"{GEN_AI_COMPLETION}.0.content", traced_response.output_text
            )
        tool_call_index = 0
        for block in traced_response.output_blocks.values():
            block_dict = model_as_dict(block)
            if block_dict.get("type") == "message":
                # either a refusal or handled in output_text above
                continue
            if block_dict.get("type") == "function_call":
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    block_dict.get("name"),
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
                    block_dict.get("arguments"),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "file_search_call":
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "file_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "web_search_call":
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("id"),
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "web_search_call",
                )
                tool_call_index += 1
            elif block_dict.get("type") == "computer_call":
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.id",
                    block_dict.get("call_id"),
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.name",
                    "computer_call",
                )
                set_span_attribute(
                    span,
                    f"{GEN_AI_COMPLETION}.0.tool_calls.{tool_call_index}.arguments",
                    json.dumps(block_dict.get("action")),
                )
                tool_call_index += 1
            elif block_dict.get("type") == "reasoning":
                set_span_attribute(
                    span, f"{GEN_AI_COMPLETION}.0.reasoning", block_dict.get("summary")
                )
            # TODO: handle other block types, in particular other calls


@with_tracer_wrapper
def responses_get_or_create_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)
    start_time = time.time_ns()

    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        response_id = kwargs.get("response_id")
        existing_data = {}
        if response_id and response_id in responses:
            existing_data = responses[response_id].model_dump()
        traced_data = TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=response_id or "",
            input=kwargs.get("input", existing_data.get("input", [])),
            instructions=kwargs.get("instructions", existing_data.get("instructions")),
            tools=get_tools_from_kwargs(kwargs) or existing_data.get("tools", {}),
            output_blocks=existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage"),
            output_text=kwargs.get("output_text", existing_data.get("output_text", "")),
            request_model=kwargs.get("model", existing_data.get("request_model", "")),
            response_model=existing_data.get("response_model", ""),
        )
        span = tracer.start_span(
            "openai.responses",
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        if traced_data:
            set_data_attributes(traced_data, span)
        span.end()
        raise
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(kwargs)

    merged_tools = existing_data.get("tools", {}) | request_tools

    traced_data = TracedData(
        start_time=existing_data.get("start_time", start_time),
        response_id=parsed_response.id,
        input=existing_data.get("input", kwargs.get("input")),
        instructions=existing_data.get("instructions", kwargs.get("instructions")),
        tools=merged_tools if merged_tools else None,
        output_blocks={block.id: block for block in parsed_response.output}
        | existing_data.get("output_blocks", {}),
        usage=existing_data.get("usage", parsed_response.usage),
        output_text=existing_data.get("output_text", parsed_response.output_text),
        request_model=existing_data.get("request_model", kwargs.get("model")),
        response_model=existing_data.get("response_model", parsed_response.model),
    )
    responses[parsed_response.id] = traced_data

    if parsed_response.status == "completed":
        span = tracer.start_span(
            "openai.responses",
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        set_data_attributes(traced_data, span)
        span.end()

    return response


@with_tracer_wrapper
async def async_responses_get_or_create_wrapper(
    tracer: Tracer, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)
    start_time = time.time_ns()

    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        response_id = kwargs.get("response_id")
        existing_data = {}
        if response_id and response_id in responses:
            existing_data = responses[response_id].model_dump()
        traced_data = TracedData(
            start_time=existing_data.get("start_time", start_time),
            response_id=response_id or "",
            input=kwargs.get("input", existing_data.get("input", [])),
            instructions=kwargs.get(
                "instructions", existing_data.get("instructions", "")
            ),
            tools=get_tools_from_kwargs(kwargs) or existing_data.get("tools", {}),
            output_blocks=existing_data.get("output_blocks", {}),
            usage=existing_data.get("usage"),
            output_text=kwargs.get("output_text", existing_data.get("output_text")),
            request_model=kwargs.get("model", existing_data.get("request_model")),
            response_model=existing_data.get("response_model"),
        )
        span = tracer.start_span(
            "openai.responses",
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, str(e))
        if traced_data:
            set_data_attributes(traced_data, span)
        span.end()
        raise
    parsed_response = parse_response(response)

    existing_data = responses.get(parsed_response.id)
    if existing_data is None:
        existing_data = {}
    else:
        existing_data = existing_data.model_dump()

    request_tools = get_tools_from_kwargs(kwargs)

    merged_tools = existing_data.get("tools", {}) | request_tools

    traced_data = TracedData(
        start_time=existing_data.get("start_time", start_time),
        response_id=parsed_response.id,
        input=existing_data.get("input", kwargs.get("input")),
        instructions=existing_data.get("instructions", kwargs.get("instructions")),
        tools=merged_tools if merged_tools else None,
        output_blocks={block.id: block for block in parsed_response.output}
        | existing_data.get("output_blocks", {}),
        usage=existing_data.get("usage", parsed_response.usage),
        output_text=existing_data.get("output_text", parsed_response.output_text),
        request_model=existing_data.get("request_model", kwargs.get("model")),
        response_model=existing_data.get("response_model", parsed_response.model),
    )
    responses[parsed_response.id] = traced_data

    if parsed_response.status == "completed":
        span = tracer.start_span(
            "openai.responses",
            kind=SpanKind.CLIENT,
            start_time=int(traced_data.start_time),
        )
        set_data_attributes(traced_data, span)
        span.end()

    return response


@with_tracer_wrapper
def responses_cancel_wrapper(tracer: Tracer, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    response = wrapped(*args, **kwargs)
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        span = tracer.start_span(
            "openai.responses.create_background",
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
        )
        span.record_exception(Exception("Response cancelled"))
        set_data_attributes(existing_data, span)
        span.end()
    return response


@with_tracer_wrapper
async def async_responses_cancel_wrapper(
    tracer: Tracer, wrapped, instance, args, kwargs
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    response = await wrapped(*args, **kwargs)
    parsed_response = parse_response(response)
    existing_data = responses.pop(parsed_response.id, None)
    if existing_data is not None:
        span = tracer.start_span(
            "openai.responses.create_background",
            kind=SpanKind.CLIENT,
            start_time=existing_data.start_time,
            record_exception=True,
        )
        span.record_exception(Exception("Response cancelled"))
        set_data_attributes(existing_data, span)
        span.end()
    return response


# TODO: build streaming responses

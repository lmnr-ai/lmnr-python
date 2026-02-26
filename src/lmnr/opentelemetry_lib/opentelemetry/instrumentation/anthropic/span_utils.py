import json
import logging

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)

from lmnr.sdk.utils import json_dumps

from .utils import (
    _aextract_response_data,
    _extract_response_data,
    dont_throw,
    model_as_dict,
    set_span_attribute,
    should_send_prompts,
)

logger = logging.getLogger(__name__)


def _process_content_for_message(content):
    """Convert message content to a serializable format.

    For strings, returns as-is. For lists, converts each item via model_as_dict.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return [model_as_dict(item) for item in content]
    return content


@dont_throw
async def aset_input_attributes(span, kwargs):
    from .utils import set_span_attribute

    set_span_attribute(span, GEN_AI_REQUEST_MODEL, kwargs.get("model"))
    set_span_attribute(
        span, GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens_to_sample")
    )
    set_span_attribute(span, GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, "llm.is_streaming", kwargs.get("stream"))
    set_span_attribute(
        span, "anthropic.request.service_tier", kwargs.get("service_tier")
    )

    if should_send_prompts():
        if kwargs.get("prompt") is not None:
            # Legacy completions API
            messages = [{"role": "user", "content": kwargs.get("prompt")}]
            set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))

        elif kwargs.get("messages") is not None:
            messages = []

            # Add system message if present
            if kwargs.get("system"):
                system_content = _process_content_for_message(kwargs.get("system"))
                messages.append({"role": "system", "content": system_content})

            # Add all user/assistant messages
            for i, message in enumerate(kwargs.get("messages")):
                msg = dict(message)
                content = msg.get("content")
                if content is not None:
                    msg["content"] = _process_content_for_message(content)
                messages.append(msg)

            set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))

        if kwargs.get("tools") is not None:
            set_span_attribute(
                span,
                "gen_ai.tool.definitions",
                json_dumps(kwargs.get("tools")),
            )


def _build_output_from_response(response):
    """Build the output messages list from a non-streaming Anthropic response.

    Returns a list of content block dicts as returned by the Anthropic API,
    representing the single candidate/choice.
    """
    result = {
        "role": response.get("role", "assistant"),
        "content": [],
    }

    stop_reason = response.get("stop_reason")
    if stop_reason:
        result["stop_reason"] = stop_reason

    if response.get("completion"):
        result["content"].append(
            {
                "type": "text",
                "text": response.get("completion"),
            }
        )
    elif response.get("content"):
        for block in response.get("content"):
            result["content"].append(model_as_dict(block))

    return [result]


async def _aset_span_completions(span, response):
    if not should_send_prompts():
        return

    response = await _aextract_response_data(response)
    output = _build_output_from_response(response)
    set_span_attribute(span, "gen_ai.output.messages", json_dumps(output))


def _set_span_completions(span, response):
    if not should_send_prompts():
        return
    from .utils import set_span_attribute

    output = _build_output_from_response(response)
    set_span_attribute(span, "gen_ai.output.messages", json_dumps(output))


@dont_throw
async def aset_response_attributes(span, response):
    response = await _aextract_response_data(response)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    if usage := response.get("usage"):
        if hasattr(usage, "service_tier"):
            set_span_attribute(
                span,
                "anthropic.response.service_tier",
                usage.service_tier,
            )
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)

    await _aset_span_completions(span, response)


@dont_throw
def set_response_attributes(span, response):
    response = _extract_response_data(response)
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, response.get("model"))
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    if usage := response.get("usage"):
        if hasattr(usage, "service_tier"):
            set_span_attribute(
                span,
                "anthropic.response.service_tier",
                usage.service_tier,
            )
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)

    _set_span_completions(span, response)


@dont_throw
def set_streaming_response_attributes(span, complete_response_events):
    if not span or not span.is_recording() or not complete_response_events:
        return None

    # Build an output in the same format as non-streaming responses
    result = {
        "role": "assistant",
        "content": [],
    }

    finish_reason = None
    for event in complete_response_events:
        event_type = event.get("type")
        if event.get("finish_reason"):
            finish_reason = event.get("finish_reason")

        if event_type == "thinking":
            result["content"].append(
                {
                    "type": "thinking",
                    "thinking": event.get("text", ""),
                }
            )
        elif event_type == "tool_use":
            tool_input = event.get("input", "")
            # input may be a stringified JSON from streaming, try to parse
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except (json.JSONDecodeError, TypeError):
                    pass
            result["content"].append(
                {
                    "type": "tool_use",
                    "id": event.get("id", ""),
                    "name": event.get("name", ""),
                    "input": tool_input,
                }
            )
        else:
            # text block
            text = event.get("text", "")
            if text:
                result["content"].append(
                    {
                        "type": "text",
                        "text": text,
                    }
                )

    if finish_reason:
        result["stop_reason"] = finish_reason

    if should_send_prompts():
        set_span_attribute(span, "gen_ai.output.messages", json_dumps([result]))

    return result

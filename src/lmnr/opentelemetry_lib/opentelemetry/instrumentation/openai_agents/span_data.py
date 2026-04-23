"""Span data extraction and error handling for OpenAI Agents instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from agents.tracing import Span as AgentsSpan

    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan

from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.utils import json_dumps

from .helpers import (
    agent_name,
    export_span_data,
    get_current_system_instructions,
    span_kind,
)
from .messages import (
    apply_llm_attributes,
    response_to_llm_data,
    set_gen_ai_input_messages,
    set_gen_ai_output_messages,
    set_gen_ai_output_messages_from_response,
    set_lmnr_span_io,
    set_tool_definitions_from_response,
)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def apply_span_error(lmnr_span: LaminarSpan, span: AgentsSpan[Any]) -> None:
    error = getattr(span, "error", None)
    if not error:
        return
    try:
        message = getattr(error, "message", None) or str(error)
        lmnr_span.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Span data extraction
# ---------------------------------------------------------------------------


def apply_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    if span_data is None:
        return

    kind = span_kind(span_data)

    if kind == "agent":
        _apply_agent_span_data(lmnr_span, span_data)
    elif kind in {"function", "tool"}:
        _apply_function_span_data(lmnr_span, span_data)
    elif kind == "generation":
        _apply_generation_span_data(lmnr_span, span_data)
    elif kind == "response":
        _apply_response_span_data(lmnr_span, span_data)
    elif kind == "handoff":
        _apply_handoff_span_data(lmnr_span, span_data)
    elif kind == "guardrail":
        _apply_guardrail_span_data(lmnr_span, span_data)
    elif kind == "custom":
        _apply_custom_span_data(lmnr_span, span_data)
    elif kind in {"mcp_list_tools", "mcp_tools"}:
        _apply_mcp_span_data(lmnr_span, span_data)
    elif kind == "speech":
        _apply_speech_span_data(lmnr_span, span_data)
    elif kind == "transcription":
        _apply_transcription_span_data(lmnr_span, span_data)
    elif kind == "speech_group":
        _apply_speech_group_span_data(lmnr_span, span_data)
    else:
        # Fallback: try to set generic I/O
        data = export_span_data(span_data)
        set_lmnr_span_io(lmnr_span, data.get("input"), data.get("output"))


def _apply_agent_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    res_dict = {}
    name = data.get("name") or getattr(span_data, "name", None)
    if name:
        res_dict["name"] = name

    # Record handoffs and tools as metadata
    handoffs = data.get("handoffs")
    if handoffs:
        res_dict["handoffs"] = handoffs
    tools = data.get("tools")
    if tools:
        res_dict["tools"] = tools
    output_type = data.get("output_type")
    if output_type:
        res_dict["output_type"] = output_type

    set_lmnr_span_io(lmnr_span, res_dict, None)


def _apply_function_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    # Use gen_ai messages for input/output
    set_lmnr_span_io(lmnr_span, data.get("input"), data.get("output"))


def _apply_generation_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    """Handle 'generation' spans - these are LLM calls with input/output/usage."""
    data = export_span_data(span_data)

    # Set gen_ai.input.messages from the input messages
    input_data = data.get("input")
    if input_data is None:
        input_data = getattr(span_data, "input", None)
    set_gen_ai_input_messages(
        lmnr_span, input_data, system_instructions=get_current_system_instructions()
    )

    # Set gen_ai.output.messages from the output
    output_data = data.get("output")
    if output_data is None:
        output_data = getattr(span_data, "output", None)
    set_gen_ai_output_messages(lmnr_span, output_data)

    # Apply LLM attributes (model, usage, etc.) with fallback to direct attrs
    llm_data = dict(data)
    if llm_data.get("model") is None:
        llm_data["model"] = getattr(span_data, "model", None)
    if llm_data.get("usage") is None:
        llm_data["usage"] = getattr(span_data, "usage", None)
    if llm_data.get("response_id") is None and llm_data.get("id") is None:
        llm_data["response_id"] = getattr(span_data, "response_id", None)
    apply_llm_attributes(lmnr_span, llm_data)


def _apply_response_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    """Handle 'response' spans - these wrap the actual OpenAI API Response."""
    response = getattr(span_data, "response", None)
    response_input = getattr(span_data, "input", None)

    # Set gen_ai.input.messages, prepending the agent's system instructions
    # captured during the model call.
    set_gen_ai_input_messages(
        lmnr_span,
        response_input,
        system_instructions=get_current_system_instructions(),
    )

    # Set gen_ai.output.messages from the response output
    if response is not None:
        set_gen_ai_output_messages_from_response(lmnr_span, response)

        # Set tool definitions from response.tools
        set_tool_definitions_from_response(lmnr_span, response)

        # Apply LLM attributes from response
        apply_llm_attributes(lmnr_span, response_to_llm_data(response))


def _apply_handoff_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    from_agent = data.get("from_agent")
    to_agent = data.get("to_agent")
    if from_agent:
        lmnr_span.set_attribute("openai.agents.handoff.from", agent_name(from_agent))
    if to_agent:
        lmnr_span.set_attribute("openai.agents.handoff.to", agent_name(to_agent))


def _apply_guardrail_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    name = data.get("name")
    if name:
        lmnr_span.set_attribute("openai.agents.guardrail.name", name)
    triggered = data.get("triggered")
    if triggered is not None:
        lmnr_span.set_attribute("openai.agents.guardrail.triggered", triggered)


def _apply_custom_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    name = data.get("name")
    if name:
        lmnr_span.set_attribute("openai.agents.custom.name", name)
    custom_data = data.get("data")
    if custom_data is not None:
        lmnr_span.set_attribute("openai.agents.custom.data", json_dumps(custom_data))


def _apply_mcp_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    server = data.get("server")
    if server:
        lmnr_span.set_attribute("openai.agents.mcp.server", server)
    result = data.get("result")
    if result is not None:
        lmnr_span.set_attribute("openai.agents.mcp.result", json_dumps(result))


def _apply_speech_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    model = data.get("model") or getattr(span_data, "model", None)
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    input_text = data.get("input")
    if input_text is None:
        input_text = getattr(span_data, "input", None)
    if input_text:
        set_gen_ai_input_messages(lmnr_span, input_text)

    output_data = data.get("output")
    if output_data is None:
        output_data = getattr(span_data, "output", None)
    if output_data:
        if isinstance(output_data, dict):
            # Speech output is {data: ..., format: ...}
            set_gen_ai_output_messages(lmnr_span, output_data.get("data"))
        else:
            set_gen_ai_output_messages(lmnr_span, output_data)


def _apply_transcription_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    model = data.get("model") or getattr(span_data, "model", None)
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    input_data = data.get("input")
    if input_data is None:
        input_data = getattr(span_data, "input", None)
    if input_data:
        if isinstance(input_data, dict):
            set_gen_ai_input_messages(lmnr_span, input_data.get("data"))
        else:
            set_gen_ai_input_messages(lmnr_span, input_data)

    output_text = data.get("output")
    if output_text is None:
        output_text = getattr(span_data, "output", None)
    if output_text:
        set_gen_ai_output_messages(lmnr_span, output_text)


def _apply_speech_group_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    input_text = data.get("input")
    if input_text is None:
        input_text = getattr(span_data, "input", None)
    if input_text:
        set_gen_ai_input_messages(lmnr_span, input_text)

    output_text = data.get("output")
    if output_text is None:
        output_text = getattr(span_data, "output", None)
    if output_text:
        set_gen_ai_output_messages(lmnr_span, output_text)

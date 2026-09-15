"""Span data extraction and error handling for OpenAI Agents instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_STACKTRACE,
    EXCEPTION_TYPE,
)
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from agents.tracing import Span as AgentsSpan

    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan

from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.utils import json_dumps

from .helpers import (
    export_span_data,
    get_current_model_name,
    get_current_system_instructions,
    name_from_span_data,
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
        # `SpanError` is a TypedDict, so at runtime its fields are dict keys.
        # `message` is a short label, `data` carries the specifics.
        if isinstance(error, dict):
            label = error.get("message")
            data = error.get("data")
        else:
            label = getattr(error, "message", None)
            data = getattr(error, "data", None)
        label = label or str(error)

        # app-server derives a span's error status from the presence of an
        # `exception` event, never from the OTel status code. Nothing was
        # raised here, so wrap the payload and blank the synthetic stacktrace.
        lmnr_span.record_exception(
            Exception(json_dumps(data) if data else label),
            attributes={
                **get_event_attributes_from_context(),
                EXCEPTION_TYPE: label,
                EXCEPTION_STACKTRACE: "",
            },
        )
        lmnr_span.set_status(Status(StatusCode.ERROR, label))
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
    elif kind == "task":
        _apply_task_span_data(lmnr_span, span_data)
    elif kind == "turn":
        _apply_turn_span_data(lmnr_span, span_data)
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
    # FunctionSpanData.export() renders output as `str(self.output)`, which turns
    # a structured tool result into an unparseable Python repr and drops falsy
    # ones, so prefer the raw attributes and fall back to the export.
    data = export_span_data(span_data)
    input_data = getattr(span_data, "input", None)
    if input_data is None:
        input_data = data.get("input")
    output_data = getattr(span_data, "output", None)
    if output_data is None:
        output_data = data.get("output")
    set_lmnr_span_io(lmnr_span, input_data, output_data)


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
    else:
        # A failed call leaves no response, but the model was known before the
        # request, so the span can still be attributed to it.
        apply_llm_attributes(lmnr_span, {"model": get_current_model_name()})


def _apply_handoff_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    from_agent = data.get("from_agent")
    to_agent = data.get("to_agent")
    if from_agent:
        lmnr_span.set_attribute(
            "openai.agents.handoff.from", name_from_span_data(from_agent)
        )
    if to_agent:
        lmnr_span.set_attribute(
            "openai.agents.handoff.to", name_from_span_data(to_agent)
        )


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


def _exported_payload(span_data: Any) -> dict[str, Any]:
    """Fields of a span_data that exports as `{type: custom, data: {...}}`."""
    data = export_span_data(span_data).get("data")
    return data if isinstance(data, dict) else {}


def _apply_task_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    """Handle 'task' spans - one top-level Runner run."""
    data = _exported_payload(span_data)
    name = data.get("name") or getattr(span_data, "name", None)
    if name:
        lmnr_span.set_attribute("openai.agents.task.name", name)
    _apply_aggregate_usage(lmnr_span, "task", data, span_data)


def _apply_turn_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    """Handle 'turn' spans - one iteration of the agent loop."""
    data = _exported_payload(span_data)
    turn = data.get("turn")
    if turn is None:
        turn = getattr(span_data, "turn", None)
    if turn is not None:
        lmnr_span.set_attribute("openai.agents.turn.index", turn)
    agent_name = data.get("agent_name") or getattr(span_data, "agent_name", None)
    if agent_name:
        lmnr_span.set_attribute("openai.agents.turn.agent_name", agent_name)
    _apply_aggregate_usage(lmnr_span, "turn", data, span_data)


def _apply_aggregate_usage(
    lmnr_span: LaminarSpan, kind: str, data: dict[str, Any], span_data: Any
) -> None:
    """Roll-up usage for a wrapper span.

    Deliberately not `gen_ai.usage.*`: the wrapped LLM spans already report
    their own usage, and `gen_ai.*` would make the backend type this as an LLM
    call.
    """
    usage = data.get("usage")
    if usage is None:
        usage = getattr(span_data, "usage", None)
    if usage:
        lmnr_span.set_attribute(f"openai.agents.{kind}.usage", json_dumps(usage))


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

"""Span data extraction and error handling for OpenAI Agents instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from opentelemetry.semconv.attributes.exception_attributes import (
    EXCEPTION_STACKTRACE,
    EXCEPTION_TYPE,
)
from opentelemetry.trace import Status, StatusCode

from lmnr.sdk.log import get_default_logger

if TYPE_CHECKING:
    from agents.tracing import Span as AgentsSpan

    from lmnr.opentelemetry_lib.tracing.span import LaminarSpan

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.helpers import (
    export_span_data,
    get_current_model_name,
    get_current_system_instructions,
    name_from_span_data,
    span_kind,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai_agents.messages import (
    apply_llm_attributes,
    response_to_llm_data,
    set_gen_ai_input_messages,
    set_gen_ai_output_messages,
    set_gen_ai_output_messages_from_response,
    set_lmnr_span_io,
    set_tool_definitions_from_response,
)
from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def apply_span_error(lmnr_span: LaminarSpan, span: AgentsSpan[Any]) -> None:
    error = getattr(span, "error", None)
    if not error:
        return
    try:
        # `SpanError` is a TypedDict, so at runtime the fields are dict keys,
        # not attributes. `message` is a short label ("Error running tool
        # (non-fatal)"), `data` carries the specifics.
        if isinstance(error, dict):
            fields = cast("dict[str, Any]", error)
            raw_label = fields.get("message")
            data = fields.get("data")
            fallback = str(fields)
        else:
            raw_label = getattr(error, "message", None)
            data = getattr(error, "data", None)
            fallback = str(error)
        label = str(raw_label) if raw_label else fallback
        details = json_dumps(data) if data else label

        # app-server derives a span's error status from the presence of an
        # `exception` event, never from the OTel status code, so the status
        # below is invisible on its own. The Agents SDK reports failures as a
        # `SpanError` payload rather than a raised exception, so wrap it like
        # `openai/v1/responses_wrappers.py` does for cancelled responses, and
        # override `exception.type` so the trace view's error card is headlined
        # by the SDK's label instead of `Exception`.
        lmnr_span.record_exception(
            Exception(details),
            attributes={
                **get_event_attributes_from_context(),
                EXCEPTION_TYPE: label,
                # Nothing was raised here, so the stacktrace `record_exception`
                # derives from our wrapper is just `Exception: <details>`, which
                # the error card would show under "Stack trace" as a duplicate
                # of the message. Blank it rather than mislead.
                EXCEPTION_STACKTRACE: "",
            },
        )
        lmnr_span.set_status(Status(StatusCode.ERROR, label))
    except Exception:
        logger.debug("Failed to apply span error", exc_info=True)


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
    # a structured tool result into an unparseable Python repr, so prefer the
    # raw attributes and fall back to the export.
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
        lmnr_span.set_attribute("openai.agents.handoff.from", name_from_span_data(from_agent))
    if to_agent:
        lmnr_span.set_attribute("openai.agents.handoff.to", name_from_span_data(to_agent))


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
    return cast("dict[str, Any]", data) if isinstance(data, dict) else {}


def _apply_task_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    """Handle 'task' spans - one top-level Runner run."""
    data = _exported_payload(span_data)
    name = data.get("name") or getattr(span_data, "name", None)
    if name:
        lmnr_span.set_attribute("openai.agents.task.name", name)
    usage = _apply_aggregate_usage(lmnr_span, "task", data, span_data)
    set_lmnr_span_io(lmnr_span, {"name": name} if name else None, usage)


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
    usage = _apply_aggregate_usage(lmnr_span, "turn", data, span_data)
    turn_input = {
        key: value
        for key, value in (("turn", turn), ("agent_name", agent_name))
        if value is not None
    }
    set_lmnr_span_io(lmnr_span, turn_input or None, usage)


def _apply_aggregate_usage(
    lmnr_span: LaminarSpan,
    kind: str,
    data: dict[str, Any],
    span_data: Any,
) -> Any:
    """Roll-up usage for a wrapper span; returns it for the span output.

    Deliberately not `gen_ai.usage.*`: the wrapped LLM spans already report
    their own usage, and the backend would read a rollup there as one more LLM
    call's tokens.
    """
    usage = data.get("usage")
    if usage is None:
        usage = getattr(span_data, "usage", None)
    if not usage:
        return None
    lmnr_span.set_attribute(f"openai.agents.{kind}.usage", json_dumps(usage))
    return usage


def _apply_mcp_span_data(lmnr_span: LaminarSpan, span_data: Any) -> None:
    data = export_span_data(span_data)
    server = data.get("server")
    if server:
        lmnr_span.set_attribute("openai.agents.mcp.server", server)
    result = data.get("result")
    if result is not None:
        lmnr_span.set_attribute("openai.agents.mcp.result", json_dumps(result))
    # The attributes above are not rendered as span I/O, so without this the
    # span shows up blank in the trace view even though it listed tools fine.
    set_lmnr_span_io(
        lmnr_span,
        {"server": server} if server else None,
        result,
    )


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
            set_gen_ai_output_messages(lmnr_span, output_data.get("data"))  # pyright: ignore[reportUnknownMemberType]
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
            set_gen_ai_input_messages(lmnr_span, input_data.get("data"))  # pyright: ignore[reportUnknownMemberType]
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

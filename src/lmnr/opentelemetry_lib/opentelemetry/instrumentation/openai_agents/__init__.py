"""OpenTelemetry OpenAI Agents SDK instrumentation for Laminar."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import Status, StatusCode

logger = get_default_logger(__name__)

_instruments = ("openai-agents >= 0.7.0",)

def _suppress_inputs() -> bool:
    return os.getenv("LMNR_SUPPRESS_INPUTS") == "1"


def _suppress_outputs() -> bool:
    return os.getenv("LMNR_SUPPRESS_OUTPUTS") == "1"


def _capture_span_export() -> bool:
    return os.getenv("LMNR_OPENAI_AGENTS_CAPTURE_EXPORT") == "1"


@dataclass
class _TraceState:
    root_span: Any
    spans: Dict[str, Any] = field(default_factory=dict)


class LaminarAgentsTraceProcessor:
    """TracingProcessor implementation that mirrors OpenAI Agents spans into Laminar."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._traces: Dict[str, _TraceState] = {}

    def on_trace_start(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            return
        state = self._get_or_create_trace(trace)
        self._apply_trace_metadata(state.root_span, trace)

    def on_trace_end(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            return
        with self._lock:
            state = self._traces.pop(trace_id, None)
        if not state:
            return
        for span in list(state.spans.values()):
            try:
                span.end()
            except Exception:
                pass
        try:
            state.root_span.end()
        except Exception:
            pass

    def on_span_start(self, span: Any) -> None:
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return
        state = self._get_or_create_trace(span)
        parent_span = None
        parent_id = getattr(span, "parent_id", None)
        if parent_id:
            parent_span = state.spans.get(parent_id)
        if parent_span is None:
            parent_span = state.root_span

        parent_ctx = None
        if parent_span is not None and hasattr(parent_span, "get_laminar_span_context"):
            parent_ctx = parent_span.get_laminar_span_context()

        span_data = getattr(span, "span_data", None)
        span_type = _map_span_type(span_data)
        span_name = _span_name(span, span_data)

        lmnr_span = Laminar.start_span(
            span_name,
            span_type=span_type,
            parent_span_context=parent_ctx,
            tags=["openai-agents"],
        )
        if hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.span.type", _span_kind(span_data))
            span_id = getattr(span, "span_id", "")
            if span_id:
                lmnr_span.set_attribute("openai.agents.span.id", span_id)

        # Use span_id as key, fall back to span_name
        key = getattr(span, "span_id", None) or span_name
        with self._lock:
            state.spans[key] = lmnr_span

    def on_span_end(self, span: Any) -> None:
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return

        # Use consistent key lookup: try span_id first, then span_name
        span_id = getattr(span, "span_id", None)
        span_data = getattr(span, "span_data", None)
        span_name_key = _span_name(span, span_data)
        key = span_id or span_name_key

        with self._lock:
            state = self._traces.get(trace_id)
            lmnr_span = state.spans.pop(key, None) if state else None

        if not lmnr_span:
            return

        _apply_span_data(lmnr_span, span_data)
        _apply_span_error(lmnr_span, span)
        try:
            lmnr_span.end()
        except Exception:
            pass

    def shutdown(self) -> None:
        with self._lock:
            states = list(self._traces.values())
            self._traces.clear()
        for state in states:
            for span in list(state.spans.values()):
                try:
                    span.end()
                except Exception:
                    pass
            try:
                state.root_span.end()
            except Exception:
                pass
        try:
            Laminar.flush()
        except Exception:
            pass

    def force_flush(self) -> bool:
        try:
            return Laminar.flush()
        except Exception:
            return False

    def _get_or_create_trace(self, trace_or_span: Any) -> _TraceState:
        trace_id = getattr(trace_or_span, "trace_id", None)
        if not trace_id:
            trace_id = "unknown"
        with self._lock:
            state = self._traces.get(trace_id)
            if state is None:
                name = getattr(trace_or_span, "name", None) or "agents.trace"
                root_span = Laminar.start_span(
                    name,
                    tags=["openai-agents"],
                )
                state = _TraceState(root_span=root_span)
                self._traces[trace_id] = state
        return state

    def _apply_trace_metadata(self, root_span: Any, trace: Any) -> None:
        if not hasattr(root_span, "set_trace_metadata"):
            return
        metadata: Dict[str, Any] = {}
        trace_metadata = getattr(trace, "metadata", None)
        if isinstance(trace_metadata, dict):
            metadata.update(trace_metadata)
        trace_id = getattr(trace, "trace_id", None)
        if trace_id:
            metadata["openai.agents.trace_id"] = trace_id
        group_id = getattr(trace, "group_id", None)
        if group_id:
            metadata["openai.agents.group_id"] = group_id
        trace_name = getattr(trace, "name", None)
        if trace_name:
            metadata["openai.agents.trace_name"] = trace_name
        if metadata:
            try:
                root_span.set_trace_metadata(metadata)
            except Exception:
                pass
        session_id = metadata.get("session_id") if metadata else None
        user_id = metadata.get("user_id") if metadata else None
        if not session_id:
            session_id = os.getenv("LMNR_SESSION_ID")
        if not user_id:
            user_id = os.getenv("LMNR_USER_ID")
        if session_id and hasattr(root_span, "set_trace_session_id"):
            root_span.set_trace_session_id(session_id)
        if user_id and hasattr(root_span, "set_trace_user_id"):
            root_span.set_trace_user_id(user_id)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrumentor for the OpenAI Agents SDK tracing module."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        try:
            from agents.tracing import add_trace_processor
        except Exception:
            logger.debug("OpenAI Agents SDK not available; skipping instrumentation")
            return

        processor = LaminarAgentsTraceProcessor()
        try:
            add_trace_processor(processor)
            self._processor = processor
            logger.debug("Laminar OpenAI Agents trace processor registered")
        except Exception as exc:
            logger.warning("Failed to register Laminar Agents processor: %s", exc)

    def _uninstrument(self, **kwargs):
        processor = getattr(self, "_processor", None)
        if processor is None:
            return
        try:
            import agents.tracing as _tracing
            if hasattr(_tracing, "GLOBAL_TRACE_PROVIDER"):
                provider = _tracing.GLOBAL_TRACE_PROVIDER
                if hasattr(provider, "_multi_processor"):
                    mp = provider._multi_processor
                    if hasattr(mp, "_processors"):
                        mp._processors = [
                            p for p in mp._processors if p is not processor
                        ]
        except Exception:
            pass
        processor.shutdown()
        self._processor = None


# ---------------------------------------------------------------------------
# Span naming / type mapping helpers
# ---------------------------------------------------------------------------

def _span_name(span: Any, span_data: Any) -> str:
    name = getattr(span, "name", None)
    if name:
        return name
    kind = _span_kind(span_data)
    if kind:
        return f"agents.{kind}"
    return "agents.span"


def _span_kind(span_data: Any) -> str:
    if span_data is None:
        return ""
    return getattr(span_data, "type", "")


def _map_span_type(span_data: Any) -> str:
    kind = _span_kind(span_data)
    if kind in {"generation", "response", "transcription", "speech", "speech_group"}:
        return "LLM"
    if kind in {"function", "tool", "mcp_list_tools", "mcp_tools"}:
        return "TOOL"
    return "DEFAULT"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _apply_span_error(lmnr_span: Any, span: Any) -> None:
    error = getattr(span, "error", None)
    if not error or not hasattr(lmnr_span, "set_status"):
        return
    try:
        message = getattr(error, "message", None) or str(error)
        lmnr_span.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Span data extraction
# ---------------------------------------------------------------------------

def _apply_span_data(lmnr_span: Any, span_data: Any) -> None:
    if span_data is None:
        return

    if _capture_span_export() and hasattr(span_data, "export"):
        try:
            export = span_data.export()
            if hasattr(lmnr_span, "set_attribute"):
                lmnr_span.set_attribute(
                    "openai.agents.span.export", json_dumps(export)
                )
        except Exception:
            pass

    kind = _span_kind(span_data)

    if kind == "agent":
        _apply_agent_span_data(lmnr_span, span_data)
    elif kind == "function":
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
        data = _export_span_data(span_data)
        _set_gen_ai_messages(lmnr_span, data.get("input"), data.get("output"))


def _apply_agent_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    name = data.get("name") or getattr(span_data, "name", None)
    if name:
        lmnr_span.set_attribute("openai.agents.agent.name", name)

    # Record handoffs and tools as metadata
    handoffs = data.get("handoffs")
    if handoffs:
        lmnr_span.set_attribute(
            "openai.agents.agent.handoffs", json_dumps(handoffs)
        )
    tools = data.get("tools")
    if tools:
        lmnr_span.set_attribute(
            "openai.agents.agent.tools", json_dumps(tools)
        )
    output_type = data.get("output_type")
    if output_type:
        lmnr_span.set_attribute("openai.agents.agent.output_type", output_type)


def _apply_function_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    name = data.get("name") or getattr(span_data, "name", None)
    if name:
        lmnr_span.set_attribute("openai.agents.tool.name", name)

    # Use gen_ai messages for input/output
    _set_gen_ai_messages(lmnr_span, data.get("input"), data.get("output"))


def _apply_generation_span_data(lmnr_span: Any, span_data: Any) -> None:
    """Handle 'generation' spans - these are LLM calls with input/output/usage."""
    if not hasattr(lmnr_span, "set_attribute"):
        return

    data = _export_span_data(span_data)

    # Set gen_ai.input.messages from the input messages
    input_data = data.get("input") or getattr(span_data, "input", None)
    _set_gen_ai_input_messages(lmnr_span, input_data)

    # Set gen_ai.output.messages from the output
    output_data = data.get("output") or getattr(span_data, "output", None)
    _set_gen_ai_output_messages(lmnr_span, output_data)

    # Apply LLM attributes (model, usage, etc.) with fallback to direct attrs
    llm_data = dict(data)
    if not llm_data.get("model"):
        llm_data.setdefault("model", getattr(span_data, "model", None))
    if llm_data.get("usage") is None:
        llm_data.setdefault("usage", getattr(span_data, "usage", None))
    if not llm_data.get("response_id") and not llm_data.get("id"):
        llm_data.setdefault("response_id", getattr(span_data, "response_id", None))
    _apply_llm_attributes(lmnr_span, llm_data)


def _apply_response_span_data(lmnr_span: Any, span_data: Any) -> None:
    """Handle 'response' spans - these wrap the actual OpenAI API Response."""
    if not hasattr(lmnr_span, "set_attribute"):
        return

    response = getattr(span_data, "response", None)
    response_input = getattr(span_data, "input", None)

    # Set gen_ai.input.messages
    _set_gen_ai_input_messages(lmnr_span, response_input)

    # Set gen_ai.output.messages from the response output
    if response is not None:
        _set_gen_ai_output_messages_from_response(lmnr_span, response)

        # Set tool definitions from response.tools
        _set_tool_definitions_from_response(lmnr_span, response)

        # Apply LLM attributes from response
        _apply_llm_attributes(lmnr_span, _response_to_llm_data(response))


def _apply_handoff_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    from_agent = data.get("from_agent")
    to_agent = data.get("to_agent")
    if from_agent:
        lmnr_span.set_attribute(
            "openai.agents.handoff.from", _agent_name(from_agent)
        )
    if to_agent:
        lmnr_span.set_attribute(
            "openai.agents.handoff.to", _agent_name(to_agent)
        )


def _apply_guardrail_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    name = data.get("name")
    if name:
        lmnr_span.set_attribute("openai.agents.guardrail.name", name)
    triggered = data.get("triggered")
    if triggered is not None:
        lmnr_span.set_attribute("openai.agents.guardrail.triggered", triggered)


def _apply_custom_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    name = data.get("name")
    if name:
        lmnr_span.set_attribute("openai.agents.custom.name", name)
    custom_data = data.get("data")
    if custom_data is not None:
        lmnr_span.set_attribute(
            "openai.agents.custom.data", json_dumps(custom_data)
        )


def _apply_mcp_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    server = data.get("server")
    if server:
        lmnr_span.set_attribute("openai.agents.mcp.server", server)
    result = data.get("result")
    if result is not None:
        lmnr_span.set_attribute(
            "openai.agents.mcp.result", json_dumps(result)
        )


def _apply_speech_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    model = data.get("model") or getattr(span_data, "model", None)
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    input_text = data.get("input") or getattr(span_data, "input", None)
    if input_text and not _suppress_inputs():
        _set_gen_ai_input_messages(lmnr_span, input_text)

    output_data = data.get("output")
    if output_data and not _suppress_outputs():
        if isinstance(output_data, dict):
            # Speech output is {data: ..., format: ...}
            _set_gen_ai_output_messages(lmnr_span, output_data.get("data"))
        else:
            _set_gen_ai_output_messages(lmnr_span, output_data)


def _apply_transcription_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    model = data.get("model") or getattr(span_data, "model", None)
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    input_data = data.get("input")
    if input_data and not _suppress_inputs():
        if isinstance(input_data, dict):
            _set_gen_ai_input_messages(lmnr_span, input_data.get("data"))
        else:
            _set_gen_ai_input_messages(lmnr_span, input_data)

    output_text = data.get("output") or getattr(span_data, "output", None)
    if output_text and not _suppress_outputs():
        _set_gen_ai_output_messages(lmnr_span, output_text)


def _apply_speech_group_span_data(lmnr_span: Any, span_data: Any) -> None:
    data = _export_span_data(span_data)
    if not hasattr(lmnr_span, "set_attribute"):
        return

    input_text = data.get("input") or getattr(span_data, "input", None)
    if input_text and not _suppress_inputs():
        _set_gen_ai_input_messages(lmnr_span, input_text)


# ---------------------------------------------------------------------------
# gen_ai.input.messages / gen_ai.output.messages helpers
# ---------------------------------------------------------------------------

def _set_gen_ai_messages(
    lmnr_span: Any,
    input_data: Any,
    output_data: Any,
) -> None:
    """Set gen_ai.input.messages and gen_ai.output.messages on the span."""
    if input_data is not None and not _suppress_inputs():
        _set_gen_ai_input_messages(lmnr_span, input_data)
    if output_data is not None and not _suppress_outputs():
        _set_gen_ai_output_messages(lmnr_span, output_data)


def _set_gen_ai_input_messages(lmnr_span: Any, input_data: Any) -> None:
    """Set gen_ai.input.messages on the span."""
    if _suppress_inputs() or input_data is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    messages = _normalize_messages(input_data)
    if messages:
        lmnr_span.set_attribute("gen_ai.input.messages", json_dumps(messages))


def _set_gen_ai_output_messages(lmnr_span: Any, output_data: Any) -> None:
    """Set gen_ai.output.messages on the span."""
    if _suppress_outputs() or output_data is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    messages = _normalize_messages(output_data)
    if messages:
        lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(messages))


def _set_gen_ai_output_messages_from_response(
    lmnr_span: Any, response: Any
) -> None:
    """Extract and set gen_ai.output.messages from a Response object."""
    if _suppress_outputs() or response is None:
        return
    if not hasattr(lmnr_span, "set_attribute"):
        return

    output_items = getattr(response, "output", None)
    if not output_items:
        return

    messages: List[Dict[str, Any]] = []
    for item in output_items:
        item_dict = _model_as_dict(item)
        if not item_dict:
            continue
        item_type = item_dict.get("type")
        if item_type == "message":
            content_list = item_dict.get("content", [])
            text_parts = []
            for content in (content_list if isinstance(content_list, list) else []):
                if isinstance(content, dict):
                    ct = content.get("type", "")
                    if ct in ("output_text", "text"):
                        text_parts.append(content.get("text", ""))
                else:
                    ct = getattr(content, "type", "")
                    if ct in ("output_text", "text"):
                        text_parts.append(getattr(content, "text", ""))
            if text_parts:
                messages.append({
                    "role": item_dict.get("role", "assistant"),
                    "content": "".join(text_parts),
                })
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item_dict.get("call_id", item_dict.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item_dict.get("name", ""),
                        "arguments": item_dict.get("arguments", ""),
                    },
                }],
            })

    if messages:
        lmnr_span.set_attribute("gen_ai.output.messages", json_dumps(messages))


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _set_tool_definitions_from_response(lmnr_span: Any, response: Any) -> None:
    """Extract gen_ai.tool.definitions from a Response object's tools field."""
    if not hasattr(lmnr_span, "set_attribute"):
        return

    tools = getattr(response, "tools", None)
    if not tools:
        return

    tool_defs = []
    for tool in tools:
        tool_dict = _model_as_dict(tool)
        if not tool_dict:
            continue
        tool_type = tool_dict.get("type")
        if tool_type == "function":
            func_def = {}
            func_def["type"] = "function"
            function_info = tool_dict.get("function") or tool_dict
            func_def["function"] = {
                "name": function_info.get("name", ""),
            }
            desc = function_info.get("description")
            if desc:
                func_def["function"]["description"] = desc
            params = function_info.get("parameters")
            if params:
                func_def["function"]["parameters"] = params
            strict = function_info.get("strict")
            if strict is not None:
                func_def["function"]["strict"] = strict
            tool_defs.append(func_def)
        elif tool_type in ("web_search", "file_search", "code_interpreter",
                           "computer_use"):
            # Built-in tools, record their type
            tool_defs.append(tool_dict)
        else:
            # Other tool types (MCP, etc.)
            tool_defs.append(tool_dict)

    if tool_defs:
        lmnr_span.set_attribute("gen_ai.tool.definitions", json_dumps(tool_defs))


# ---------------------------------------------------------------------------
# LLM attributes (model, usage, response_id)
# ---------------------------------------------------------------------------

def _apply_llm_attributes(lmnr_span: Any, data: Optional[Dict[str, Any]]) -> None:
    if not data or not hasattr(lmnr_span, "set_attribute"):
        return

    model = data.get("model")
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    usage = data.get("usage")
    if usage is not None:
        _apply_usage(lmnr_span, usage)

    response_id = data.get("response_id") or data.get("id")
    if response_id:
        lmnr_span.set_attribute(Attributes.RESPONSE_ID.value, response_id)


def _apply_usage(lmnr_span: Any, usage: Any) -> None:
    """Extract token usage from a usage object or dict, handling zero correctly."""
    if usage is None:
        return

    if isinstance(usage, dict):
        input_tokens = _get_first_not_none(
            usage, "input_tokens", "prompt_tokens", "input"
        )
        output_tokens = _get_first_not_none(
            usage, "output_tokens", "completion_tokens", "output"
        )
        total_tokens = _get_first_not_none(
            usage, "total_tokens", "total"
        )
    else:
        # Object with attributes (e.g. ResponseUsage)
        input_tokens = _get_attr_not_none(usage, "input_tokens", "prompt_tokens")
        output_tokens = _get_attr_not_none(usage, "output_tokens", "completion_tokens")
        total_tokens = _get_attr_not_none(usage, "total_tokens")

    if input_tokens is not None:
        lmnr_span.set_attribute(Attributes.INPUT_TOKEN_COUNT.value, input_tokens)
    if output_tokens is not None:
        lmnr_span.set_attribute(Attributes.OUTPUT_TOKEN_COUNT.value, output_tokens)
    if total_tokens is not None:
        lmnr_span.set_attribute(Attributes.TOTAL_TOKEN_COUNT.value, total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        lmnr_span.set_attribute(
            Attributes.TOTAL_TOKEN_COUNT.value, input_tokens + output_tokens
        )


def _get_first_not_none(d: dict, *keys: str) -> Optional[int]:
    """Get the first key whose value is not None from a dict.

    Unlike using `or`, this correctly handles 0 as a valid value.
    """
    for key in keys:
        val = d.get(key)
        if val is not None:
            return val
    return None


def _get_attr_not_none(obj: Any, *attrs: str) -> Optional[int]:
    """Get the first attribute whose value is not None from an object.

    Unlike using `or`, this correctly handles 0 as a valid value.
    """
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    return None


def _response_to_llm_data(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    usage = getattr(response, "usage", None)
    # Convert usage to dict if it's an object
    usage_dict = None
    if usage is not None:
        if isinstance(usage, dict):
            usage_dict = usage
        elif hasattr(usage, "input_tokens"):
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    return {
        "model": getattr(response, "model", None),
        "usage": usage_dict,
        "response_id": getattr(response, "id", None),
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _export_span_data(span_data: Any) -> Dict[str, Any]:
    if span_data is None:
        return {}
    if hasattr(span_data, "export"):
        try:
            exported = span_data.export()
            if isinstance(exported, dict):
                return exported
        except Exception:
            return {}
    return {}


def _normalize_messages(data: Any) -> List[Dict[str, Any]]:
    """Normalize various input/output formats into a list of message dicts."""
    if data is None:
        return []

    if isinstance(data, str):
        return [{"role": "user", "content": data}]

    if isinstance(data, list):
        messages = []
        for item in data:
            if isinstance(item, dict):
                messages.append(item)
            elif hasattr(item, "model_dump"):
                try:
                    messages.append(item.model_dump())
                except Exception:
                    messages.append({"content": str(item)})
            else:
                item_dict = _model_as_dict(item)
                if item_dict:
                    messages.append(item_dict)
                else:
                    messages.append({"content": str(item)})
        return messages

    if isinstance(data, dict):
        return [data]

    # If it's a pydantic model or similar
    as_dict = _model_as_dict(data)
    if as_dict:
        return [as_dict]

    return [{"content": str(data)}]


def _model_as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Convert a pydantic model or similar object to a dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {
            k: v for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    return None


def _agent_name(agent: Any) -> str:
    if isinstance(agent, dict):
        return agent.get("name") or ""
    if isinstance(agent, str):
        return agent
    if hasattr(agent, "name"):
        return getattr(agent, "name") or ""
    return ""

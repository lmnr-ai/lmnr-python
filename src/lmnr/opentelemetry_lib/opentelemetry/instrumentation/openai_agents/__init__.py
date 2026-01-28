"""OpenTelemetry OpenAI Agents SDK instrumentation."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Optional

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.log import get_default_logger
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import Status, StatusCode

logger = get_default_logger(__name__)

_instruments = ("openai-agents >= 0.0.0",)

SUPPRESS_INPUTS = os.getenv("LMNR_SUPPRESS_INPUTS") == "1"
SUPPRESS_OUTPUTS = os.getenv("LMNR_SUPPRESS_OUTPUTS") == "1"
CAPTURE_SPAN_EXPORT = os.getenv("LMNR_OPENAI_AGENTS_CAPTURE_EXPORT") == "1"


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
            lmnr_span.set_attribute("openai.agents.span.id", getattr(span, "span_id", ""))
        with self._lock:
            state.spans[getattr(span, "span_id", span_name)] = lmnr_span

    def on_span_end(self, span: Any) -> None:
        trace_id = getattr(span, "trace_id", None)
        span_id = getattr(span, "span_id", None)
        if not trace_id or not span_id:
            return
        with self._lock:
            state = self._traces.get(trace_id)
            lmnr_span = state.spans.pop(span_id, None) if state else None
        if not lmnr_span:
            return
        span_data = getattr(span, "span_data", None)
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
            logger.debug("Laminar OpenAI Agents trace processor registered")
        except Exception as exc:
            logger.warning("Failed to register Laminar Agents processor: %s", exc)

    def _uninstrument(self, **kwargs):
        # The SDK does not currently expose a remove_trace_processor API.
        pass


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


def _apply_span_error(lmnr_span: Any, span: Any) -> None:
    error = getattr(span, "error", None)
    if not error or not hasattr(lmnr_span, "set_status"):
        return
    try:
        lmnr_span.set_status(Status(StatusCode.ERROR, str(error)))
    except Exception:
        pass


def _apply_span_data(lmnr_span: Any, span_data: Any) -> None:
    if span_data is None:
        return

    if CAPTURE_SPAN_EXPORT and hasattr(span_data, "export"):
        try:
            export = span_data.export()
            if hasattr(lmnr_span, "set_attribute"):
                lmnr_span.set_attribute("openai.agents.span.export", export)
        except Exception:
            pass

    kind = _span_kind(span_data)
    data = _export_span_data(span_data)

    if kind == "agent":
        _set_span_io(lmnr_span, data)
        agent = data.get("agent") if data else None
        if isinstance(agent, dict) and hasattr(lmnr_span, "set_attribute"):
            if agent.get("name"):
                lmnr_span.set_attribute("openai.agents.agent.name", agent.get("name"))
            if agent.get("id"):
                lmnr_span.set_attribute("openai.agents.agent.id", agent.get("id"))
    elif kind == "function":
        _set_span_io(lmnr_span, data)
        if data.get("name") and hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.tool.name", data.get("name"))
    elif kind == "generation":
        _set_span_input(lmnr_span, data.get("input") if data else None)
        output_items = data.get("output") if data else None
        output_text = _extract_output_text_from_items(output_items)
        _set_span_output(lmnr_span, output_text if output_text is not None else output_items)
        _apply_llm_attributes(lmnr_span, data)
    elif kind == "response":
        response = getattr(span_data, "response", None)
        response_input = getattr(span_data, "input", None)
        _set_span_input(lmnr_span, response_input)
        response_text = _extract_response_output(response)
        if response_text is not None:
            _set_span_output(lmnr_span, response_text)
        else:
            _set_span_output(lmnr_span, _serialize_response_output(response))
        _apply_llm_attributes(lmnr_span, _response_to_llm_data_from_response(response))
    elif kind == "handoff":
        if hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute(
                "openai.agents.handoff.from",
                _agent_name(data.get("from_agent") if data else None),
            )
            lmnr_span.set_attribute(
                "openai.agents.handoff.to",
                _agent_name(data.get("to_agent") if data else None),
            )
    elif kind == "guardrail":
        _set_span_io(lmnr_span, data)
        if hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.guardrail.name", data.get("name"))
            lmnr_span.set_attribute(
                "openai.agents.guardrail.triggered", data.get("triggered")
            )
    elif kind == "custom":
        if data and hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.custom.name", data.get("name"))
            lmnr_span.set_attribute("openai.agents.custom.data", data.get("data"))
    elif kind in {"mcp_list_tools", "mcp_tools"}:
        if data and hasattr(lmnr_span, "set_attribute"):
            lmnr_span.set_attribute("openai.agents.mcp.server", data.get("server"))
            lmnr_span.set_attribute("openai.agents.mcp.result", data.get("result"))
    else:
        _set_span_io(lmnr_span, data)


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


def _set_span_io(lmnr_span: Any, data: Optional[Dict[str, Any]]) -> None:
    if not data:
        return
    _set_span_input(lmnr_span, data.get("input"))
    _set_span_output(lmnr_span, data.get("output"))


def _set_span_input(lmnr_span: Any, payload: Any) -> None:
    if SUPPRESS_INPUTS:
        return
    if payload is None or not hasattr(lmnr_span, "set_input"):
        return
    try:
        lmnr_span.set_input(payload)
    except Exception:
        pass


def _set_span_output(lmnr_span: Any, payload: Any) -> None:
    if SUPPRESS_OUTPUTS:
        return
    if payload is None or not hasattr(lmnr_span, "set_output"):
        return
    try:
        lmnr_span.set_output(payload)
    except Exception:
        pass


def _apply_llm_attributes(lmnr_span: Any, data: Optional[Dict[str, Any]]) -> None:
    if not data or not hasattr(lmnr_span, "set_attribute"):
        return
    model = data.get("model")
    if model:
        lmnr_span.set_attribute(Attributes.REQUEST_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.RESPONSE_MODEL.value, model)
        lmnr_span.set_attribute(Attributes.PROVIDER.value, "openai")

    usage = data.get("usage") or {}
    if isinstance(usage, dict):
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("input")
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("output")
        )
        total_tokens = usage.get("total_tokens") or usage.get("total")
        if input_tokens is not None:
            lmnr_span.set_attribute(Attributes.INPUT_TOKEN_COUNT.value, input_tokens)
        if output_tokens is not None:
            lmnr_span.set_attribute(Attributes.OUTPUT_TOKEN_COUNT.value, output_tokens)
        if total_tokens is not None:
            lmnr_span.set_attribute(Attributes.TOTAL_TOKEN_COUNT.value, total_tokens)

    response_id = data.get("response_id") or data.get("id")
    if response_id:
        lmnr_span.set_attribute(Attributes.RESPONSE_ID.value, response_id)


def _response_to_llm_data_from_response(response: Any) -> Dict[str, Any]:
    if response is None:
        return {}
    return {
        "model": getattr(response, "model", None),
        "usage": getattr(response, "usage", None),
        "response_id": getattr(response, "id", None),
    }


def _extract_response_output(response: Any) -> Optional[str]:
    if response is None:
        return None
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    return _extract_output_text_from_items(getattr(response, "output", None))


def _extract_output_text_from_items(items: Any) -> Optional[str]:
    if not isinstance(items, list):
        return None

    parts: list[str] = []
    for item in items:
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "message":
            continue
        content_list = getattr(item, "content", None)
        if content_list is None and isinstance(item, dict):
            content_list = item.get("content")
        if not isinstance(content_list, list):
            continue
        for content in content_list:
            content_type = getattr(content, "type", None)
            if content_type is None and isinstance(content, dict):
                content_type = content.get("type")
            if content_type not in {"output_text", "text"}:
                continue
            text = getattr(content, "text", None)
            if text is None and isinstance(content, dict):
                text = content.get("text")
            if text:
                parts.append(text)

    return "".join(parts) if parts else None


def _serialize_response_output(response: Any) -> Optional[list[Any]]:
    if response is None:
        return None
    outputs = getattr(response, "output", None)
    if not isinstance(outputs, list):
        return None
    serialized: list[Any] = []
    for item in outputs:
        if hasattr(item, "model_dump"):
            serialized.append(item.model_dump())
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            serialized.append(str(item))
    return serialized if serialized else None


def _agent_name(agent: Any) -> str:
    if isinstance(agent, dict) and agent.get("name"):
        return agent.get("name")
    if hasattr(agent, "name"):
        return getattr(agent, "name")
    return ""

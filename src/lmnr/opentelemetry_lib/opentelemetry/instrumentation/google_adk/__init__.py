"""OpenTelemetry Google ADK instrumentation.

Google ADK instruments itself with OpenTelemetry: its module-level tracer
(``gcp.vertex.agent``) is a ``ProxyTracer``, so once ``Laminar.initialize()``
registers the global tracer provider, every ADK span (``invocation``,
``invoke_agent {name}``, ``call_llm``, ``execute_tool {name}``) already lands
in Laminar. What this instrumentor adds on top of that raw wiring:

- ``execute_tool`` spans get ``lmnr.span.type = TOOL`` plus span input/output,
  so they render as tool calls instead of ``Default`` spans. Input and output
  are copied from the ``gcp.vertex.agent.tool_call_args`` /
  ``gcp.vertex.agent.tool_response`` attributes ADK just stamped, which means
  ADK's own content toggle (``run_config.telemetry`` / env var) is inherited:
  when ADK redacts content to ``"{}"``, we don't set input/output at all.
- ``invoke_agent`` spans carry the ADK session id and user id as Laminar
  association properties, so traces group by ADK session out of the box.
- ADK skips its own ``generate_content {model}`` span when it detects an
  external google-genai instrumentation, but its detector walks
  ``__wrapped__`` chains looking for the otel-contrib package's filename and
  doesn't recognize Laminar's google-genai wrapper. We extend the detector to
  also report Laminar's instrumentation, so each LLM call is covered by a
  single span (Laminar's, with token counts and cost) instead of two.

All wraps are defensive: a hook that is missing in the installed ADK version
is skipped, and a failure inside a wrapper never breaks the traced call.
"""

import contextlib
import logging
from typing import Any, Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SESSION_ID,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
    USER_ID,
)

logger = logging.getLogger(__name__)

_TRACING_MODULE = "google.adk.telemetry.tracing"
_TELEMETRY_PACKAGE = "google.adk.telemetry"
_FLOW_FUNCTIONS_MODULE = "google.adk.flows.llm_flows.functions"
_TOOL_ARGS_ATTRIBUTE = "gcp.vertex.agent.tool_call_args"
_TOOL_RESPONSE_ATTRIBUTE = "gcp.vertex.agent.tool_response"
_GENAI_DETECTOR = (
    "_instrumented_with_opentelemetry_instrumentation_google_genai"
)


def _resolve_span(
    args: tuple, kwargs: dict, position: int
) -> trace.Span | None:
    """The ADK trace_* functions accept the target span as an optional
    argument and fall back to the current span, mirror that resolution."""
    span = kwargs.get("span")
    if span is None and len(args) > position:
        span = args[position]
    if span is None:
        span = trace.get_current_span()
    if span is None or not span.is_recording():
        return None
    return span


def _copy_content_attribute(
    span: trace.Span, source: str, target: str, actual_is_empty: bool = False
) -> None:
    """Copies an ADK content attribute onto a Laminar one, unless ADK's
    content toggle redacted it, which ADK signals by stamping ``"{}"``.
    That sentinel collides with a genuinely empty dict serialized under an
    enabled toggle, so the caller passes ``actual_is_empty`` computed from
    the real value: a ``"{}"`` for an actually-empty payload is content,
    not redaction."""
    attributes = getattr(span, "attributes", None)
    if not attributes:
        return
    value = attributes.get(source)
    if not isinstance(value, str) or not value:
        return
    if value == "{}" and not actual_is_empty:
        return
    span.set_attribute(target, value)


def _tool_call_args(args: tuple, kwargs: dict):
    if "args" in kwargs:
        return kwargs["args"]
    if len(args) > 1:
        return args[1]
    return None


def _tool_response(args: tuple, kwargs: dict):
    """Extracts the tool response the same way trace_tool_call does:
    ``function_response_event.content.parts[0].function_response.response``."""
    event = kwargs.get("function_response_event")
    if event is None and len(args) > 2:
        event = args[2]
    try:
        return event.content.parts[0].function_response.response
    except (AttributeError, IndexError, TypeError):
        return None


def _wrap_trace_tool_call(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        # `span` is the 5th positional parameter of trace_tool_call.
        span = _resolve_span(args, kwargs, position=4)
        if span is None:
            return result
        span.set_attribute(SPAN_TYPE, "TOOL")
        _copy_content_attribute(
            span,
            _TOOL_ARGS_ATTRIBUTE,
            SPAN_INPUT,
            actual_is_empty=_tool_call_args(args, kwargs) == {},
        )
        _copy_content_attribute(
            span,
            _TOOL_RESPONSE_ATTRIBUTE,
            SPAN_OUTPUT,
            actual_is_empty=_tool_response(args, kwargs) == {},
        )
    except Exception:
        logger.debug("Failed to enrich ADK tool span", exc_info=True)
    return result


def _wrap_trace_merged_tool_calls(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        # trace_merged_tool_calls always stamps the current span.
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return result
        span.set_attribute(SPAN_TYPE, "TOOL")
        _copy_content_attribute(span, _TOOL_RESPONSE_ATTRIBUTE, SPAN_OUTPUT)
    except Exception:
        logger.debug("Failed to enrich ADK merged tool span", exc_info=True)
    return result


def _wrap_trace_agent_invocation(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        span = _resolve_span(args, kwargs, position=0)
        if span is None:
            return result
        ctx = kwargs.get("ctx") if "ctx" in kwargs else None
        if ctx is None and len(args) > 2:
            ctx = args[2]
        session = getattr(ctx, "session", None)
        session_id = getattr(session, "id", None)
        if session_id:
            span.set_attribute(
                f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}", str(session_id)
            )
        user_id = getattr(session, "user_id", None)
        if user_id:
            span.set_attribute(
                f"{ASSOCIATION_PROPERTIES}.{USER_ID}", str(user_id)
            )
    except Exception:
        logger.debug("Failed to enrich ADK agent span", exc_info=True)
    return result


def _laminar_genai_instrumented() -> bool:
    try:
        from ..google_genai import GoogleGenAiSdkInstrumentor

        return GoogleGenAiSdkInstrumentor().is_instrumented_by_opentelemetry
    except Exception:
        return False


@contextlib.contextmanager
def _noop_context():
    yield


def _wrap_use_extra_generate_content_attributes(
    wrapped, instance, args, kwargs
):
    """When ADK delegates the LLM span to an external genai instrumentation,
    it forwards agent/session attributes through a context key imported from
    the otel-contrib google-genai package. If that package isn't installed,
    ADK warns that it is "installed but has insufficient version" — misleading
    when the active instrumentation is Laminar's, and repeated on every LLM
    call. Skip the forwarding quietly in that case; without the contrib
    context key the attributes have nowhere to go anyway."""
    try:
        from opentelemetry.instrumentation.google_genai import (  # noqa: F401
            GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
        )
    except (ImportError, AttributeError):
        if _laminar_genai_instrumented():
            return _noop_context()
    return wrapped(*args, **kwargs)


def _wrap_genai_detection(wrapped, instance, args, kwargs):
    if wrapped(*args, **kwargs):
        return True
    return _laminar_genai_instrumented()


class GoogleAdkInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()
        self._wrapped_functions: list[tuple[str, str]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        # The wrapped functions (trace_tool_call, trace_agent_invocation,
        # trace_merged_tool_calls, the genai detector) all exist with
        # compatible signatures since 2.0.0; validated against 2.7.1. The
        # upper bound guards against a 3.x rework of the telemetry module.
        return ("google-adk >= 2.0.0, < 3.0.0",)

    def _instrument(self, **kwargs: Any):
        # `trace_merged_tool_calls` is bound by name at import time in
        # `flows.llm_flows.functions` (and re-exported from the `telemetry`
        # package), so patching only the `tracing` module attribute would
        # miss the call site whenever ADK's flow modules were imported before
        # `initialize()` — the usual ordering. Patch every module that holds
        # a binding; a call routed through two patched bindings just runs the
        # idempotent enrichment twice.
        for module_name, function_name, wrapper in (
            (_TRACING_MODULE, "trace_tool_call", _wrap_trace_tool_call),
            (
                _TRACING_MODULE,
                "trace_merged_tool_calls",
                _wrap_trace_merged_tool_calls,
            ),
            (
                _FLOW_FUNCTIONS_MODULE,
                "trace_merged_tool_calls",
                _wrap_trace_merged_tool_calls,
            ),
            (
                _TELEMETRY_PACKAGE,
                "trace_merged_tool_calls",
                _wrap_trace_merged_tool_calls,
            ),
            (
                _TRACING_MODULE,
                "trace_agent_invocation",
                _wrap_trace_agent_invocation,
            ),
            (_TRACING_MODULE, _GENAI_DETECTOR, _wrap_genai_detection),
            (
                _TRACING_MODULE,
                "_use_extra_generate_content_attributes",
                _wrap_use_extra_generate_content_attributes,
            ),
        ):
            try:
                wrap_function_wrapper(module_name, function_name, wrapper)
                self._wrapped_functions.append((module_name, function_name))
            except (AttributeError, ModuleNotFoundError):
                logger.debug(
                    "ADK telemetry hook %s.%s not found, skipping",
                    module_name,
                    function_name,
                )

    def _uninstrument(self, **kwargs: Any):
        import importlib

        for module_name, function_name in self._wrapped_functions:
            try:
                unwrap(importlib.import_module(module_name), function_name)
            except Exception:
                pass
        self._wrapped_functions = []

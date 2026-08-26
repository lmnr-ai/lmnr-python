"""OpenTelemetry Google ADK instrumentation.

ADK instruments itself: its module-level tracer (``gcp.vertex.agent``) is a
``ProxyTracer``, so every ADK span already lands in Laminar once
``initialize()`` registers the global tracer provider. This instrumentor only
enriches that wiring. Tool spans get ``lmnr.span.type = TOOL`` and their
input/output, copied from the attributes ADK itself stamps so that ADK's
content toggle keeps working. Agent spans get the ADK session/user ids as
association properties. ADK's detection of an external google-genai
instrumentation is extended to recognize this SDK's wrapper, which makes ADK
skip its own ``generate_content`` span; without that, every LLM call is
spanned twice (lmnr-ai/lmnr#2234).

Wraps are defensive: hooks missing from the installed ADK version are
skipped, and a wrapper failure never breaks the traced call.
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
    """Mirrors ADK's own resolution: explicit span argument, else current."""
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
    """Copies an ADK content attribute onto a Laminar one.

    ADK stamps ``"{}"`` when its content toggle redacts a payload, but an
    actually empty dict serializes to ``"{}"`` too, so the caller tells the
    two apart via ``actual_is_empty``."""
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
        # An id already on the span was set explicitly through Laminar;
        # the derived ADK id must not override it (context < parent <
        # explicit, per laminar.py).
        existing = getattr(span, "attributes", None) or {}
        session_key = f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"
        session_id = getattr(session, "id", None)
        if session_id and session_key not in existing:
            span.set_attribute(session_key, str(session_id))
        user_key = f"{ASSOCIATION_PROPERTIES}.{USER_ID}"
        user_id = getattr(session, "user_id", None)
        if user_id and user_key not in existing:
            span.set_attribute(user_key, str(user_id))
    except Exception:
        logger.debug("Failed to enrich ADK agent span", exc_info=True)
    return result


def _laminar_genai_instrumented() -> bool:
    try:
        from ..google_genai import GoogleGenAiSdkInstrumentor

        # Constructing the singleton here would rerun its __init__ on
        # every LLM call, resetting a user-provided Config.exception_logger.
        instance = GoogleGenAiSdkInstrumentor._instance
        return bool(instance and instance.is_instrumented_by_opentelemetry)
    except Exception:
        return False


@contextlib.contextmanager
def _noop_context():
    yield


def _wrap_use_extra_generate_content_attributes(
    wrapped, instance, args, kwargs
):
    """ADK forwards agent/session attributes to a delegated genai span
    through a context key imported from the otel-contrib package, and logs a
    bogus "insufficient version" warning on every LLM call when that package
    is missing. If Laminar's genai instrumentation is the active one, skip
    the forwarding instead of warning; without the contrib key the
    attributes have nowhere to go anyway."""
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
    # Not set in __init__: BaseInstrumentor is a singleton whose __init__
    # runs again on every construction, which would clear the tracking while
    # the wraps stay applied and leave _uninstrument with nothing to unwrap.
    _wrapped_functions: list[tuple[str, str, Any]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        # All wrapped hooks exist with compatible signatures since 2.0.0;
        # validated against 2.7.1. The upper bound guards against a 3.x
        # rework of the telemetry module.
        return ("google-adk >= 2.0.0, < 3.0.0",)

    def _instrument(self, **kwargs: Any):
        self._wrapped_functions = []
        # trace_merged_tool_calls is bound by name at import time in
        # flows.llm_flows.functions (and re-exported from the telemetry
        # package), so patching only the tracing module misses that call
        # site whenever ADK gets imported before initialize(), which is the
        # usual ordering. Patch every module holding a binding; a call that
        # goes through two patched bindings repeats the idempotent
        # enrichment, which is harmless. Order matters: the tracing module
        # goes last, otherwise a lazy import of the other modules would
        # bind the already-wrapped function and come out nested.
        for module_name, function_name, wrapper in (
            (_TRACING_MODULE, "trace_tool_call", _wrap_trace_tool_call),
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
                self._wrapped_functions.append(
                    (module_name, function_name, wrapper)
                )
            except (AttributeError, ModuleNotFoundError):
                logger.debug(
                    "ADK telemetry hook %s.%s not found, skipping",
                    module_name,
                    function_name,
                )

    def _uninstrument(self, **kwargs: Any):
        import importlib

        for module_name, function_name, wrapper in self._wrapped_functions:
            try:
                module = importlib.import_module(module_name)
                # Not a plain unwrap: a binding created from an
                # already-wrapped name can be nested.
                while (
                    getattr(
                        getattr(module, function_name, None),
                        "_self_wrapper",
                        None,
                    )
                    is wrapper
                ):
                    unwrap(module, function_name)
            except Exception:
                pass
        self._wrapped_functions = []

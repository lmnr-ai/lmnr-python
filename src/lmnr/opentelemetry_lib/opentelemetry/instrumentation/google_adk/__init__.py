"""OpenTelemetry Google ADK instrumentation.

ADK instruments itself: its module-level tracer (``gcp.vertex.agent``) is a
``ProxyTracer``, so every ADK span already lands in Laminar once
``initialize()`` registers the global tracer provider. This instrumentor only
enriches that wiring. Tool spans get ``lmnr.span.type = TOOL`` and their
input/output, copied from the attributes ADK itself stamps so that ADK's
content toggle keeps working. Agent spans get the ADK session/user ids as
association properties.

ADK's own ``call_llm`` span is enriched directly from the real
``LlmRequest``/``LlmResponse`` objects into ``gen_ai.input.messages`` /
``gen_ai.tool.definitions`` / ``gen_ai.output.messages`` /
``gen_ai.response.model`` — the shape the frontend already parses for every
other provider — instead of ADK's own JSON-blob
``gcp.vertex.agent.llm_request``/``llm_response`` attributes. Because of
this, ``Instruments.GOOGLE_GENAI`` is auto-removed from the default set
whenever ``google-adk`` is installed (see ``_GOOGLE_ADK_GENAI_CONFLICTS`` in
``tracing/instruments.py``): wrapping the genai SDK's own
``generate_content`` on top would double-cover the same call with a span
whose parent depends on ADK's fragile generator-scoped context (its
``call_llm`` span can outlive the ``yield`` that hands control back to the
caller, so whether it's still "current" by the time the SDK issues the HTTP
call depends on task/thread hops inside the SDK's own transport). The
``call_llm`` wrap also detaches the span from the ambient OTel context right
after stamping it, so that immediately-following tool execution attaches as
a sibling instead of nesting under ``call_llm`` (contextvars persist across
an async generator's ``yield`` within the same task, so without this ADK's
own ``with`` block keeps ``call_llm`` "current" through the postprocessing
that runs tools). For the same reason, the wrap also ends ``call_llm`` as
soon as the model's last token arrives (a non-partial ``LlmResponse``)
rather than letting its recorded duration stretch across the tool
execution/callback postprocessing that follows before ADK's own ``with``
block naturally unwinds; ADK's later call to ``span.end()`` is a harmless
no-op. ADK's own detection of an external genai instrumentation
is unconditionally forced to report one present, so ADK always skips its
native ``generate_content <model>`` span — it would otherwise double-cover
the call, either against our own ``call_llm`` enrichment (the default) or
against Laminar's ``google_genai`` span (the explicit-opt-in case where a
caller re-enables ``GOOGLE_GENAI`` alongside ADK) (lmnr-ai/lmnr#2234).

Wraps are defensive: hooks missing from the installed ADK version are
skipped, and a wrapper failure never breaks the traced call.
"""

import contextlib
import logging
from typing import Any, Collection

from opentelemetry import context as context_api
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
from lmnr.sdk.utils import json_dumps

logger = logging.getLogger(__name__)

_TRACING_MODULE = "google.adk.telemetry.tracing"
_TELEMETRY_PACKAGE = "google.adk.telemetry"
_FLOW_FUNCTIONS_MODULE = "google.adk.flows.llm_flows.functions"
_BASE_LLM_FLOW_MODULE = "google.adk.flows.llm_flows.base_llm_flow"
_TOOL_ARGS_ATTRIBUTE = "gcp.vertex.agent.tool_call_args"
_TOOL_RESPONSE_ATTRIBUTE = "gcp.vertex.agent.tool_response"
_LLM_REQUEST_ATTRIBUTE = "gcp.vertex.agent.llm_request"
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


def _tool_declarations_from_config(config) -> list:
    """Flattens `config.tools` (a list of `types.Tool` and/or dicts shaped
    like `{"function_declarations": [...]}`) into a flat list of function
    declarations, mirroring the flattening the google_genai instrumentor does
    for its own `gen_ai.tool.definitions` attribute."""
    from google.genai import types

    declarations = []
    for tool in getattr(config, "tools", None) or []:
        if isinstance(tool, types.Tool):
            declarations.extend(tool.function_declarations or [])
        elif isinstance(tool, dict) and isinstance(
            tool.get("function_declarations"), list
        ):
            declarations.extend(tool.get("function_declarations", []))
    return declarations


def _enrich_call_llm_span(span: trace.Span, llm_request, llm_response) -> None:
    """Stamps `gen_ai.*` attributes onto ADK's own `call_llm` span from the
    real `LlmRequest`/`LlmResponse` objects, so the span carries the same
    shape the frontend already parses for every other provider, instead of
    the raw `gcp.vertex.agent.llm_request`/`llm_response` JSON blobs ADK
    stamps for its own legacy UI."""
    span.set_attribute(SPAN_TYPE, "LLM")
    attributes = getattr(span, "attributes", None) or {}
    if attributes.get(_LLM_REQUEST_ATTRIBUTE) == "{}":
        # ADK's own content toggle redacted the request/response (see
        # trace_call_llm's should_add_content_to_legacy_spans branch); keep
        # the new gen_ai.* attributes redacted too rather than leaking
        # message content through a side door.
        return

    from ..google_genai.utils import content_union_to_dict, to_dict

    config = getattr(llm_request, "config", None)

    messages = []
    system_instruction = getattr(config, "system_instruction", None)
    if system_instruction:
        msg = content_union_to_dict(system_instruction, default_role="system")
        msg["role"] = "system"
        messages.append(msg)
    for content in getattr(llm_request, "contents", None) or []:
        messages.append(content_union_to_dict(content))
    if messages:
        span.set_attribute("gen_ai.input.messages", json_dumps(messages))

    declarations = _tool_declarations_from_config(config)
    if declarations:
        span.set_attribute(
            "gen_ai.tool.definitions",
            json_dumps([to_dict(declaration) for declaration in declarations]),
        )

    response_content = getattr(llm_response, "content", None)
    output_messages = (
        [content_union_to_dict(response_content, default_role="model")]
        if response_content is not None
        else []
    )
    span.set_attribute("gen_ai.output.messages", json_dumps(output_messages))

    model_version = getattr(llm_response, "model_version", None)
    if model_version:
        span.set_attribute("gen_ai.response.model", model_version)


def _detach_from_current_context(span: trace.Span) -> None:
    """Ends `call_llm`'s reign as the ambient "current span" the moment its
    attributes are stamped, instead of letting it stay attached — via ADK's
    still-open `start_as_current_span` block — through the tool-execution
    postprocessing that immediately follows. Without this, every tool span
    nests under `call_llm` instead of following it as a sibling.

    Deliberately an unbalanced `attach()` with no matching `detach()`: when
    ADK's own `with tracer.start_as_current_span('call_llm')` block
    eventually exits, its token-based detach restores the context to
    whatever was current when call_llm was attached, regardless of what
    happened to the context in between — contextvars.Token.reset() restores
    the exact value captured at attach time, not "whatever is current now".
    """
    parent = getattr(span, "parent", None)
    new_span = (
        trace.NonRecordingSpan(parent)
        if parent is not None
        else trace.INVALID_SPAN
    )
    # Pass the current context explicitly rather than None: omitting it
    # would build a brand new empty Context, discarding Laminar's own
    # association-properties/debug-context state carried in context vars.
    new_context = trace.set_span_in_context(new_span, context_api.get_current())
    context_api.attach(new_context)


def _wrap_trace_call_llm(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    # trace_call_llm(invocation_context, event_id, llm_request, llm_response,
    # span=None) — span is the 5th positional parameter.
    span = _resolve_span(args, kwargs, position=4)
    if span is None:
        return result
    llm_request = kwargs.get("llm_request")
    if llm_request is None and len(args) > 2:
        llm_request = args[2]
    llm_response = kwargs.get("llm_response")
    if llm_response is None and len(args) > 3:
        llm_response = args[3]
    try:
        _enrich_call_llm_span(span, llm_request, llm_response)
    except Exception:
        logger.debug("Failed to enrich ADK call_llm span", exc_info=True)

    # Streaming turns run this hook once per chunk with `llm_response`
    # marked `partial=True` for every fragment but the last (see
    # `StreamingResponseAggregator` in `google/adk/models/google_llm.py`).
    # Detaching or ending call_llm on a partial chunk would be premature:
    # the call hasn't actually finished, so a later `trace_call_llm` call
    # for the same turn — falling back to `trace.get_current_span()` in
    # `_resolve_span` if ADK ever omits the explicit span argument — would
    # resolve to call_llm's parent instead, and enrich/end that span rather
    # than call_llm.
    if getattr(llm_response, "partial", False):
        return result

    try:
        _detach_from_current_context(span)
    except Exception:
        logger.debug(
            "Failed to detach ADK call_llm span from context", exc_info=True
        )
    try:
        # Ends call_llm as soon as the model's last token arrives, instead
        # of letting its recorded duration bleed into the tool-execution
        # and after-model-callback postprocessing that follows inside
        # ADK's still-open `with tracer.start_as_current_span('call_llm')`
        # block. Safe even though that block will call `span.end()` again
        # later when it finally unwinds: `Span.end()` is idempotent — a
        # second call just logs "Calling end() on an ended span." and
        # returns, without re-exporting the span.
        span.end()
    except Exception:
        logger.debug(
            "Failed to end ADK call_llm span early", exc_info=True
        )
    return result


@contextlib.contextmanager
def _noop_context():
    yield


def _wrap_use_extra_generate_content_attributes(
    wrapped, instance, args, kwargs
):
    """ADK forwards agent/session attributes to a delegated genai span
    through a context key imported from the otel-contrib package, and logs a
    bogus "insufficient version" warning on every LLM call when that package
    is missing. This wrap only ever runs on the branch where ADK's own
    native span is already suppressed (`_wrap_genai_detection` below always
    reports an external genai instrumentation while this instrumentor is
    active), so there is never a delegated span for the forwarded attributes
    to reach — skip the forwarding instead of warning unconditionally,
    rather than gating it on whether Laminar's own google_genai
    instrumentor happens to be instrumented too."""
    try:
        from opentelemetry.instrumentation.google_genai import (  # noqa: F401
            GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
        )
    except (ImportError, AttributeError):
        return _noop_context()
    return wrapped(*args, **kwargs)


def _wrap_genai_detection(wrapped, instance, args, kwargs):
    """ADK's own native `generate_content <model>` span is always redundant
    while this instrumentor is active: by default `call_llm` is enriched
    directly (GOOGLE_GENAI is auto-removed from the default set — see
    `_GOOGLE_ADK_GENAI_CONFLICTS`), and if a caller explicitly opts
    GOOGLE_GENAI back in alongside ADK, Laminar's own google_genai span
    already covers the call. Either way, tell ADK an external genai
    instrumentation is present so it skips its native span."""
    return True


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
            # trace_call_llm is bound by name at import time in
            # base_llm_flow (and re-exported from the telemetry package),
            # same reasoning as trace_merged_tool_calls above: patch every
            # module holding a binding, tracing module last.
            (
                _BASE_LLM_FLOW_MODULE,
                "trace_call_llm",
                _wrap_trace_call_llm,
            ),
            (
                _TELEMETRY_PACKAGE,
                "trace_call_llm",
                _wrap_trace_call_llm,
            ),
            (
                _TRACING_MODULE,
                "trace_call_llm",
                _wrap_trace_call_llm,
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

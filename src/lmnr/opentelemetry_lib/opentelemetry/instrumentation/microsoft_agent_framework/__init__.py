"""OpenTelemetry Microsoft Agent Framework instrumentation.

Microsoft Agent Framework (`agent-framework` on PyPI, imported as
`agent_framework`) instruments itself: `agent_framework.observability` emits
OTel GenAI semconv spans (`invoke_agent <name>`, `chat <model>`,
`execute_tool <name>`, workflow spans) through a module-level `get_tracer()`.
Out of the box those spans already land in Laminar once `initialize()`
registers the global tracer provider, but a few gaps remain, and this
instrumentor closes them:

1. Instrumentation itself is OFF by default in older framework releases
   (`ENABLE_INSTRUMENTATION`), so we turn it on unless the user set it.
2. Message content is OFF by default in the framework ("sensitive data").
   Without it `chat` spans carry no prompts/completions and `execute_tool`
   spans carry no arguments/results. We turn it on, unless the user set
   `ENABLE_SENSITIVE_DATA` explicitly or disabled Laminar content tracing with
   `LMNR_TRACE_CONTENT=false`.
3. The framework activates its spans only in the global OTel context, while
   Laminar's own spans (`@observe`, the provider instrumentors) parent off
   Laminar's isolated context. So an `@observe` function called from a tool
   attached to the outer `@observe` root instead of the `execute_tool` span.
   We mirror every framework span activation into the isolated context.
4. The framework's `chat` span IS the LLM span (model, usage, messages), and
   the provider SDK call underneath it (OpenAI, Anthropic, ...) would be
   traced a second time by Laminar's provider instrumentors, double-counting
   tokens and cost. Instead of removing the provider instrumentors globally
   (which would also drop direct provider calls made outside the framework),
   we suppress instrumentation only while a framework `chat` / `embeddings`
   span is active.
5. The framework stamps `gen_ai.provider.name` (current semconv) but not the
   legacy `gen_ai.system` that Laminar's backend reads for cost lookup, so we
   copy it over on `chat` / `embeddings` spans.
6. Tool definitions (`gen_ai.tool.definitions`) are only recorded on the
   `invoke_agent` span, because `chat` span attributes are built from the
   client kwargs, not the request options. We carry the request's tools over
   to the `chat` span so the LLM span shows which tools the model was offered.

`get_tracer()` is also routed to Laminar's tracer provider so the spans are
exported even with `set_global_tracer_provider=False`, and the tracer copies
Laminar's association properties (session id, user id, metadata, trace type)
into the context each span starts in, so they are stamped on framework spans
like on any other Laminar span.

All hooks are module-level helpers in `agent_framework.observability` that
the framework looks up by global name at call time, so patching the module
attribute is enough. Hooks missing from the installed version are skipped,
and a hook failure never breaks the framework call.
"""

import inspect
import os
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, Collection, Iterator

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_USER_ID_KEY,
    get_current_context,
    pop_span_context,
    push_span_context,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_OBSERVABILITY_MODULE = "agent_framework.observability"
_MCP_MODULE = "agent_framework._mcp"

_OPERATION_NAME = "gen_ai.operation.name"
_PROVIDER_NAME = "gen_ai.provider.name"
_SYSTEM = "gen_ai.system"
_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
# Operations whose span already represents the model call. The provider SDK
# call made underneath is suppressed so it is not traced a second time.
_MODEL_CALL_OPERATIONS = ("chat", "embeddings")

# Tools of the chat request currently building its span attributes. Set only
# for the synchronous part of `ChatTelemetryLayer.get_response`, which is
# where the framework computes the `chat` span attributes.
_request_tools: ContextVar[Any] = ContextVar("lmnr_maf_request_tools", default=None)


def _is_model_call_span(span: Any) -> bool:
    attributes = getattr(span, "attributes", None) or {}
    try:
        return attributes.get(_OPERATION_NAME) in _MODEL_CALL_OPERATIONS
    except Exception:
        return False


@contextmanager
def _laminar_activation(span: Any, suppress_providers: bool) -> Iterator[None]:
    """Make `span` current in Laminar's isolated context for the block, and
    optionally suppress provider instrumentation in the global context."""
    pushed = False
    suppress_token = None
    try:
        if isinstance(span, trace.Span) and span.get_span_context().is_valid:
            push_span_context(trace.set_span_in_context(span, get_current_context()))
            pushed = True
        if suppress_providers:
            suppress_token = context_api.attach(
                context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
            )
    except Exception:
        logger.debug("Failed to activate Agent Framework span", exc_info=True)
    try:
        yield
    finally:
        try:
            if suppress_token is not None:
                context_api.detach(suppress_token)
            if pushed:
                pop_span_context()
        except Exception:
            logger.debug("Failed to deactivate Agent Framework span", exc_info=True)


@contextmanager
def _bridge_span_cm(cm: Any, span: Any = None) -> Iterator[Any]:
    """Enter a framework context manager that activates a span, and mirror the
    activation into Laminar's context. `span` is taken from the context
    manager's value when not given (`_get_span`, `start_as_current_span`)."""
    with cm as value:
        target = span if span is not None else value
        with _laminar_activation(target, _is_model_call_span(target)):
            yield value


def _wrap_span_cm(wrapped, instance, args, kwargs):
    return _bridge_span_cm(wrapped(*args, **kwargs))


def _wrap_activate_span(wrapped, instance, args, kwargs):
    span = kwargs.get("span", args[0] if args else None)
    return _bridge_span_cm(wrapped(*args, **kwargs), span=span)


def _wrap_chat_get_response(wrapped, instance, args, kwargs):
    tools = None
    try:
        options = kwargs.get("options")
        if isinstance(options, dict):
            tools = options.get("tools")
    except Exception:
        pass
    if not tools:
        return wrapped(*args, **kwargs)
    token = _request_tools.set(tools)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _request_tools.reset(token)


def _wrap_get_span_attributes(wrapped, instance, args, kwargs):
    attributes = wrapped(*args, **kwargs)
    if not isinstance(attributes, dict):
        return attributes
    operation = attributes.get(_OPERATION_NAME)
    try:
        if (
            operation in _MODEL_CALL_OPERATIONS
            and attributes.get(_PROVIDER_NAME)
            and _SYSTEM not in attributes
        ):
            attributes[_SYSTEM] = attributes[_PROVIDER_NAME]
    except Exception:
        logger.debug("Failed to set gen_ai.system", exc_info=True)
    try:
        tools = _request_tools.get()
        if operation == "chat" and tools and _TOOL_DEFINITIONS not in attributes:
            # Serialize the tools the way the framework does for the agent
            # span (its helpers are private and renamed across releases).
            definitions = wrapped(options={"tools": tools}).get(_TOOL_DEFINITIONS)
            if definitions:
                attributes[_TOOL_DEFINITIONS] = definitions
    except Exception:
        logger.debug("Failed to set gen_ai.tool.definitions", exc_info=True)
    return attributes


class _LaminarActivatingTracer(trace.Tracer):
    """Tracer handed to the framework: spans come from Laminar's tracer
    provider, and `start_as_current_span` also activates the span in
    Laminar's context (tool and workflow spans go through this path)."""

    def __init__(self, tracer: trace.Tracer):
        self._tracer = tracer

    def start_span(self, *args: Any, **kwargs: Any) -> trace.Span:
        return self._tracer.start_span(*args, **_with_association_context(args, kwargs))

    @contextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[trace.Span]:
        kwargs = _with_association_context(args, kwargs)
        cm = self._tracer.start_as_current_span(*args, **kwargs)
        with _bridge_span_cm(cm) as span:
            yield span


_ASSOCIATION_KEYS = (
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_USER_ID_KEY,
    CONTEXT_METADATA_KEY,
    CONTEXT_TRACE_TYPE_KEY,
)


def _with_association_context(args: tuple, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy Laminar's association properties into the span's start context.

    Laminar's span processor reads session id, user id, metadata and trace type
    from the context a span starts in. The framework starts its spans in the
    global context, where `Laminar.set_trace_session_id()` and friends are not
    visible, so they are copied over from Laminar's isolated context."""
    if len(args) > 1:
        # `context` passed positionally: leave the call untouched.
        return kwargs
    try:
        laminar_context = get_current_context()
        start_context = kwargs.get("context") or context_api.get_current()
        changed = False
        for key in _ASSOCIATION_KEYS:
            value = context_api.get_value(key, laminar_context)
            if value is not None and context_api.get_value(key, start_context) is None:
                start_context = context_api.set_value(key, value, start_context)
                changed = True
        if changed:
            return {**kwargs, "context": start_context}
    except Exception:
        logger.debug(
            "Failed to propagate Laminar association properties", exc_info=True
        )
    return kwargs


def _should_enable_instrumentation() -> bool:
    # Off by default in older framework releases. An explicit
    # `ENABLE_INSTRUMENTATION` from the user wins, in either direction.
    return os.getenv("ENABLE_INSTRUMENTATION") is None


def _should_enable_sensitive_data() -> bool:
    # The user's explicit framework setting wins, in either direction.
    if os.getenv("ENABLE_SENSITIVE_DATA") is not None:
        return False
    return (os.getenv("LMNR_TRACE_CONTENT") or "true").lower() == "true"


class MicrosoftAgentFrameworkInstrumentor(BaseInstrumentor):
    _wrapped: list[tuple[str, str]] = []
    # Framework settings we changed, with their previous values.
    _previous_settings: dict[str, bool] = {}

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("agent-framework-core >= 1.0.0, < 2.0.0",)

    def _instrument(self, **kwargs: Any):
        import agent_framework.observability as observability

        tracer_provider = kwargs.get("tracer_provider")

        def _wrap_get_tracer(wrapped, instance, args, call_kwargs):
            tracer = None
            if tracer_provider is not None:
                try:
                    # The framework calls `get_tracer()` with no arguments and
                    # relies on its own defaults (name, version), which the
                    # SDK provider's `get_tracer` does not have.
                    bound = inspect.signature(wrapped).bind(*args, **call_kwargs)
                    bound.apply_defaults()
                    tracer = tracer_provider.get_tracer(*bound.args, **bound.kwargs)
                except Exception:
                    logger.debug("Failed to get Laminar tracer", exc_info=True)
            if tracer is None:
                tracer = wrapped(*args, **call_kwargs)
            return _LaminarActivatingTracer(tracer)

        self._wrapped = []
        for module, name, wrapper in (
            (_OBSERVABILITY_MODULE, "get_tracer", _wrap_get_tracer),
            (_OBSERVABILITY_MODULE, "_get_span", _wrap_span_cm),
            (_OBSERVABILITY_MODULE, "_activate_span", _wrap_activate_span),
            (_OBSERVABILITY_MODULE, "_get_span_attributes", _wrap_get_span_attributes),
            (
                _OBSERVABILITY_MODULE,
                "ChatTelemetryLayer.get_response",
                _wrap_chat_get_response,
            ),
            # Imported by name into `_mcp`, so patch that binding too.
            (_OBSERVABILITY_MODULE, "create_mcp_client_span", _wrap_span_cm),
            (_MCP_MODULE, "create_mcp_client_span", _wrap_span_cm),
        ):
            try:
                wrap_function_wrapper(module, name, wrapper)
                self._wrapped.append((module, name))
            except (AttributeError, ImportError, ModuleNotFoundError):
                logger.debug("Agent Framework hook %s.%s not found", module, name)

        self._previous_settings = {}
        for setting, should_enable in (
            ("enable_instrumentation", _should_enable_instrumentation),
            ("enable_sensitive_data", _should_enable_sensitive_data),
        ):
            try:
                settings = observability.OBSERVABILITY_SETTINGS
                if should_enable() and not getattr(settings, setting):
                    self._previous_settings[setting] = False
                    # Recent releases ignore this while the user has called
                    # `disable_instrumentation()`, so that opt-out still wins.
                    setattr(settings, setting, True)
            except Exception:
                logger.debug("Failed to set Agent Framework %s", setting, exc_info=True)

    def _uninstrument(self, **kwargs: Any):
        import importlib

        for module, name in self._wrapped:
            try:
                owner = importlib.import_module(module)
                *parents, attribute = name.split(".")
                for parent in parents:
                    owner = getattr(owner, parent)
                unwrap(owner, attribute)
            except Exception:
                pass
        self._wrapped = []

        try:
            from agent_framework.observability import OBSERVABILITY_SETTINGS

            for setting, value in self._previous_settings.items():
                setattr(OBSERVABILITY_SETTINGS, setting, value)
        except Exception:
            pass
        self._previous_settings = {}

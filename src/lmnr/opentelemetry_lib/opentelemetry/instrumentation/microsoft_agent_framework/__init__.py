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
7. Gemini reports thinking tokens separately from candidate tokens, and the
   framework's Gemini client maps only the candidates to
   `gen_ai.usage.output_tokens`. Gemini bills thinking as output, so we add
   the thinking tokens to the output count on Gemini `chat` spans (as
   Laminar's google-genai instrumentation does) and record them under
   `gen_ai.usage.reasoning_tokens`, the key the backend reads.
8. Workflow spans are named with `OtelAttr` members (a `str` enum), which
   OTel rejects inside sequence attributes, dropping `lmnr.span.path` for the
   workflow and everything under it. Span names are coerced to plain `str`.
9. The MCP `tools/call` client span carries `gen_ai.operation.name =
   execute_tool` (per the MCP semconv), so Laminar shows it as a second tool
   span named after the tool, nested in the framework's own `execute_tool`
   span. We drop the operation name from MCP client spans so it shows as
   `tools/call <tool>` with its MCP attributes.

`get_tracer()` is also routed to Laminar's tracer provider so the spans are
exported even with `set_global_tracer_provider=False`, and the tracer copies
Laminar's association properties (session id, user id, metadata, trace type)
into the context each span starts in, so they are stamped on framework spans
like on any other Laminar span. The tracer also stamps
`lmnr.span.instrumentation_scope.{name,version}` (the installed
`agent-framework-core` version) on every framework span.

All hooks are module-level helpers in `agent_framework.observability` that
the framework looks up by global name at call time, so patching the module
attribute is enough. Hooks missing from the installed version are skipped,
and a hook failure never breaks the framework call.
"""

import inspect
import os
from collections.abc import Callable, Collection, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.metadata import version
from typing import Any, cast

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeVar, override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
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
T = TypeVar("T")

_OBSERVABILITY_MODULE = "agent_framework.observability"
_MCP_MODULE = "agent_framework._mcp"

_OPERATION_NAME = "gen_ai.operation.name"
_PROVIDER_NAME = "gen_ai.provider.name"
_SYSTEM = "gen_ai.system"
_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"
# `gen_ai.provider.name` prefix of the framework's Gemini clients ("gcp.gemini").
_GEMINI_PROVIDER_PREFIX = "gcp."
_SCOPE_NAME = "lmnr.span.instrumentation_scope.name"
_SCOPE_VERSION = "lmnr.span.instrumentation_scope.version"
# Operations whose span already represents the model call. The provider SDK
# call made underneath is suppressed so it is not traced a second time.
_MODEL_CALL_OPERATIONS = ("chat", "embeddings")

# Tools of the chat request currently building its span attributes. Set only
# for the synchronous part of `ChatTelemetryLayer.get_response`, which is
# where the framework computes the `chat` span attributes.
_request_tools: ContextVar[Any] = ContextVar("lmnr_maf_request_tools", default=None)


def _is_model_call_span(span: trace.Span) -> bool:
    attributes: dict[str, AttributeValue] = getattr(span, "attributes", None) or {}
    try:
        return attributes.get(_OPERATION_NAME) in _MODEL_CALL_OPERATIONS
    except Exception:
        return False


@contextmanager
def _laminar_activation(span: Any, suppress_providers: bool) -> Generator[None]:
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
def _bridge_span_cm(cm: Any, span: trace.Span | None = None) -> Generator[Any]:
    """Enter a framework context manager that activates a span, and mirror the
    activation into Laminar's context. `span` is taken from the context
    manager's value when not given (`_get_span`, `start_as_current_span`)."""
    with cm as value:
        target = span if span is not None else value
        with _laminar_activation(target, _is_model_call_span(target)):
            yield value


def _wrap_span_cm(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    return _bridge_span_cm(wrapped(*args, **kwargs))


def _wrap_mcp_client_span(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    try:
        attributes = kwargs.get("attributes")
        if isinstance(attributes, dict) and _OPERATION_NAME in attributes:
            attributes = cast(dict[str, AttributeValue], attributes)
            kwargs = {
                **kwargs,
                "attributes": {
                    k: v for k, v in attributes.items() if k != _OPERATION_NAME
                },
            }
    except Exception:
        logger.debug("Failed to strip MCP span operation name", exc_info=True)
    return _bridge_span_cm(wrapped(*args, **kwargs))


def _wrap_activate_span(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    span = kwargs.get("span", args[0] if args else None)
    return _bridge_span_cm(wrapped(*args, **kwargs), span=span)


def _wrap_chat_get_response(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    tools = None
    try:
        options = kwargs.get("options")
        if isinstance(options, dict):
            options = cast(dict[str, Any], options)
            tools = options.get("tools")
    except Exception:
        logger.debug("Failed to get tools from MAF request options", exc_info=True)
    if not tools:
        return wrapped(*args, **kwargs)
    token = _request_tools.set(tools)
    try:
        return wrapped(*args, **kwargs)
    finally:
        _request_tools.reset(token)


def _wrap_get_span_attributes(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., dict[str, AttributeValue] | T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> dict[str, AttributeValue] | T:
    attributes = wrapped(*args, **kwargs)
    if not isinstance(attributes, dict):
        return attributes
    attributes = cast(dict[str, AttributeValue], attributes)
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
            attributes_tools = cast(dict[str, AttributeValue], wrapped(options={"tools": tools}))
            definitions = attributes_tools.get(_TOOL_DEFINITIONS)
            if definitions:
                attributes[_TOOL_DEFINITIONS] = definitions
    except Exception:
        logger.debug("Failed to set gen_ai.tool.definitions", exc_info=True)
    return attributes


def _wrap_get_response_attributes(
    _to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    attributes = cast(dict[str, AttributeValue], wrapped(*args, **kwargs))
    try:
        _add_gemini_reasoning_tokens(attributes, args, kwargs)
    except Exception:
        logger.debug("Failed to add Gemini reasoning tokens", exc_info=True)
    return cast(Any, attributes)


def _add_gemini_reasoning_tokens(attributes: Any, args: Sequence[Any], kwargs: dict[str, Any]) -> None:
    """Count Gemini thinking tokens as output tokens on `chat` spans.

    Read from the response's usage details rather than the span attributes:
    the framework drops `gen_ai.usage.reasoning.output_tokens` when the
    experimental GenAI semconv is off."""
    if (
        not isinstance(attributes, dict)
        or attributes.get(_OPERATION_NAME) != "chat"
        # Already applied to this attribute dict.
        or _REASONING_TOKENS in attributes
        or not str(attributes.get(_PROVIDER_NAME) or "").startswith(
            _GEMINI_PROVIDER_PREFIX
        )
        or kwargs.get("capture_usage") is False
    ):
        return
    attributes = cast(dict[str, AttributeValue], attributes)
    response = kwargs.get("response", args[1] if len(args) > 1 else None)
    usage: dict[str, int | dict[str, int]] = getattr(response, "usage_details", None) or {}
    reasoning = usage.get("reasoning_output_token_count")
    output = attributes.get(_OUTPUT_TOKENS)
    if (
        not isinstance(reasoning, int)
        or isinstance(reasoning, bool)
        or reasoning <= 0
        or not isinstance(output, int)
    ):
        return
    attributes[_OUTPUT_TOKENS] = output + reasoning
    attributes[_REASONING_TOKENS] = reasoning


def _with_plain_name(args: Sequence[Any], kwargs: dict[str, Any]) -> tuple[tuple[Any], dict[str, Any]]:
    """Coerce a `str` subclass span name (the framework's `OtelAttr` enum) to
    a plain `str`. Laminar copies span names into the `lmnr.span.path`
    sequence attribute, and OTel drops sequences holding non-`str` items."""
    if args and isinstance(args[0], str) and type(args[0]) is not str:
        args = (str.__str__(args[0]), *args[1:])
    name = kwargs.get("name")
    if isinstance(name, str) and type(name) is not str:
        kwargs = {**kwargs, "name": str.__str__(name)}
    return cast(tuple[Any], args), kwargs


class _LaminarActivatingTracer(trace.Tracer):
    """Tracer handed to the framework: spans come from Laminar's tracer
    provider, are stamped with the instrumentation scope, and
    `start_as_current_span` also activates the span in Laminar's context
    (tool and workflow spans go through this path)."""

    def __init__(
        self, tracer: trace.Tracer, scope: LaminarInstrumentationScopeAttributes
    ):
        self._tracer: trace.Tracer = tracer
        self._scope: LaminarInstrumentationScopeAttributes = scope

    def _stamp_scope(self, span: trace.Span) -> None:
        span.set_attribute(_SCOPE_NAME, self._scope["name"])
        span.set_attribute(_SCOPE_VERSION, self._scope["version"])

    @override
    def start_span(self, *args: Any, **kwargs: Any) -> trace.Span:
        args, kwargs = _with_plain_name(args, kwargs)
        span = self._tracer.start_span(*args, **_with_association_context(args, kwargs))
        self._stamp_scope(span)
        return span

    @contextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Generator[trace.Span]:  # pyright: ignore[reportIncompatibleMethodOverride]
        args, kwargs = _with_plain_name(args, kwargs)
        kwargs = _with_association_context(args, kwargs)
        cm = self._tracer.start_as_current_span(*args, **kwargs)
        with _bridge_span_cm(cm) as span:
            self._stamp_scope(span)
            yield span


_ASSOCIATION_KEYS = (
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_USER_ID_KEY,
    CONTEXT_METADATA_KEY,
    CONTEXT_TRACE_TYPE_KEY,
)


def _with_association_context(args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
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


# (package_name, object_name, method_name, wrapper_function)
_WRAPPED_FUNCTIONS = (
    (_OBSERVABILITY_MODULE, None, "_get_span", _wrap_span_cm),
    (_OBSERVABILITY_MODULE, None, "_activate_span", _wrap_activate_span),
    (_OBSERVABILITY_MODULE, None, "_get_span_attributes", _wrap_get_span_attributes),
    (
        _OBSERVABILITY_MODULE,
        None,
        "_get_response_attributes",
        _wrap_get_response_attributes,
    ),
    (
        _OBSERVABILITY_MODULE,
        "ChatTelemetryLayer",
        "get_response",
        _wrap_chat_get_response,
    ),
    # Imported by name into `_mcp`, so patch that binding too. `_mcp` goes
    # first: importing it after the `observability` binding is wrapped would
    # copy the wrapper, and the second patch would wrap it twice.
    (_MCP_MODULE, None, "create_mcp_client_span", _wrap_mcp_client_span),
    (_OBSERVABILITY_MODULE, None, "create_mcp_client_span", _wrap_mcp_client_span),
)


class MicrosoftAgentFrameworkInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None
    _tracer_provider: Any = None
    # Framework settings we changed, with their previous values.
    _previous_settings: dict[str, bool]

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return ("agent-framework-core >= 1.0.0, < 2.0.0",)

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                framework_version = version("agent-framework-core")
            except Exception:
                framework_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="agent-framework", version=framework_version
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self._previous_settings = {}
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name=_OBSERVABILITY_MODULE,
                    method_name="get_tracer",
                    is_async=False,
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=self._wrap_get_tracer,
                ),
                *(
                    WrappedFunctionSpec(
                        package_name=package_name,
                        object_name=object_name,
                        method_name=method_name,
                        is_async=False,
                        instrumentation_scope=self.instrumentation_scope(),
                        wrapper_function=wrapper_function,
                    )
                    for package_name, object_name, method_name, wrapper_function in (
                        _WRAPPED_FUNCTIONS
                    )
                ),
            ]
        )

    def _wrap_get_tracer(self,
        to_wrap: WrappedFunctionSpec,
        wrapped: Callable[..., trace.Tracer],
        _instance: Any,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> _LaminarActivatingTracer:
        tracer = None
        if self._tracer_provider is not None:
            try:
                # The framework calls `get_tracer()` with no arguments and
                # relies on its own defaults (name, version), which the SDK
                # provider's `get_tracer` does not have.
                bound = inspect.signature(wrapped).bind(*args, **kwargs)
                bound.apply_defaults()
                tracer = self._tracer_provider.get_tracer(*bound.args, **bound.kwargs)
            except Exception:
                logger.debug("Failed to get Laminar tracer", exc_info=True)
        if tracer is None:
            tracer = wrapped(*args, **kwargs)
        return _LaminarActivatingTracer(tracer, cast(Any, to_wrap.get("instrumentation_scope")))

    @override
    def _instrument(self, **kwargs: Any):
        import agent_framework.observability as observability

        self._tracer_provider = kwargs.get("tracer_provider")
        super()._instrument(**kwargs)

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

    @override
    def _uninstrument(self, **kwargs: Any):
        super()._uninstrument(**kwargs)
        self._tracer_provider = None

        try:
            from agent_framework.observability import OBSERVABILITY_SETTINGS

            for setting, value in self._previous_settings.items():
                setattr(OBSERVABILITY_SETTINGS, setting, value)
        except Exception:
            logger.debug("Failed to uninstrument MAF", exc_info=True)
        self._previous_settings = {}

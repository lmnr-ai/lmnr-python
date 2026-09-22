import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, cast

from opentelemetry import trace
from opentelemetry.context import Context, create_key, get_value, set_value
from typing_extensions import override

from lmnr.opentelemetry_lib.tracing.attributes import (
    METADATA,
    SESSION_ID,
    TRACE_TYPE,
    USER_ID,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import MetadataMemberType, TraceType

logger = get_default_logger(__name__)


class _IsolatedRuntimeContext(ABC):
    """The isolated RuntimeContext interface, identical to OpenTelemetry's _RuntimeContext
    but isolated from the global context.
    """

    @abstractmethod
    def attach(self, context: Context) -> Token[Context]:
        """Sets the current `Context` object. Returns a
        token that can be used to reset to the previous `Context`.

        Args:
            context: The Context to set.
        """

    @abstractmethod
    def get_current(self) -> Context:
        """Returns the current `Context` object."""

    @abstractmethod
    def detach(self, token: Token[Context]) -> None:
        """Resets Context to a previous value

        Args:
            token: A reference to a previous Context.
        """


class IsolatedContextVarsRuntimeContext(_IsolatedRuntimeContext):
    """An isolated implementation of the RuntimeContext interface which wraps ContextVar
    but uses its own ContextVar instead of the global one.
    """

    def __init__(self) -> None:
        self._current_context: ContextVar[Context] = ContextVar("isolated_current_context")
        _set_token = self._current_context.set(Context())

    @override
    def attach(self, context: Context) -> Token[Context]:
        """Sets the current `Context` object. Returns a
        token that can be used to reset to the previous `Context`.

        Args:
            context: The Context to set.
        """
        return self._current_context.set(context)

    @override
    def get_current(self) -> Context:
        """Returns the current `Context` object."""
        return self._current_context.get()

    @override
    def detach(self, token: Token[Context]) -> None:
        """Resets Context to a previous value

        Args:
            token: A reference to a previous Context.
        """
        self._current_context.reset(token)


# Create the isolated runtime context
_ISOLATED_RUNTIME_CONTEXT = IsolatedContextVarsRuntimeContext()

# Token stack for push/pop API compatibility - much lighter than copying contexts
_isolated_token_stack: ContextVar[list[Token[Context]]] = ContextVar("isolated_token_stack")
_set_token = _isolated_token_stack.set([])

# Thread-local storage for threading support
_isolated_token_stack_storage = threading.local()

# ContextVar to track if we're in a LiteLLM context
_in_litellm_context: ContextVar[bool] = ContextVar("in_litellm_context", default=False)


def get_token_stack() -> list[Token[Context]]:
    """Get the token stack, supporting both asyncio and threading."""
    try:
        return _isolated_token_stack.get()
    except LookupError:
        if not hasattr(_isolated_token_stack_storage, "token_stack"):
            _isolated_token_stack_storage.token_stack = []
        return _isolated_token_stack_storage.token_stack  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def set_token_stack(stack: list[Token[Context]]) -> None:
    """Set the token stack, supporting both asyncio and threading."""
    try:
        _set_token = _isolated_token_stack.set(stack)
    except LookupError:
        _isolated_token_stack_storage.token_stack = stack


def get_current_context() -> Context:
    """Get the current isolated context."""
    return _ISOLATED_RUNTIME_CONTEXT.get_current()


def attach_context(context: Context) -> Token[Context]:
    """Attach a context to the isolated runtime context."""
    return _ISOLATED_RUNTIME_CONTEXT.attach(context)


def detach_context(token: Token[Context]) -> None:
    """Detach a context from the isolated runtime context."""
    _ISOLATED_RUNTIME_CONTEXT.detach(token)


CONTEXT_USER_ID_KEY = create_key(f"lmnr.{USER_ID}")
CONTEXT_SESSION_ID_KEY = create_key(f"lmnr.{SESSION_ID}")
CONTEXT_METADATA_KEY = create_key(f"lmnr.{METADATA}")
CONTEXT_TRACE_TYPE_KEY = create_key(f"lmnr.{TRACE_TYPE}")


def get_event_attributes_from_context(context: Context | None = None) -> dict[str, str]:
    """Get the event attributes from the context."""
    attributes: dict[str, str] = {}
    try:
        context = context or get_current_context()
        if session_id := cast(str, get_value(CONTEXT_SESSION_ID_KEY, context)):
            attributes["lmnr.event.session_id"] = session_id
        if user_id := cast(str, get_value(CONTEXT_USER_ID_KEY, context)):
            attributes["lmnr.event.user_id"] = user_id
    except Exception:
        logger.debug("Error getting event attributes from context", exc_info=True)
    return attributes


def set_association_prop_context(
    user_id: str | None = None,
    session_id: str | None = None,
    trace_type: TraceType | None = None,
    context: Context | None = None,
    metadata: dict[str, MetadataMemberType] | None = None,
    attach: bool = True,
) -> Context:
    context = context or get_current_context()
    if user_id is not None:
        context = set_value(CONTEXT_USER_ID_KEY, user_id, context)
    if session_id is not None:
        context = set_value(CONTEXT_SESSION_ID_KEY, session_id, context)
    if trace_type is not None:
        context = set_value(CONTEXT_TRACE_TYPE_KEY, trace_type.value, context)
    if metadata is not None:
        context = set_value(CONTEXT_METADATA_KEY, metadata, context)
    if attach:
        _attach_token = attach_context(context)
    return context


def pop_span_context() -> None:
    """Pop the current span context from the stack."""
    current_stack = get_token_stack().copy()
    if current_stack:
        token = current_stack.pop()
        set_token_stack(current_stack)
        detach_context(token)


def push_span_context(context: Context) -> None:
    """Push a new span context onto the stack."""
    token = attach_context(context)
    token_stack = get_token_stack().copy()
    token_stack.append(token)
    set_token_stack(token_stack)


def push_span(span: trace.Span, from_ctx: Context | None = None) -> Context:
    """Push a new context with the given span onto the stack."""
    new_context = trace.set_span_in_context(span, from_ctx or get_current_context())
    push_span_context(new_context)
    return new_context


def clear_context() -> None:
    """Clear the isolated context and token stack.

    This is primarily used during force_flush operations in Lambda-like
    environments to ensure subsequent invocations don't continue traces
    from previous invocations.

    Warning: This should only be called when you're certain no spans are
    actively being processed, as it will reset all context state.
    """
    # Clear the token stack first
    try:
        _set_token = _isolated_token_stack.set([])
    except LookupError:
        pass

    # Clear thread-local storage if it exists
    if hasattr(_isolated_token_stack_storage, "token_stack"):
        _isolated_token_stack_storage.token_stack = []

    # Reset the context to a fresh empty context
    # This doesn't require manually detaching tokens since we're
    # intentionally resetting everything to a clean state
    _set_token = _ISOLATED_RUNTIME_CONTEXT._current_context.set(Context())  # pyright: ignore[reportPrivateUsage] same file


def is_in_litellm_context() -> bool:
    """Check if we're currently in a LiteLLM context."""
    return _in_litellm_context.get()


@contextmanager
def in_litellm_context() -> Generator[None, None, None]:
    """Context manager to run code in a LiteLLM context.

    This sets a flag that can be checked by instrumentation code to determine
    if it's being called from within LiteLLM, allowing it to avoid double-instrumentation.
    """
    token = _in_litellm_context.set(True)
    try:
        yield
    finally:
        _in_litellm_context.reset(token)


# Set once, process-wide: `threading.Thread.__init__` is monkey-patched so that
# threads inherit Laminar's isolated context, and the original must survive
# repeated initialize()/shutdown() cycles.
_original_thread_init: Callable[..., None] | None = None


def setup_thread_context_inheritance() -> None:
    """Make new threads inherit the current isolated context and token stack."""
    global _original_thread_init
    if _original_thread_init is not None:
        return

    _original_thread_init = threading.Thread.__init__

    def patched_thread_init(thread_self: threading.Thread, *args: Any, **kwargs: Any):  # pyright: ignore[reportExplicitAny, reportAny]
        # Capture current isolated context and token stack for inheritance
        current_context = get_current_context()
        current_token_stack = get_token_stack().copy()

        # Get the original target function
        original_target = kwargs.get("target")
        if not original_target and args:
            original_target = args[0]  # pyright: ignore[reportAny]

        # Only inherit if we have a target function
        if original_target:
            # Create a wrapper function that sets up context
            def thread_wrapper(*target_args: Any, **target_kwargs: Any):  # pyright: ignore[reportExplicitAny, reportAny]
                # Set inherited context and token stack in the new thread
                _attach_token = attach_context(current_context)
                set_token_stack(current_token_stack)
                # Run original target
                return original_target(*target_args, **target_kwargs)  # pyright: ignore[reportAny]

            # Replace the target with our wrapper
            if "target" in kwargs:
                kwargs["target"] = thread_wrapper
            elif args:
                args = (thread_wrapper,) + args[1:]

        # Call original init
        if _original_thread_init is not None:
            _original_thread_init(thread_self, *args, **kwargs)

    threading.Thread.__init__ = patched_thread_init  # pyright: ignore[reportAttributeAccessIssue] thread_self -> self aliasing

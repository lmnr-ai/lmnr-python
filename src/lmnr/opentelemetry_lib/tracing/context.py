import threading

from abc import ABC, abstractmethod
from contextvars import ContextVar
from opentelemetry.context import Context, Token, create_key, get_value

from lmnr.opentelemetry_lib.tracing.attributes import SESSION_ID, USER_ID


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
        self._current_context = ContextVar(
            "isolated_current_context", default=Context()
        )

    def attach(self, context: Context) -> Token[Context]:
        """Sets the current `Context` object. Returns a
        token that can be used to reset to the previous `Context`.

        Args:
            context: The Context to set.
        """
        return self._current_context.set(context)

    def get_current(self) -> Context:
        """Returns the current `Context` object."""
        return self._current_context.get()

    def detach(self, token: Token[Context]) -> None:
        """Resets Context to a previous value

        Args:
            token: A reference to a previous Context.
        """
        self._current_context.reset(token)


# Create the isolated runtime context
_ISOLATED_RUNTIME_CONTEXT = IsolatedContextVarsRuntimeContext()

# Token stack for push/pop API compatibility - much lighter than copying contexts
_isolated_token_stack: ContextVar[list[Token[Context]]] = ContextVar(
    "isolated_token_stack", default=[]
)

# Thread-local storage for threading support
_isolated_token_stack_storage = threading.local()


def get_token_stack() -> list[Token[Context]]:
    """Get the token stack, supporting both asyncio and threading."""
    try:
        return _isolated_token_stack.get()
    except LookupError:
        if not hasattr(_isolated_token_stack_storage, "token_stack"):
            _isolated_token_stack_storage.token_stack = []
        return _isolated_token_stack_storage.token_stack


def set_token_stack(stack: list[Token[Context]]) -> None:
    """Set the token stack, supporting both asyncio and threading."""
    try:
        _isolated_token_stack.set(stack)
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


def get_event_attributes_from_context(context: Context | None = None) -> dict[str, str]:
    """Get the event attributes from the context."""
    context = context or get_current_context()
    attributes = {}
    if session_id := get_value(CONTEXT_SESSION_ID_KEY, context):
        attributes["lmnr.event.session_id"] = session_id
    if user_id := get_value(CONTEXT_USER_ID_KEY, context):
        attributes["lmnr.event.user_id"] = user_id
    return attributes


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
        _isolated_token_stack.set([])
    except LookupError:
        pass

    # Clear thread-local storage if it exists
    if hasattr(_isolated_token_stack_storage, "token_stack"):
        _isolated_token_stack_storage.token_stack = []

    # Reset the context to a fresh empty context
    # This doesn't require manually detaching tokens since we're
    # intentionally resetting everything to a clean state
    _ISOLATED_RUNTIME_CONTEXT._current_context.set(Context())

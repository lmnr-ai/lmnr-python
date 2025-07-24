import threading

from abc import ABC, abstractmethod
from contextvars import ContextVar
from opentelemetry.context import Context, Token


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

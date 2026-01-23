import importlib
import sys

from lmnr.sdk.log import get_default_logger

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing import Any, Collection
from wrapt import FunctionWrapper, wrap_function_wrapper

from .wrappers import (
    wrap_sync,
    wrap_async,
    wrap_async_gen,
    wrap_transport_connect,
    wrap_transport_close,
    wrap_query,
    wrap_client_init,
)

logger = get_default_logger(__name__)

_instruments = ("claude-agent-sdk >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "claude_agent_sdk._internal.transport.subprocess_cli",
        "object": "SubprocessCLITransport",
        "method": "connect",
        "class_name": "SubprocessCLITransport",
        "wrapper": wrap_transport_connect,
    },
    {
        "package": "claude_agent_sdk._internal.transport.subprocess_cli",
        "object": "SubprocessCLITransport",
        "method": "close",
        "class_name": "SubprocessCLITransport",
        "wrapper": wrap_transport_close,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "__init__",
        "class_name": "ClaudeSDKClient",
        "wrapper": wrap_client_init,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "connect",
        "class_name": "ClaudeSDKClient",
        "should_publish_span_context": True,
        "wrapper": wrap_async,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "query",
        "class_name": "ClaudeSDKClient",
        "should_publish_span_context": True,
        "wrapper": wrap_async,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "receive_messages",
        "class_name": "ClaudeSDKClient",
        "wrapper": wrap_async_gen,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "receive_response",
        "class_name": "ClaudeSDKClient",
        "wrapper": wrap_async_gen,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "interrupt",
        "class_name": "ClaudeSDKClient",
        "wrapper": wrap_async,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "disconnect",
        "class_name": "ClaudeSDKClient",
        "wrapper": wrap_async,
    },
    {
        # Module-level query function (streaming)
        "package": "claude_agent_sdk",
        "method": "query",
        "should_publish_span_context": True,
        "wrapper": wrap_query,
    },
    {
        # Module-level create_sdk_mcp_server function (sync)
        "package": "claude_agent_sdk",
        "method": "create_sdk_mcp_server",
        "wrapper": wrap_sync,
    },
]

_MODULE_FUNCTION_ORIGINALS: dict[tuple[str, str], Any] = {}


def _replace_function_aliases(original, wrapped):
    for module in list(sys.modules.values()):
        module_dict = getattr(module, "__dict__", None)
        if not module_dict:
            continue
        for attr, value in list(module_dict.items()):
            if value is original:
                setattr(module, attr, wrapped)


def _wrap_module_function(module_name: str, function_name: str, wrapper):
    try:
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
    except ModuleNotFoundError:
        return

    try:
        original = getattr(module, function_name)
    except AttributeError:
        return

    key = (module_name, function_name)
    if key not in _MODULE_FUNCTION_ORIGINALS:
        _MODULE_FUNCTION_ORIGINALS[key] = original

    wrapped_function = FunctionWrapper(original, wrapper)
    setattr(module, function_name, wrapped_function)
    _replace_function_aliases(original, wrapped_function)


def _unwrap_module_function(module_name: str, function_name: str):
    key = (module_name, function_name)
    original = _MODULE_FUNCTION_ORIGINALS.get(key)
    if not original:
        return

    module = sys.modules.get(module_name)
    if not module:
        return

    current = getattr(module, function_name, None)
    setattr(module, function_name, original)
    if current is not None:
        _replace_function_aliases(current, original)
    del _MODULE_FUNCTION_ORIGINALS[key]


class ClaudeAgentInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        # Wrap methods
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrapper_func = wrapped_method.get("wrapper")

            # Create wrapper instance with metadata
            wrapper = wrapper_func(wrapped_method)

            if wrap_object:
                target = f"{wrap_object}.{wrap_method}"
                try:
                    wrap_function_wrapper(
                        wrap_package,
                        target,
                        wrapper,
                    )
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok, we don't want to fail if some methods do not exist
            else:
                try:
                    _wrap_module_function(
                        wrap_package,
                        wrap_method,
                        wrapper,
                    )
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            if wrap_object:
                module_path = f"{wrap_package}.{wrap_object}"
                try:
                    unwrap(module_path, wrap_method)
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok, we don't want to fail if some methods do not exist
            else:
                _unwrap_module_function(wrap_package, wrap_method)

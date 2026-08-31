from importlib.metadata import version
from typing import Collection

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
)
from lmnr.sdk.log import get_default_logger

from .types import ClaudeAgentSpec
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

WRAPPED_FUNCTIONS: list[ClaudeAgentSpec] = [
    ClaudeAgentSpec(
        package_name="claude_agent_sdk._internal.transport.subprocess_cli",
        object_name="SubprocessCLITransport",
        method_name="connect",
        is_async=True,
        class_name="SubprocessCLITransport",
        wrapper_function=wrap_transport_connect,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk._internal.transport.subprocess_cli",
        object_name="SubprocessCLITransport",
        method_name="close",
        is_async=True,
        class_name="SubprocessCLITransport",
        wrapper_function=wrap_transport_close,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="__init__",
        is_async=False,
        class_name="ClaudeSDKClient",
        wrapper_function=wrap_client_init,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="connect",
        is_async=True,
        class_name="ClaudeSDKClient",
        should_publish_span_context=True,
        wrapper_function=wrap_async,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="query",
        is_async=True,
        class_name="ClaudeSDKClient",
        should_publish_span_context=True,
        wrapper_function=wrap_async,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="receive_messages",
        # the wrapper is a plain function returning an async generator
        is_async=False,
        is_streaming=True,
        class_name="ClaudeSDKClient",
        wrapper_function=wrap_async_gen,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="receive_response",
        is_async=False,
        is_streaming=True,
        class_name="ClaudeSDKClient",
        wrapper_function=wrap_async_gen,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="interrupt",
        is_async=True,
        class_name="ClaudeSDKClient",
        wrapper_function=wrap_async,
    ),
    ClaudeAgentSpec(
        package_name="claude_agent_sdk.client",
        object_name="ClaudeSDKClient",
        method_name="disconnect",
        is_async=True,
        class_name="ClaudeSDKClient",
        wrapper_function=wrap_async,
    ),
    # Module-level query function (streaming). `replace_aliases` is what makes
    # `from claude_agent_sdk import query` pick up the wrapper regardless of
    # import order -- previously this package carried its own private copy of
    # that machinery; it now comes from BaseLaminarInstrumentor.
    ClaudeAgentSpec(
        package_name="claude_agent_sdk",
        object_name=None,
        method_name="query",
        is_async=False,
        is_streaming=True,
        replace_aliases=True,
        should_publish_span_context=True,
        wrapper_function=wrap_query,
    ),
    # Module-level create_sdk_mcp_server function (sync)
    ClaudeAgentSpec(
        package_name="claude_agent_sdk",
        object_name=None,
        method_name="create_sdk_mcp_server",
        is_async=False,
        replace_aliases=True,
        wrapper_function=wrap_sync,
    ),
]


class ClaudeAgentInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                sdk_version = version("claude-agent-sdk")
            except Exception as e:
                logger.debug(f"Failed to get claude-agent-sdk version {e}")
                sdk_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="claude-agent-sdk",
                version=sdk_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

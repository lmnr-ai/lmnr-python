"""OpenTelemetry instrumentation for the MCP Python SDK (`mcp`).

The 1.x and 2.x wrappers live in `v1.py` and `v2.py`; the instrumentor
installs the set matching the installed major. They are not simply all
registered: 2.x's `ClientSession.send_request` exists on 1.x too (inherited
from `BaseSession`), and wrapping it there would double every client span.
"""

from __future__ import annotations

from collections.abc import Collection
from importlib.metadata import version

from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.v1 import V1_WRAPPED_FUNCTIONS
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.mcp.v2 import V2_WRAPPED_FUNCTIONS
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_instruments = ("mcp >= 1.6.0",)


def wrapped_functions(mcp_version: str) -> list[WrappedFunctionSpec]:
    """The specs for an installed mcp version; 2.x's when it can't be parsed."""
    try:
        major = int(mcp_version.split(".")[0])
    except ValueError:
        major = 2
    return V1_WRAPPED_FUNCTIONS if major < 2 else V2_WRAPPED_FUNCTIONS


class McpInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the MCP Python SDK."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def __init__(self):
        super().__init__()
        scope = self.instrumentation_scope()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": scope}
                for spec in wrapped_functions(scope["version"])
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                mcp_version = version("mcp")
            except Exception:
                logger.debug("Failed to get mcp version", exc_info=True)
                mcp_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="mcp",
                version=mcp_version,
            )
        return self._scope

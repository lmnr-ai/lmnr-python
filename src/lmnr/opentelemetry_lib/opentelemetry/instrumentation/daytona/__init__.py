"""OpenTelemetry Daytona SDK instrumentation

This module instruments the Daytona SDK to capture traces and logs for
execute_session_command calls.

The instrumentation handles both synchronous and asynchronous commands:

Synchronous commands (run_async/var_async=False):
1. Create a span when execute_session_command is called
2. Execute the command and wait for completion
3. End the span after the command returns
4. Emit logs immediately from response.stdout/stderr

Asynchronous commands (run_async/var_async=True):
1. Create a span when execute_session_command is called
2. Execute the command (returns immediately with cmd_id)
3. End the span after execute_session_command returns
4. Start background log streaming to capture stdout/stderr as they arrive
5. Emit OpenTelemetry logs for each line using the Logs API

Note: The module path for the Daytona SDK may need adjustment based on the
actual package structure. Update WRAPPED_METHODS if the module path differs.
"""

from typing import Collection
from importlib.metadata import version

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import BaseLaminarInstrumentor, LaminarInstrumentorConfig
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import WrappedFunctionSpec, LaminarInstrumentationScopeAttributes

from .wrappers import _wrap, _awrap

_instruments = ("daytona >= 0.1.0",)

class DaytonaSDKInstrumentor(BaseLaminarInstrumentor):

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        daytona_version = "unknown"
        try:
            daytona_version = version("daytona")
        except Exception:
            pass
        return LaminarInstrumentationScopeAttributes(
            name="daytona",
            version=daytona_version,
        )

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is not None:
            return self._scope
        self._scope = self._instrumentation_scope()
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config =LaminarInstrumentorConfig(
            wrapped_functions= [
                WrappedFunctionSpec(
                    package_name="daytona._sync.process",
                    object_name="Process",
                    method_name="execute_session_command",
                    is_async=False,
                    is_streaming=False,
                    span_name="daytona.sandbox.process.execute_session_command",
                    span_type="DEFAULT",
                    wrapper_function=_wrap,
                ),
                WrappedFunctionSpec(
                    package_name="daytona._async.process",
                    object_name="AsyncProcess",
                    method_name="execute_session_command",
                    is_async=True,
                    is_streaming=False,
                    span_name="daytona.sandbox.process.execute_session_command",
                    span_type="DEFAULT",
                    wrapper_function=_awrap,
                ),
            ]
        )
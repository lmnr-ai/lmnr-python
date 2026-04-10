"""OpenTelemetry Modal SDK instrumentation

This module instruments the Modal SDK to capture traces and logs for
Sandbox.create() and Sandbox.exec() calls.

Sandbox.create:
1. Create a span when Sandbox.create is called
2. Record sandbox creation parameters (image, timeout, cpu, memory, etc.)
3. End the span after the sandbox is created
4. Capture the sandbox_id in the response

Sandbox.exec:
1. Create a span when exec is called
2. Execute the command and return an instrumented ContainerProcess
3. Stdout/stderr are captured as OTel logs as the user iterates over them
4. The span is ended when process.wait() is called, capturing the exit code
"""

from typing import Collection
from importlib.metadata import version

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
    LaminarInstrumentorConfig,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
    LaminarInstrumentationScopeAttributes,
)

from .wrappers import _wrap_create, _awrap_create, _wrap_exec, _awrap_exec

_instruments = ("modal >= 0.50.0",)


class ModalSDKInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        modal_version = "unknown"
        try:
            modal_version = version("modal")
        except Exception:
            pass
        return LaminarInstrumentationScopeAttributes(
            name="modal",
            version=modal_version,
        )

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is not None:
            return self._scope
        self._scope = self._instrumentation_scope()
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    package_name="modal.sandbox",
                    object_name="Sandbox",
                    method_name="create",
                    is_async=False,
                    is_streaming=False,
                    span_name="modal.sandbox.create",
                    span_type="DEFAULT",
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_wrap_create,
                ),
                WrappedFunctionSpec(
                    package_name="modal.sandbox",
                    object_name="Sandbox",
                    method_name="exec",
                    is_async=False,
                    is_streaming=False,
                    span_name="modal.sandbox.exec",
                    span_type="DEFAULT",
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_wrap_exec,
                ),
            ]
        )

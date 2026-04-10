"""OpenTelemetry Modal Sandbox instrumentation

Instruments Modal Sandbox operations:

Sandbox.create:
1. Create a span when Sandbox.create is called
2. Execute the creation and wait for completion
3. End the span after sandbox is ready
4. Record sandbox ID and configuration as span attributes

Sandbox.exec:
1. Create a span when sandbox.exec is called
2. Execute the command (returns ContainerProcess)
3. End the span after exec returns
4. Wrap stdout/stderr iterators to emit OTel logs as user reads them

Only the public `Sandbox` class is instrumented. The internal `_Sandbox` class
is not instrumented because `Sandbox` is a wrapper that users interact with
directly.
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

from .wrappers import _wrap_create, _wrap_exec

_instruments = ("modal >= 0.73.0",)


class ModalSandboxInstrumentor(BaseLaminarInstrumentor):
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
        scope = self.instrumentation_scope()
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                # Sandbox.create (sync) - wraps the FunctionWithAio __call__
                WrappedFunctionSpec(
                    package_name="modal.sandbox",
                    object_name="Sandbox",
                    method_name="create",
                    is_async=False,
                    is_streaming=False,
                    span_name="modal.sandbox.create",
                    span_type="DEFAULT",
                    instrumentation_scope=scope,
                    wrapper_function=_wrap_create,
                ),
                # Sandbox.exec (sync) - wraps the partial
                WrappedFunctionSpec(
                    package_name="modal.sandbox",
                    object_name="Sandbox",
                    method_name="exec",
                    is_async=False,
                    is_streaming=False,
                    span_name="modal.sandbox.exec",
                    span_type="DEFAULT",
                    instrumentation_scope=scope,
                    wrapper_function=_wrap_exec,
                ),
            ]
        )

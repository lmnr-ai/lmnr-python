"""OpenTelemetry Modal Sandbox instrumentation.

Modal's public SDK methods are exposed via the ``synchronicity`` library:

* ``Sandbox.create`` is a ``FunctionWithAio`` whose ``__call__`` dispatches to a
  blocking implementation (``_func``) and whose ``.aio`` attribute points at an
  async implementation (``_aio_func``).
* ``Sandbox.exec`` is a ``MethodWithAio`` descriptor whose ``__get__`` returns
  a freshly-bound ``functools.partial`` on every attribute access. The partial
  wraps ``_func``/``_aio_func`` with the receiver already bound as the first
  positional argument.

Instrumenting just the class attribute via ``wrap_function_wrapper`` would
miss the async code path entirely, because ``.aio`` holds an independent
reference to the underlying coroutine function. Instead, this instrumentor
replaces the inner ``_func``/``_aio_func`` (and the ``aio`` attribute on
``FunctionWithAio``) with ``wrapt.FunctionWrapper`` instances. This way,
whether the caller invokes ``Sandbox.create(...)``, ``Sandbox.create.aio(...)``,
``sandbox.exec(...)`` or ``sandbox.exec.aio(...)``, the corresponding span
is emitted.

Only the public ``Sandbox`` class is instrumented. The internal ``_Sandbox``
class is intentionally left alone because it is a library-private
implementation detail that end-users are not expected to interact with.
"""

from typing import Any, Collection
from importlib.metadata import version

from wrapt import FunctionWrapper

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
)

from .wrappers import (
    _wrap_create,
    _wrap_create_async,
    _wrap_exec,
    _wrap_exec_async,
)

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
        self._patched: list[tuple[Any, str, Any]] = []

    # --- Custom instrumentation -------------------------------------------------

    def _instrument(self, **kwargs):
        try:
            import modal.sandbox
            from synchronicity.combined_types import (
                FunctionWithAio,
                MethodWithAio,
            )
        except (ImportError, ModuleNotFoundError) as e:
            self.logger.debug(f"Modal not available, skipping instrumentation: {e}")
            return

        Sandbox = modal.sandbox.Sandbox

        self._patch_function_with_aio(
            owner=Sandbox,
            attr="create",
            expected_type=FunctionWithAio,
            sync_wrapper=_wrap_create,
            async_wrapper=_wrap_create_async,
        )
        self._patch_method_with_aio(
            owner=Sandbox,
            attr="exec",
            expected_type=MethodWithAio,
            sync_wrapper=_wrap_exec,
            async_wrapper=_wrap_exec_async,
        )

    def _uninstrument(self, **kwargs):
        for target, attr, original in reversed(self._patched):
            try:
                setattr(target, attr, original)
            except Exception as e:
                self.logger.debug(
                    f"Failed to restore {target!r}.{attr}: {e}"
                )
        self._patched.clear()

    # --- Helpers ---------------------------------------------------------------

    def _record_patch(self, target: Any, attr: str) -> Any:
        """Snapshot ``target.attr`` so we can restore it during uninstrument."""
        original = getattr(target, attr)
        self._patched.append((target, attr, original))
        return original

    def _patch_function_with_aio(
        self,
        owner,
        attr: str,
        expected_type,
        sync_wrapper,
        async_wrapper,
    ):
        """Wrap both the blocking and async implementations of a FunctionWithAio.

        The blocking path goes through ``FunctionWithAio.__call__`` which
        internally calls ``self._func``; the async path is a direct reference
        to ``self._aio_func`` exposed as ``self.aio``. Both must be wrapped
        for instrumentation to see every invocation.
        """
        try:
            fwa = owner.__dict__.get(attr)
            if not isinstance(fwa, expected_type):
                self.logger.debug(
                    f"{owner.__name__}.{attr} is not a FunctionWithAio "
                    f"(got {type(fwa).__name__}); skipping"
                )
                return

            orig_func = self._record_patch(fwa, "_func")
            orig_aio_func = self._record_patch(fwa, "_aio_func")
            # ``aio`` is initialised to the same object as ``_aio_func`` in
            # FunctionWithAio.__init__, so we don't need the return value - but
            # we still need to snapshot it so _uninstrument restores the slot.
            self._record_patch(fwa, "aio")

            fwa._func = FunctionWrapper(orig_func, sync_wrapper)
            wrapped_aio = FunctionWrapper(orig_aio_func, async_wrapper)
            fwa._aio_func = wrapped_aio
            fwa.aio = wrapped_aio

            self.logger.debug(
                f"Instrumented {owner.__name__}.{attr} (FunctionWithAio)"
            )
        except Exception as e:
            self.logger.debug(
                f"Failed to instrument {owner.__name__}.{attr}: {e}"
            )

    def _patch_method_with_aio(
        self,
        owner,
        attr: str,
        expected_type,
        sync_wrapper,
        async_wrapper,
    ):
        """Wrap both sides of a MethodWithAio descriptor.

        MethodWithAio rebuilds a ``functools.partial`` on every attribute
        access, so instance-level caching won't work. Instead, we wrap the
        descriptor's ``_func`` / ``_aio_func`` attributes directly: every
        partial subsequently produced by ``__get__`` will reuse our wrapped
        callables.
        """
        try:
            mwa = owner.__dict__.get(attr)
            if not isinstance(mwa, expected_type):
                self.logger.debug(
                    f"{owner.__name__}.{attr} is not a MethodWithAio "
                    f"(got {type(mwa).__name__}); skipping"
                )
                return

            orig_func = self._record_patch(mwa, "_func")
            orig_aio_func = self._record_patch(mwa, "_aio_func")

            mwa._func = FunctionWrapper(orig_func, sync_wrapper)
            mwa._aio_func = FunctionWrapper(orig_aio_func, async_wrapper)

            self.logger.debug(
                f"Instrumented {owner.__name__}.{attr} (MethodWithAio)"
            )
        except Exception as e:
            self.logger.debug(
                f"Failed to instrument {owner.__name__}.{attr}: {e}"
            )

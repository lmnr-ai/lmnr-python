import importlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import Logger
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing_extensions import override
from wrapt import FunctionWrapper, wrap_function_wrapper

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WraptWrapper,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    add_spec_wrapper,
)
from lmnr.sdk.log import get_default_logger


class BaseLaminarInstrumentor(BaseInstrumentor, ABC):
    instrumentor_config: LaminarInstrumentorConfig
    logger: Logger = get_default_logger(__name__)

    # Store original functions for alias replacement and uninstrumentation
    _module_function_originals: dict[tuple[str, str], Callable[..., Any]] | None = None
    # The wrapper callable installed per (package_name, target), so uninstrument
    # can find and remove OUR wrapt layer even when another instrumentor has
    # stacked its own layer on top of it
    _own_wrappers: dict[tuple[str, str], WraptWrapper] | None = None

    @abstractmethod
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        pass

    def wrapper_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments handed to every wrapper on every call.

        Override when a wrapper needs an instrumentor-level collaborator that is
        not per-method config and so does not belong in the spec — the browser
        instrumentors use this to pass their `AsyncLaminarClient`. Returned
        values are forwarded by `add_spec_wrapper`, so a wrapper receives them as
        keyword-only arguments after `(to_wrap, wrapped, instance, args, kwargs)`.
        """
        return {}

    @staticmethod
    def _replace_function_aliases(
        original: Callable[..., Any],
        wrapped: Callable[..., Any],
    ) -> None:
        """
        Replace all references to the original function across ALL loaded modules.

        This enables instrumentation to work regardless of import order. When a user does
        `from package import function`, their module gets an entry in sys.modules with a
        reference to the original function. This method finds all such references and
        replaces them with the wrapped version.

        This is particularly useful for module-level functions that can be imported
        directly (e.g., `from litellm import completion`).

        Args:
            original: The original unwrapped function
            wrapped: The wrapped/instrumented function
        """
        for module in list(sys.modules.values()):
            module_dict = getattr(module, "__dict__", None)
            if not module_dict:
                continue
            for attr, value in list(module_dict.items()):
                if value is original:
                    try:
                        setattr(module, attr, wrapped)
                    except (AttributeError, TypeError):
                        # Some modules may have read-only attributes
                        pass

    def _wrap_module_function_with_alias_replacement(
        self, module_name: str, function_name: str, wrapper: WraptWrapper
    ) -> bool:
        """
        Wrap a module-level function and replace all aliases across loaded modules.

        Args:
            module_name: The name of the module containing the function
            function_name: The name of the function to wrap
            wrapper: The wrapper function to apply

        Returns:
            bool: True if wrapping succeeded, False otherwise
        """
        try:
            module = sys.modules.get(module_name) or importlib.import_module(module_name)
        except (ModuleNotFoundError, ImportError):
            return False

        try:
            original = getattr(module, function_name)
        except AttributeError:
            return False

        if self._module_function_originals is None:
            self._module_function_originals = {}
        key = (module_name, function_name)
        if key not in self._module_function_originals:
            self._module_function_originals[key] = original

        wrapped_function = FunctionWrapper(original, wrapper)
        setattr(module, function_name, wrapped_function)

        # Replace all existing references to the original function across ALL loaded modules
        self._replace_function_aliases(original, wrapped_function)

        return True

    def _unwrap_module_function_with_alias_replacement(
        self, module_name: str, function_name: str, own_wrapper: WraptWrapper | None = None
    ) -> None:
        """
        Restore the original function and replace all aliases back.

        Args:
            module_name: The name of the module containing the function
            function_name: The name of the function to unwrap
            own_wrapper: The wrapper we installed; when another instrumentor has
                wrapped the module attribute on top of it, only our layer is
                spliced out from under theirs
        """
        key = (module_name, function_name)
        original = (self._module_function_originals or {}).get(key)
        if not original:
            return

        module = sys.modules.get(module_name)
        if not module:
            return

        current = getattr(module, function_name, None)
        if own_wrapper is not None and getattr(current, "_self_wrapper", None) is not own_wrapper:
            removed = self._remove_own_layer(module, function_name, own_wrapper)
            # The later wrap only replaced the module attribute; every alias we
            # retargeted at instrument time (`from module import fn` bindings,
            # litellm's dispatch `_PYTHON_*` fallback) still holds our layer
            if removed is not None:
                self._replace_function_aliases(removed, removed.__wrapped__)
        else:
            setattr(module, function_name, original)
            if current is not None:
                self._replace_function_aliases(current, original)
        if self._module_function_originals is not None:
            del self._module_function_originals[key]

    @staticmethod
    def _remove_own_layer(holder: object, attr: str, wrapper: WraptWrapper) -> Any:
        """Remove the wrapt layer whose wrapper is `wrapper` from `holder.attr`.

        `opentelemetry.instrumentation.utils.unwrap` always strips the OUTERMOST
        layer, whoever owns it. When a third-party instrumentor wraps the same
        target after us (e.g. opentelemetry-instrumentation-langchain wraps
        `Pregel.stream`), that peels their layer and leaves ours. Instead, walk
        the chain and splice out only our layer.

        Reads the holder's own `__dict__`, never `getattr`: on a class, getattr
        of a wrapt proxy returns a freshly bound proxy, and writing that back
        would replace the stored descriptor.

        Returns the removed layer (so callers can retarget references to it),
        or None if our layer was not found.
        """
        outer: Any = None
        current: Any = vars(holder).get(attr)
        while current is not None and hasattr(current, "_self_wrapper"):
            if current._self_wrapper is wrapper:
                if outer is None:
                    setattr(holder, attr, current.__wrapped__)
                else:
                    outer.__wrapped__ = current.__wrapped__
                return current
            outer, current = current, current.__wrapped__
        return None

    # default implementation, can be overridden by subclasses
    @override
    def _instrument(self, **kwargs: dict[str, Any]):
        handler_kwargs = self.wrapper_kwargs()
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            package_name = wrapped_function_spec["package_name"]
            object_name = wrapped_function_spec.get("object_name")
            method_name = wrapped_function_spec["method_name"]
            wrapper_function = wrapped_function_spec["wrapper_function"]

            # Check if this is a module-level function that needs alias replacement
            replace_aliases = wrapped_function_spec.get("replace_aliases", False)

            target = f"{object_name}.{method_name}" if object_name else method_name

            wrapper = add_spec_wrapper(wrapper_function, wrapped_function_spec, **handler_kwargs)

            try:
                if replace_aliases and not object_name:
                    # Module-level function with alias replacement
                    success = self._wrap_module_function_with_alias_replacement(
                        package_name, method_name, wrapper
                    )
                    if success:
                        self.logger.debug(
                            f"Successfully instrumented {package_name}.{target} with alias replacement"
                        )
                    else:
                        self.logger.debug(f"Failed to instrument {package_name}.{target}")
                else:
                    # Standard class method or module function without alias replacement
                    _ = wrap_function_wrapper(package_name, target, wrapper)
                    self.logger.debug(f"Successfully instrumented {package_name}.{target}")
                if self._own_wrappers is None:
                    self._own_wrappers = {}
                self._own_wrappers[(package_name, target)] = wrapper
            except (AttributeError, ModuleNotFoundError, ImportError):
                # that's ok, we don't want to fail if some methods do not exist
                self.logger.debug(f"Failed to instrument {package_name}.{target}", exc_info=True)
            except Exception:
                self.logger.exception(f"Failed to instrument {package_name}.{target}")
                # don't re-raise, we don't want to fail the entire program

    # default implementation, can be overridden by subclasses
    @override
    def _uninstrument(self, **kwargs: dict[str, Any]):
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            package_name = wrapped_function_spec["package_name"]
            object_name = wrapped_function_spec.get("object_name")
            method_name = wrapped_function_spec["method_name"]
            replace_aliases = wrapped_function_spec.get("replace_aliases", False)

            target = f"{object_name}.{method_name}" if object_name else method_name
            own_wrapper = (self._own_wrappers or {}).pop((package_name, target), None)

            try:
                if replace_aliases and not object_name:
                    # Unwrap with alias replacement
                    self._unwrap_module_function_with_alias_replacement(
                        package_name, method_name, own_wrapper
                    )
                elif own_wrapper is not None:
                    holder: object = sys.modules.get(package_name) or importlib.import_module(
                        package_name
                    )
                    for part in object_name.split(".") if object_name else []:
                        holder = getattr(holder, part)
                    if self._remove_own_layer(holder, method_name, own_wrapper) is None:
                        self.logger.debug(f"{package_name}.{target} was no longer wrapped by us")
                elif object_name:
                    # `unwrap` resolves its first argument as `module.attr`
                    # and looks the method up on it, so the owning class has
                    # to be part of the path, not of the method name.
                    unwrap(f"{package_name}.{object_name}", method_name)
                else:
                    # NOTE the split differs from `wrap_function_wrapper` above.
                    # wrapt takes (module, "Object.method") and resolves the
                    # dotted attribute path itself; `unwrap` takes
                    # (holder, "attr") and does a single `getattr`. Passing
                    # wrapt's split here makes `unwrap` look up an attribute
                    # literally named "Object.method", find nothing, and return
                    # silently — so uninstrument becomes a no-op that leaves
                    # every method wrapped. Pinned by
                    # tests/test_instrumentor_targets.py.
                    holder = f"{package_name}.{object_name}" if object_name else package_name
                    unwrap(holder, method_name)
            except Exception:
                self.logger.debug(
                    f"Failed to uninstrument {package_name}.{target}",
                    exc_info=True,
                )
                # don't re-raise, we don't want to fail the entire program

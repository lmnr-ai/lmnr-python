from abc import abstractmethod
from logging import Logger
from typing import Any

import importlib
import sys

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper, FunctionWrapper

from .types import LaminarInstrumentationScopeAttributes, LaminarInstrumentorConfig
from .wrapper_helpers import add_spec_wrapper
from lmnr.sdk.log import get_default_logger


class BaseLaminarInstrumentor(BaseInstrumentor):
    instrumentor_config: LaminarInstrumentorConfig
    logger: Logger = get_default_logger(__name__)

    # Store original functions for alias replacement and uninstrumentation
    _module_function_originals: dict[tuple[str, str], Any] = {}

    @abstractmethod
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        pass

    @staticmethod
    def _replace_function_aliases(original, wrapped):
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
        self, module_name: str, function_name: str, wrapper
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
            module = sys.modules.get(module_name) or importlib.import_module(
                module_name
            )
        except (ModuleNotFoundError, ImportError):
            return False

        try:
            original = getattr(module, function_name)
        except AttributeError:
            return False

        key = (module_name, function_name)
        if key not in self._module_function_originals:
            self._module_function_originals[key] = original

        wrapped_function = FunctionWrapper(original, wrapper)
        setattr(module, function_name, wrapped_function)

        # Replace all existing references to the original function across ALL loaded modules
        self._replace_function_aliases(original, wrapped_function)

        return True

    def _unwrap_module_function_with_alias_replacement(
        self, module_name: str, function_name: str
    ):
        """
        Restore the original function and replace all aliases back.

        Args:
            module_name: The name of the module containing the function
            function_name: The name of the function to unwrap
        """
        key = (module_name, function_name)
        original = self._module_function_originals.get(key)
        if not original:
            return

        module = sys.modules.get(module_name)
        if not module:
            return

        current = getattr(module, function_name, None)
        setattr(module, function_name, original)
        if current is not None:
            self._replace_function_aliases(current, original)
        del self._module_function_originals[key]

    # default implementation, can be overridden by subclasses
    def _instrument(self, **kwargs):
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            package_name = wrapped_function_spec["package_name"]
            object_name = wrapped_function_spec.get("object_name")
            method_name = wrapped_function_spec["method_name"]
            wrapper_function = wrapped_function_spec["wrapper_function"]

            # Check if this is a module-level function that needs alias replacement
            replace_aliases = wrapped_function_spec.get("replace_aliases", False)

            target = f"{object_name}.{method_name}" if object_name else method_name

            wrapper = add_spec_wrapper(wrapper_function, wrapped_function_spec)

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
                        self.logger.debug(
                            f"Failed to instrument {package_name}.{target}"
                        )
                else:
                    # Standard class method or module function without alias replacement
                    wrap_function_wrapper(package_name, target, wrapper)
                    self.logger.debug(
                        f"Successfully instrumented {package_name}.{target}"
                    )
            except (AttributeError, ModuleNotFoundError, ImportError) as e:
                # that's ok, we don't want to fail if some methods do not exist
                self.logger.debug(f"Failed to instrument {package_name}.{target}: {e}")
            except Exception as e:
                self.logger.error(f"Failed to instrument {package_name}.{target}: {e}")
                # don't re-raise, we don't want to fail the entire program

    # default implementation, can be overridden by subclasses
    def _uninstrument(self, **kwargs):
        for wrapped_function_spec in self.instrumentor_config["wrapped_functions"]:
            package_name = wrapped_function_spec["package_name"]
            object_name = wrapped_function_spec.get("object_name")
            method_name = wrapped_function_spec["method_name"]
            replace_aliases = wrapped_function_spec.get("replace_aliases", False)

            target = f"{object_name}.{method_name}" if object_name else method_name

            try:
                if replace_aliases and not object_name:
                    # Unwrap with alias replacement
                    self._unwrap_module_function_with_alias_replacement(
                        package_name, method_name
                    )
                else:
                    # Standard unwrap
                    unwrap(package_name, target)
            except Exception as e:
                self.logger.debug(
                    f"Failed to uninstrument {package_name}.{target}: {e}"
                )
                # don't re-raise, we don't want to fail the entire program

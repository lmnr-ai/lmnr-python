from typing import Callable, TypedDict

from opentelemetry.trace import Tracer


class LaminarInstrumentationScopeAttributes(TypedDict):
    name: str
    version: str


class WrappedFunctionSpec(TypedDict, total=False):
    # Required fields
    package_name: str
    method_name: str
    is_async: bool
    wrapper_function: Callable[[Tracer, Callable], Callable]

    # Optional fields
    object_name: str | None
    is_streaming: bool | None
    span_name: str | None
    span_type: str | None
    replace_aliases: (
        bool  # When True, replaces all references to the function across loaded modules
    )
    instrumentation_scope: LaminarInstrumentationScopeAttributes


class LaminarInstrumentorConfig(TypedDict):
    wrapped_functions: list[WrappedFunctionSpec]

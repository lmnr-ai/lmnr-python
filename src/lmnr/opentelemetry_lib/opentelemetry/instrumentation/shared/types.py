from typing import Callable, TypedDict

from opentelemetry.trace import Tracer


class WrappedFunctionSpec(TypedDict):
    package_name: str
    object_name: str | None
    method_name: str
    is_async: bool
    is_streaming: bool | None
    span_name: str | None
    span_type: str | None
    wrapper_function: Callable[[Tracer, Callable], Callable]


class LaminarInstrumentorConfig(TypedDict):
    wrapped_functions: list[WrappedFunctionSpec]

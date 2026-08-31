from typing import Any, Callable, Sequence, TypedDict, TypeVar


class LaminarInstrumentationScopeAttributes(TypedDict):
    name: str
    version: str


#: Bound is a forward reference because `WrappedFunctionSpec` is declared below —
#: the alias `WrapperHandler` has to exist before the TypedDict that stores one.
SpecT = TypeVar("SpecT", bound="WrappedFunctionSpec")

#: A wrapper handler: the function an instrumentation actually writes. It is NOT
#: wrapt-shaped — `add_spec_wrapper` adapts it by binding the spec as the first
#: argument, so the wrapper can read `span_name` / `span_type` / its own extras
#: off `to_wrap` without a closure factory.
WrapperHandler = Callable[
    # (to_wrap, wrapped, instance, args, kwargs) -> Any
    [SpecT, Callable[..., Any], Any, Sequence[Any], dict[str, Any]],
    Any,
]

#: The shape wrapt itself expects from `wrap_function_wrapper`.
WraptWrapper = Callable[[Callable[..., Any], Any, Sequence[Any], dict[str, Any]], Any]


class _RequiredWrappedFunctionSpec(TypedDict):
    """The half of a spec that must always be present.

    Kept as a separate `total=True` base because `BaseLaminarInstrumentor`
    subscripts these keys directly (`spec["package_name"]`); declaring them on a
    `total=False` TypedDict would make every one of those an unchecked KeyError.
    """

    package_name: str
    method_name: str
    is_async: bool
    # `WrapperHandler[Any]` rather than `WrapperHandler[WrappedFunctionSpec]`:
    # handlers are contravariant in their spec parameter, so one annotated
    # against a narrower subclass (e.g. `KernelSpec`) would not be assignable to
    # the base type. Precision lives at the handler's own definition site, which
    # is where an instrumentation author actually reads `to_wrap`.
    wrapper_function: "WrapperHandler[Any]"


class WrappedFunctionSpec(_RequiredWrappedFunctionSpec, total=False):
    """Declarative description of one function to instrument.

    Instrumentations that need extra per-method config subclass this and annotate
    their handler against the subclass:

        class KernelSpec(WrappedFunctionSpec, total=False):
            output_formatter: Callable[[Any], Any]

        def _wrap(to_wrap: KernelSpec, wrapped, instance, args, kwargs): ...

    Note only `package_name` / `method_name` / `object_name` / `wrapper_function`
    / `replace_aliases` are read by the instrumentor itself. The rest are passed
    through untouched for the handler to consume.
    """

    object_name: str | None
    is_streaming: bool | None
    span_name: str | None
    span_type: str | None
    # When True, replaces all references to the function across loaded modules
    replace_aliases: bool
    instrumentation_scope: LaminarInstrumentationScopeAttributes


class LaminarInstrumentorConfig(TypedDict):
    # `Sequence` rather than `list` so it is covariant: an instrumentation can
    # hand over a `list[KernelSpec]` without the invariance error `list` gives.
    wrapped_functions: Sequence[WrappedFunctionSpec]

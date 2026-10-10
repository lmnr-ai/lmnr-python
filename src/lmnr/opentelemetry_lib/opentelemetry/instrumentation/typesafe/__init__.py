"""OpenTelemetry TypeSafe AI (Jev) instrumentation"""

from collections.abc import Awaitable, Callable, Collection, Sequence
from importlib.metadata import version
from typing import Any, cast

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from typing_extensions import TypeVar, override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
    to_dict,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.typesafe.span_utils import (
    set_request_attributes,
    set_response_attributes,
)
from lmnr.sdk.types import LaminarSpanType

_instruments = ("typesafe-sdk >= 0.5.0",)
T = TypeVar("T")

# `system_one(state, questions, *, ...)` — the two leading parameters are
# positional-or-keyword, so merge `args` into the kwargs dict before reading.
_POSITIONAL_PARAMS = ("state", "questions")


def _call_kwargs(args: Sequence[Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(zip(_POSITIONAL_PARAMS, args))
    merged.update(kwargs)
    return merged


def _start_span(to_wrap: WrappedFunctionSpec) -> Span | None:
    scope = to_wrap.get("instrumentation_scope", {})
    return safe_start_span(
        name=to_wrap.get("span_name") or "typesafe",
        attributes={
            "gen_ai.system": "typesafe",
            "lmnr.span.instrumentation_scope.name": cast(str, scope.get("name") or "typesafe-sdk"),
            "lmnr.span.instrumentation_scope.version": cast(str, scope.get("version") or "unknown"),
        },
        span_type=cast(LaminarSpanType, to_wrap.get("span_type") or "DEFAULT"),
    )


def _record_error(span: Span, error: Exception):
    span.set_attribute("error.type", type(error).__name__)
    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., T],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(to_wrap)
    if not span:
        return wrapped(*args, **kwargs)

    set_request_attributes(span, _call_kwargs(args, kwargs), instance)
    try:
        response = wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        span.end()
        raise

    set_response_attributes(span, to_dict(response))
    span.end()
    return response


async def _awrap(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> T:
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(to_wrap)
    if not span:
        return await wrapped(*args, **kwargs)

    set_request_attributes(span, _call_kwargs(args, kwargs), instance)
    try:
        response = await wrapped(*args, **kwargs)
    except Exception as e:
        _record_error(span, e)
        span.end()
        raise

    set_response_attributes(span, to_dict(response))
    span.end()
    return response


class TypeSafeInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for the TypeSafe AI Python SDK."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                typesafe_version = version("typesafe-sdk")
            except Exception:
                typesafe_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="typesafe", version=typesafe_version
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                WrappedFunctionSpec(
                    # The top-level re-export shares the class object with the
                    # defining `_core.client` module, so wrapping it covers both.
                    package_name="typesafe_sdk",
                    object_name=object_name,
                    method_name="system_one",
                    is_async=is_async,
                    is_streaming=False,
                    span_name="typesafe.system_one",
                    span_type="LLM",
                    instrumentation_scope=self.instrumentation_scope(),
                    wrapper_function=_awrap if is_async else _wrap,
                )
                for object_name, is_async in (
                    ("TypeSafeClient", False),
                    ("AsyncTypeSafeClient", True),
                )
            ]
        )

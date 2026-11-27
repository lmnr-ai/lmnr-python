"""What `observe` does with a mocked call: decide whether to ask the server,
ask it, apply the outcome, and mark the span."""

from __future__ import annotations

import inspect
import typing
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generator,
    Iterable,
    Iterator,
)
from functools import cached_property
from typing import Any, cast

import pydantic
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.tracing.attributes import MOCK_KEY, MOCK_OUTCOME, SPAN_MOCKED
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import is_method

from .client import MockClient, get_client
from .errors import MOCK_FAILURE, MockedError
from .protocol import Outcome, request_body

logger = get_default_logger(__name__)

# Return annotations pydantic has no schema for (or would validate lazily):
# the mocked JSON list is validated as `list[item]` and handed back as one.
_ITERATORS = (Iterator, Iterable, Generator)
_ASYNC_ITERATORS = (AsyncIterator, AsyncIterable, AsyncGenerator)
_Wrap = Callable[[list[Any]], Any]


class MockTarget:
    """One observed function's mock key, built when `observe` wraps it.
    Everything else is worked out on the first mocked call, so decorating
    costs nothing without mocks, and a forward reference in the return
    annotation has had time to be defined."""

    def __init__(self, key: str, fn: Callable[..., Any]):
        self.key: str = key
        self.fn: Callable[..., Any] = fn

    @cached_property
    def is_method(self) -> bool:
        return is_method(self.fn)

    def check_supported(self) -> None:
        unwrapped = inspect.unwrap(self.fn)
        if inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(unwrapped):
            raise MockedError(MOCK_FAILURE, f"{self.key}: generators can't be mocked yet")

    @cached_property
    def _return_type(self) -> tuple[Any, pydantic.TypeAdapter[Any] | None, _Wrap | None]:
        """The return annotation (the awaited type for `async def`), an
        adapter for it, and, for an iterator annotation, what turns the
        validated list into one. No adapter without an annotation, for `Any`,
        or when the annotation can't be resolved or pydantic can't handle it."""
        try:
            hint = typing.get_type_hints(self.fn).get("return", Any)
        except Exception:
            logger.debug(f"observe, mock {self.key!r}: no usable return type", exc_info=True)
            return Any, None, None
        if hint is Any:
            return hint, None, None
        origin = typing.get_origin(hint) or hint
        wrap = _iter if origin in _ITERATORS else _aiter if origin in _ASYNC_ITERATORS else None
        target = hint if wrap is None else list[(typing.get_args(hint) or (Any,))[0]]
        try:
            return hint, pydantic.TypeAdapter(target), wrap
        except Exception as e:
            logger.warning(
                f"observe, mock {self.key!r}: mocked values can't be converted to the return "
                + f"type {_type_name(hint)}, returning them as is: {e}"
            )
            return hint, None, wrap

    def convert(self, value: Any) -> Any:
        """The mocked JSON value as the annotated return type when it fits
        (pydantic models, dataclasses, TypedDicts are built from it; a list
        becomes an iterator), else as is, with a warning: an annotation can
        be broad or wrong, and the function's callers decide whether the
        value works."""
        hint, adapter, wrap = self._return_type
        converted = value
        if adapter is not None:
            try:
                converted = adapter.validate_python(value)
            except pydantic.ValidationError as e:
                logger.warning(
                    f"observe, mock {self.key!r}: the mocked value doesn't match the return type "
                    + f"{_type_name(hint)}, returning it as is: {e}"
                )
        if wrap is not None and isinstance(converted, list):
            return wrap(cast("list[Any]", converted))
        return converted


def mocks_enabled() -> bool:
    return get_client() is not None


def lookup(target: MockTarget | None) -> MockedCall | None:
    """The mocked call to make instead of running the function, or `None` to
    run it as usual. With `LMNR_MOCK_URL` unset this is one cached check."""
    if target is None:
        return None
    client = get_client()
    if client is None or target.key not in client.mocked_keys():
        return None
    target.check_supported()
    return MockedCall(client, target)


async def alookup(target: MockTarget) -> MockedCall | None:
    client = get_client()
    if client is None or target.key not in await client.amocked_keys():
        return None
    target.check_supported()
    return MockedCall(client, target)


class MockedCall:
    def __init__(self, client: MockClient, target: MockTarget):
        self.client: MockClient = client
        self.target: MockTarget = target

    def run(self, span: Span | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            outcome = self.client.call(self.target.key, self._body(span, args, kwargs))
        except MockedError:
            self._mark(span, "error")
            raise
        if outcome.action == "passthrough":
            self._mark(span, "passthrough")
            return self.target.fn(*args, **kwargs)
        return self._settle(span, outcome)

    async def arun(self, span: Span | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            outcome = await self.client.acall(self.target.key, self._body(span, args, kwargs))
        except MockedError:
            self._mark(span, "error")
            raise
        if outcome.action == "passthrough":
            self._mark(span, "passthrough")
            return await self.target.fn(*args, **kwargs)
        return self._settle(span, outcome)

    def _body(self, span: Span | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bytes:
        # Every argument: `ignore_input(s)` and `input_formatter` shape the
        # span, not what the handler sees.
        call_args, call_kwargs = bind_call(self.target.fn, self.target.is_method, args, kwargs)
        trace_id = span_id = None
        if span is not None:
            context = span.get_span_context()
            if context.is_valid:
                trace_id, span_id = f"{context.trace_id:032x}", f"{context.span_id:016x}"
        return request_body(call_args, call_kwargs, trace_id, span_id)

    def _settle(self, span: Span | None, outcome: Outcome) -> Any:
        if outcome.error is not None:
            self._mark(span, "raise", mocked=True)
            raise outcome.error
        self._mark(span, "respond", mocked=True)
        return self.target.convert(outcome.value)

    def _mark(self, span: Span | None, outcome: str, mocked: bool = False) -> None:
        if span is None:
            return
        try:
            span.set_attribute(MOCK_KEY, self.target.key)
            span.set_attribute(MOCK_OUTCOME, outcome)
            if mocked:
                span.set_attribute(SPAN_MOCKED, True)
        except Exception:
            logger.debug("Failed to set mock attributes", exc_info=True)


def bind_call(
    fn: Callable[..., Any], is_method: bool, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """The call as the protocol's `args` / `kwargs`, bound to the signature
    with defaults applied: every positional-only and positional-or-keyword
    parameter in order, then the `*args` values; keyword-only parameters and
    the `**kwargs` entries by name. A method's `self` / `cls` is dropped. A
    call that doesn't bind (it raises `TypeError` when it runs) is sent as
    passed."""
    try:
        signature = inspect.signature(fn)
        bound = signature.bind(*args, **kwargs)
    except (TypeError, ValueError):
        if is_method:
            args = args[1:]
            kwargs = {k: v for k, v in kwargs.items() if k not in ("self", "cls")}
        return list(args), dict(kwargs)
    bound.apply_defaults()
    positional: list[Any] = []
    named: dict[str, Any] = {}
    for i, (name, param) in enumerate(signature.parameters.items()):
        if is_method and i == 0:
            continue
        value = bound.arguments[name]
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            positional.extend(value)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            named[name] = value
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            named.update(value)
        else:
            positional.append(value)
    return positional, named


def _type_name(tp: Any) -> str:
    return tp.__name__ if isinstance(tp, type) else repr(tp)


# Generators, not `iter()`: `observe` records a generator's items as the
# span output (`_handle_generator`), and anything else as returned.
def _iter(items: list[Any]) -> Iterator[Any]:
    yield from items


async def _aiter(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item

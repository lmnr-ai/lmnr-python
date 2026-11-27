"""What `observe` does with a mocked call: decide whether to ask the server,
ask it, apply the outcome, and mark the span."""

from __future__ import annotations

import inspect
import typing
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator
from functools import cached_property
from typing import Any

import pydantic
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.tracing.attributes import MOCK_KEY, MOCK_OUTCOME, SPAN_MOCKED
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import get_input_from_func_args, is_method

from .client import MockClient, get_client
from .errors import MOCK_FAILURE, MockedError
from .protocol import Outcome, request_body

logger = get_default_logger(__name__)

_STREAMING_ORIGINS = (Generator, Iterator, AsyncGenerator, AsyncIterator)
_UNRESOLVED = object()


class MockTarget:
    """One observed function's mock key, built when `observe` wraps it.
    Everything else is worked out on the first mocked call, so decorating
    costs nothing without mocks, and a forward reference in the return
    annotation has had time to be defined."""

    def __init__(self, key: str, fn: Callable[..., Any]):
        self.key: str = key
        self.fn: Callable[..., Any] = fn
        self._return_type: Any = _UNRESOLVED
        self._adapter: pydantic.TypeAdapter[Any] | None = None

    @cached_property
    def is_method(self) -> bool:
        return is_method(self.fn)

    def check_supported(self) -> None:
        unwrapped = inspect.unwrap(self.fn)
        generator = inspect.isgeneratorfunction(unwrapped) or inspect.isasyncgenfunction(unwrapped)
        if generator or typing.get_origin(self.return_type()) in _STREAMING_ORIGINS:
            raise MockedError(MOCK_FAILURE, f"{self.key}: generators can't be mocked yet")

    def return_type(self) -> Any:
        """The return annotation (the awaited type for `async def`), or `Any`
        when there is none or it can't be resolved."""
        if self._return_type is _UNRESOLVED:
            try:
                hint = typing.get_type_hints(self.fn).get("return", Any)
            except Exception:
                logger.warning(
                    f"observe, mock {self.key!r}: can't resolve the return annotation, "
                    + "mocked values are returned without validation",
                    exc_info=True,
                )
                hint = Any
            self._return_type = hint
            if hint is not Any:
                try:
                    self._adapter = pydantic.TypeAdapter(hint)
                except Exception:
                    logger.warning(
                        f"observe, mock {self.key!r}: pydantic can't validate {hint!r}, "
                        + "mocked values are returned without validation",
                        exc_info=True,
                    )
        return self._return_type

    def validate(self, value: Any) -> Any:
        _ = self.return_type()
        if self._adapter is None:
            return value
        try:
            return self._adapter.validate_python(value)
        except pydantic.ValidationError as e:
            raise MockedError(
                MOCK_FAILURE,
                f"{self.key}: the mocked value doesn't match the return type "
                + f"{_type_name(self._return_type)}: {e}",
            ) from None


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
        # Every argument, bound by name: `ignore_input(s)` and
        # `input_formatter` shape the span, not what the handler sees.
        bound = get_input_from_func_args(
            self.target.fn, is_method=self.target.is_method, func_args=args, func_kwargs=kwargs
        )
        trace_id = span_id = None
        if span is not None:
            context = span.get_span_context()
            if context.is_valid:
                trace_id, span_id = f"{context.trace_id:032x}", f"{context.span_id:016x}"
        return request_body(bound, trace_id, span_id)

    def _settle(self, span: Span | None, outcome: Outcome) -> Any:
        if outcome.error is not None:
            self._mark(span, "raise", mocked=True)
            raise outcome.error
        try:
            value = self.target.validate(outcome.value)
        except MockedError:
            self._mark(span, "error")
            raise
        self._mark(span, "respond", mocked=True)
        return value

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


def _type_name(tp: Any) -> str:
    return tp.__name__ if isinstance(tp, type) else repr(tp)

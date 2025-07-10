from functools import wraps
import logging
import json
import pydantic
import dataclasses
import collections.abc
import types
from typing import Any, AsyncGenerator, Callable, Generator, Literal

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.trace import Span

from lmnr.sdk.utils import get_input_from_func_args, is_method
from lmnr.opentelemetry_lib import MAX_MANUAL_SPAN_PAYLOAD_SIZE
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o

        if isinstance(o, pydantic.BaseModel):
            return o.model_dump()

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        # Handle various sequence types, but not strings or bytes
        if isinstance(o, (list, tuple, set, frozenset)) or (
            isinstance(o, collections.abc.Sequence) and not isinstance(o, (str, bytes))
        ):
            return [self.default(v) for v in o]

        # Handle other iterable types but avoid consuming iterators/generators
        if hasattr(o, "__iter__") and not isinstance(o, (str, bytes, dict)):
            # Check if it's a generator or iterator by checking if it has __next__
            if hasattr(o, "__next__"):
                # It's an iterator/generator, don't consume it
                return str(o)

            # It's an iterable but not an iterator, convert to list
            try:
                return [self.default(v) for v in o]
            except (TypeError, ValueError):
                # If iteration fails, fall back to string
                return str(o)

        if isinstance(o, dict):
            return {k: self.default(v) for k, v in o.items()}

        if hasattr(o, "to_json"):
            try:
                result = o.to_json()
                # Only parse back if it's a string (likely JSON)
                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        return result
                return result
            except Exception:
                pass

        # Handle objects with json method - parse back JSON strings to avoid double serialization
        if hasattr(o, "json"):
            try:
                result = o.json()
                # Only parse back if it's a string (likely JSON)
                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        return result
                return result
            except Exception:
                pass

        try:
            return super().default(o)
        except Exception:
            try:
                return str(o)
            except Exception:
                return o


def json_dumps(data: dict) -> str:
    try:
        return json.dumps(data, cls=CustomJSONEncoder)
    except Exception:
        # Log the exception and return a placeholder if serialization completely fails
        logging.warning("Failed to serialize data to JSON, type: %s", type(data))
        return "{}"  # Return an empty JSON object as a fallback


def _setup_span(
    span_name: str, span_type: str, association_properties: dict[str, Any] | None
):
    """Set up a span with the given name, type, and association properties."""
    with get_tracer() as tracer:
        span = tracer.start_span(span_name, attributes={SPAN_TYPE: span_type})

        if association_properties is not None:
            for key, value in association_properties.items():
                span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)

        return span


def _process_input(
    span: Span,
    fn: Callable,
    args: tuple,
    kwargs: dict,
    ignore_input: bool,
    ignore_inputs: list[str] | None,
    input_formatter: Callable[..., str] | None,
):
    """Process and set input attributes on the span."""
    if ignore_input:
        return

    try:
        if input_formatter is not None:
            inp = input_formatter(*args, **kwargs)
            if not isinstance(inp, str):
                inp = json_dumps(inp)
        else:
            inp = json_dumps(
                get_input_from_func_args(
                    fn,
                    is_method=is_method(fn),
                    func_args=args,
                    func_kwargs=kwargs,
                    ignore_inputs=ignore_inputs,
                )
            )

        if len(inp) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
            span.set_attribute(SPAN_INPUT, "Laminar: input too large to record")
        else:
            span.set_attribute(SPAN_INPUT, inp)
    except Exception:
        msg = "Failed to process input, ignoring"
        if input_formatter is not None:
            # Only warn the user if they provided an input formatter
            # because it's their responsibility to make sure it works.
            logger.warning(msg, exc_info=True)
        else:
            logger.debug(msg, exc_info=True)
        pass


def _process_output(
    span: Span,
    result: Any,
    ignore_output: bool,
    output_formatter: Callable[..., str] | None,
):
    """Process and set output attributes on the span."""
    if ignore_output:
        return

    try:
        if output_formatter is not None:
            output = output_formatter(result)
            if not isinstance(output, str):
                output = json_dumps(output)
        else:
            output = json_dumps(result)

        if len(output) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
            span.set_attribute(SPAN_OUTPUT, "Laminar: output too large to record")
        else:
            span.set_attribute(SPAN_OUTPUT, output)
    except Exception:
        msg = "Failed to process output, ignoring"
        if output_formatter is not None:
            # Only warn the user if they provided an output formatter
            # because it's their responsibility to make sure it works.
            logger.warning(msg, exc_info=True)
        else:
            logger.debug(msg, exc_info=True)
        pass


def _cleanup_span(span: Span, ctx_token):
    """Clean up span and context."""
    span.end()
    context_api.detach(ctx_token)


def observe_base(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            span_name = name or fn.__name__

            span = _setup_span(span_name, span_type, association_properties)
            ctx = trace.set_span_in_context(span, context_api.get_current())
            ctx_token = context_api.attach(ctx)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = fn(*args, **kwargs)
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, ctx_token)
                raise e

            # span will be ended in the generator
            if isinstance(res, types.GeneratorType):
                return _handle_generator(span, ctx_token, res)
            if isinstance(res, types.AsyncGeneratorType):
                # async def foo() -> AsyncGenerator[int, None]:
                # is not considered async in a classical sense in Python,
                # so we handle this inside the sync wrapper.
                # In particular, CO_COROUTINE is different from CO_ASYNC_GENERATOR.
                # Flags are listed from LSB here:
                # https://docs.python.org/3/library/inspect.html#inspect-module-co-flags
                # See also: https://groups.google.com/g/python-tulip/c/6rWweGXLutU?pli=1
                return _ahandle_generator(span, ctx_token, res)

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, ctx_token)
            return res

        return wrap

    return decorate


# Async Decorators
def async_observe_base(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__

            span = _setup_span(span_name, span_type, association_properties)
            ctx = trace.set_span_in_context(span, context_api.get_current())
            ctx_token = context_api.attach(ctx)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = await fn(*args, **kwargs)
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, ctx_token)
                raise e

            # span will be ended in the generator
            if isinstance(res, types.AsyncGeneratorType):
                # probably unreachable, read the comment in the similar
                # part of the sync wrapper.
                return await _ahandle_generator(span, ctx_token, res)

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, ctx_token)
            return res

        return wrap

    return decorate


def _handle_generator(span: Span, ctx_token, res: Generator[Any, Any, Any]):
    yield from res

    span.end()
    if ctx_token is not None:
        context_api.detach(ctx_token)


async def _ahandle_generator(span: Span, ctx_token, res: AsyncGenerator[Any, Any]):
    # async with contextlib.aclosing(res) as closing_gen:
    async for part in res:
        yield part

    span.end()
    if ctx_token is not None:
        context_api.detach(ctx_token)


def _process_exception(span: Span, e: Exception):
    # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
    span.record_exception(e, escaped=True)

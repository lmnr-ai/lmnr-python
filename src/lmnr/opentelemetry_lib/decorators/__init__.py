from functools import wraps
import pydantic
import orjson
import types
from typing import Any, AsyncGenerator, Callable, Generator, Literal, TypeVar

from opentelemetry import context as context_api
from opentelemetry.trace import Span, Status, StatusCode

from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_USER_ID_KEY,
    attach_context,
    detach_context,
    get_event_attributes_from_context,
)
from lmnr.sdk.utils import get_input_from_func_args, is_method
from lmnr.opentelemetry_lib import MAX_MANUAL_SPAN_PAYLOAD_SIZE
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    SPAN_INPUT,
    SPAN_OUTPUT,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_PLACEHOLDER = {}


def default_json(o):
    if isinstance(o, pydantic.BaseModel):
        return o.model_dump()

    # Handle various sequence types, but not strings or bytes
    if isinstance(o, (list, tuple, set, frozenset)):
        return list(o)

    try:
        return str(o)
    except Exception:
        logger.debug("Failed to serialize data to JSON, inner type: %s", type(o))
        pass
    return DEFAULT_PLACEHOLDER


def json_dumps(data: dict) -> str:
    try:
        return orjson.dumps(
            data,
            default=default_json,
            option=orjson.OPT_SERIALIZE_DATACLASS
            | orjson.OPT_SERIALIZE_UUID
            | orjson.OPT_UTC_Z
            | orjson.OPT_NON_STR_KEYS,
        ).decode("utf-8")
    except Exception:
        # Log the exception and return a placeholder if serialization completely fails
        logger.info("Failed to serialize data to JSON, type: %s", type(data))
        return "{}"  # Return an empty JSON object as a fallback


def _setup_span(
    span_name: str,
    span_type: str,
    association_properties: dict[str, Any] | None,
    preserve_global_context: bool = False,
):
    """Set up a span with the given name, type, and association properties."""
    with get_tracer_with_context() as (tracer, isolated_context):
        # Create span in isolated context
        span = tracer.start_span(
            span_name,
            context=isolated_context if not preserve_global_context else None,
            attributes={SPAN_TYPE: span_type},
        )

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


def _cleanup_span(span: Span, wrapper: TracerWrapper):
    """Clean up span and context."""
    span.end()
    wrapper.pop_span_context()


def observe_base(
    *,
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    preserve_global_context: bool = False,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            span_name = name or fn.__name__
            wrapper = TracerWrapper()

            span = _setup_span(
                span_name, span_type, association_properties, preserve_global_context
            )
            new_context = wrapper.push_span_context(span)
            if session_id := association_properties.get("session_id"):
                new_context = context_api.set_value(
                    CONTEXT_SESSION_ID_KEY, session_id, new_context
                )
            if user_id := association_properties.get("user_id"):
                new_context = context_api.set_value(
                    CONTEXT_USER_ID_KEY, user_id, new_context
                )
            # Some auto-instrumentations are not under our control, so they
            # don't have access to our isolated context. We attach the context
            # to the OTEL global context, so that spans know their parent
            # span and trace_id.
            ctx_token = context_api.attach(new_context)
            # update our isolated context too
            isolated_ctx_token = attach_context(new_context)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = fn(*args, **kwargs)
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, wrapper)
                raise
            finally:
                # Always restore global context
                context_api.detach(ctx_token)
                detach_context(isolated_ctx_token)
            # span will be ended in the generator
            if isinstance(res, types.GeneratorType):
                return _handle_generator(span, wrapper, res)
            if isinstance(res, types.AsyncGeneratorType):
                # async def foo() -> AsyncGenerator[int, None]:
                # is not considered async in a classical sense in Python,
                # so we handle this inside the sync wrapper.
                # In particular, CO_COROUTINE is different from CO_ASYNC_GENERATOR.
                # Flags are listed from LSB here:
                # https://docs.python.org/3/library/inspect.html#inspect-module-co-flags
                # See also: https://groups.google.com/g/python-tulip/c/6rWweGXLutU?pli=1
                return _ahandle_generator(span, wrapper, res)

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, wrapper)
            return res

        return wrap

    return decorate


# Async Decorators
def async_observe_base(
    *,
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    preserve_global_context: bool = False,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__
            wrapper = TracerWrapper()

            span = _setup_span(
                span_name, span_type, association_properties, preserve_global_context
            )
            new_context = wrapper.push_span_context(span)
            if session_id := association_properties.get("session_id"):
                new_context = context_api.set_value(
                    CONTEXT_SESSION_ID_KEY, session_id, new_context
                )
            if user_id := association_properties.get("user_id"):
                new_context = context_api.set_value(
                    CONTEXT_USER_ID_KEY, user_id, new_context
                )
            # Some auto-instrumentations are not under our control, so they
            # don't have access to our isolated context. We attach the context
            # to the OTEL global context, so that spans know their parent
            # span and trace_id.
            ctx_token = context_api.attach(new_context)
            # update our isolated context too
            isolated_ctx_token = attach_context(new_context)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = await fn(*args, **kwargs)
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, wrapper)
                raise e
            finally:
                # Always restore global context
                context_api.detach(ctx_token)
                detach_context(isolated_ctx_token)

            # span will be ended in the generator
            if isinstance(res, types.AsyncGeneratorType):
                # probably unreachable, read the comment in the similar
                # part of the sync wrapper.
                return await _ahandle_generator(span, wrapper, res)

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, wrapper)
            return res

        return wrap

    return decorate


def _handle_generator(
    span: Span,
    wrapper: TracerWrapper,
    res: Generator,
    ignore_output: bool = False,
    output_formatter: Callable[..., str] | None = None,
):
    results = []
    try:
        for part in res:
            results.append(part)
            yield part
    finally:
        _process_output(span, results, ignore_output, output_formatter)
        _cleanup_span(span, wrapper)


async def _ahandle_generator(
    span: Span,
    wrapper: TracerWrapper,
    res: AsyncGenerator,
    ignore_output: bool = False,
    output_formatter: Callable[..., str] | None = None,
):
    results = []
    try:
        async for part in res:
            results.append(part)
            yield part
    finally:
        _process_output(span, results, ignore_output, output_formatter)
        _cleanup_span(span, wrapper)


def _process_exception(span: Span, e: Exception):
    # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
    span.record_exception(
        e, attributes=get_event_attributes_from_context(), escaped=True
    )
    span.set_status(Status(StatusCode.ERROR, str(e)))

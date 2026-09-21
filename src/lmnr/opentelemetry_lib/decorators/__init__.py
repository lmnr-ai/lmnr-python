import asyncio
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal, TypeVar, cast

from opentelemetry import context as context_api
from opentelemetry.sdk.trace import Span as SdkSpan
from opentelemetry.trace import Span, Status, StatusCode

from lmnr.opentelemetry_lib.tracing import is_tracing_initialized
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    METADATA,
    SPAN_TYPE,
)
from lmnr.opentelemetry_lib.tracing.context import (
    CONTEXT_METADATA_KEY,
    get_event_attributes_from_context,
    pop_span_context,
    push_span,
)
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.opentelemetry_lib.tracing.utils import set_association_props_in_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import (
    get_input_from_func_args,
    is_method,
    is_otel_attribute_value_type,
    json_dumps,
)

logger = get_default_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])  # pyright: ignore[reportExplicitAny]


def _setup_span(
    span_name: str,
    span_type: str,
    association_properties: dict[str, Any] | None,  # pyright: ignore[reportExplicitAny]
    preserve_global_context: bool = False,
    metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
) -> Span | None:
    """Set up a span with the given name, type, and association properties."""
    span = None
    try:
        with get_tracer_with_context() as (tracer, isolated_context):
            # Create span in isolated context
            span = tracer.start_span(
                span_name,
                context=isolated_context if not preserve_global_context else None,
                attributes={SPAN_TYPE: span_type},
            )

            ctx_metadata = cast(dict[str, Any], context_api.get_value(CONTEXT_METADATA_KEY, isolated_context))  # pyright: ignore[reportExplicitAny]
            merged_metadata = {
                **(ctx_metadata or {}),
                **(metadata or {}),
            }
            for key, value in merged_metadata.items():  # pyright: ignore[reportAny]
                span.set_attribute(
                    f"{ASSOCIATION_PROPERTIES}.{METADATA}.{key}",
                    (
                        value
                        if is_otel_attribute_value_type(value)  # pyright: ignore[reportAny]
                        else json_dumps(value)  # pyright: ignore[reportAny]
                    ),
                )

            if association_properties is not None:
                for key, value in association_properties.items():  # pyright: ignore[reportAny]
                    span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)  # pyright: ignore[reportAny]

            return span
    except Exception:
        logger.warning(f"[observe] failed to setup span: {span_name}", exc_info=True)
        return span


def _process_input(
    span: Span,
    fn: Callable[..., Any],  # pyright: ignore[reportExplicitAny]
    args: tuple[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
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
        else:
            inp = get_input_from_func_args(
                fn,
                is_method=is_method(fn),
                func_args=args,
                func_kwargs=kwargs,
                ignore_inputs=ignore_inputs,
            )

        if not isinstance(span, LaminarSpan):
            span = LaminarSpan(cast(SdkSpan, span))
        span.set_input(inp)
    except Exception:
        msg = "Failed to process input, ignoring"
        if input_formatter is not None:
            # Only warn the user if they provided an input formatter
            # because it's their responsibility to make sure it works.
            logger.warning(msg, exc_info=True)
        else:
            logger.debug(msg, exc_info=True)


def _process_output(
    span: Span,
    result: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ignore_output: bool,
    output_formatter: Callable[..., str] | None,
):
    """Process and set output attributes on the span."""
    if ignore_output:
        return

    try:
        if output_formatter is not None:
            output = output_formatter(result)
        else:
            output = result  # pyright: ignore[reportAny]

        if not isinstance(span, LaminarSpan):
            span = LaminarSpan(cast(SdkSpan, span))
        span.set_output(output)
    except Exception:
        msg = "Failed to process output, ignoring"
        if output_formatter is not None:
            # Only warn the user if they provided an output formatter
            # because it's their responsibility to make sure it works.
            logger.warning(msg, exc_info=True)
        else:
            logger.debug(msg, exc_info=True)


def _cleanup_span(span: Span, do_pop_context: bool = True):
    """Clean up span and context."""
    try:
        span.end()
    except Exception:
        logger.debug("Failed to end span in _cleanup_span", exc_info=True)
    if not do_pop_context:
        return
    try:
        pop_span_context()
    except Exception:
        logger.debug("Failed to pop span context in _cleanup_span", exc_info=True)


def observe_base(
    *,
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal[
        "DEFAULT",
        "LLM",
        "TOOL",
        "EXECUTOR",
        "EVALUATOR",
        "HUMAN_EVALUATOR",
        "EVALUATION",
    ] = "DEFAULT",
    metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    association_properties: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    preserve_global_context: bool = False,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        @wraps(fn)
        def wrap(*args: Any, **kwargs: Any):   # pyright: ignore[reportExplicitAny, reportAny]:
            if not is_tracing_initialized():
                return fn(*args, **kwargs)  # pyright: ignore[reportAny]

            span_name = name or getattr(fn, "__name__", "unknown")

            span = _setup_span(
                span_name,
                span_type,
                association_properties,
                preserve_global_context,
                metadata,
            )

            if span is None:
                return fn(*args, **kwargs)  # pyright: ignore[reportAny])

            # Set association props in context before push_span_context
            # so child spans inherit them
            assoc_props_token = set_association_props_in_context(span)
            if assoc_props_token and isinstance(span, LaminarSpan):
                span.lmnr_assoc_props_token = assoc_props_token

            ctx_token = None
            current_task = None
            current_context_id = None
            did_push_context = False
            try:
                try:
                    current_task = asyncio.current_task()
                except Exception:
                    current_task = None
                current_context_id = id(current_task)
                new_context = push_span(span)
                did_push_context = True
                # Some auto-instrumentations are not under our control, so they
                # don't have access to our isolated context. We attach the context
                # to the OTEL global context, so that spans know their parent
                # span and trace_id.
                ctx_token = context_api.attach(new_context)
            except Exception:
                logger.debug("Failed to setup span context", exc_info=True)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = fn(*args, **kwargs)  # pyright: ignore[reportAny]
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, did_push_context)
                raise
            finally:
                current_task = None
                try:
                    current_task = asyncio.current_task()
                except Exception:
                    current_task = None
                # Always restore global context if we are in the same asyncio context
                if id(current_task) == current_context_id:
                    try:
                        if ctx_token is not None:
                            context_api.detach(ctx_token)
                    except Exception:
                        logger.debug("Failed to detach global context", exc_info=True)
                else:
                    logger.debug(
                        "Not detaching global context, not in the same context"
                    )
            # span will be ended in the generator
            if isinstance(res, types.GeneratorType):
                return _handle_generator(
                    span,
                    res,
                    ignore_output,
                    output_formatter,
                    did_push_context,
                )
            if isinstance(res, types.AsyncGeneratorType):
                # async def foo() -> AsyncGenerator[int, None]:
                # is not considered async in a classical sense in Python,
                # so we handle this inside the sync wrapper.
                # In particular, CO_COROUTINE is different from CO_ASYNC_GENERATOR.
                # Flags are listed from LSB here:
                # https://docs.python.org/3/library/inspect.html#inspect-module-co-flags
                # See also: https://groups.google.com/g/python-tulip/c/6rWweGXLutU?pli=1
                return _ahandle_generator(
                    span,
                    res,
                    ignore_output,
                    output_formatter,
                    did_push_context,
                )

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, did_push_context)
            return res  # pyright: ignore[reportAny]

        return cast(F, wrap)

    return decorate


# Async Decorators
def async_observe_base(
    *,
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal[
        "DEFAULT",
        "LLM",
        "TOOL",
        "EXECUTOR",
        "EVALUATOR",
        "HUMAN_EVALUATOR",
        "EVALUATION",
    ] = "DEFAULT",
    metadata: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    association_properties: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    input_formatter: Callable[..., str] | None = None,
    output_formatter: Callable[..., str] | None = None,
    preserve_global_context: bool = False,
) -> Callable[[F], F]:
    def decorate(fn: F) -> F:
        @wraps(fn)
        async def wrap(*args: Any, **kwargs: Any):  # pyright: ignore[reportExplicitAny, reportAny]
            if not is_tracing_initialized():
                return await fn(*args, **kwargs)  # pyright: ignore[reportAny]

            span_name = name or getattr(fn, "__name__", "unknown")

            span = _setup_span(
                span_name,
                span_type,
                association_properties,
                preserve_global_context,
                metadata,
            )

            if span is None:
                return await fn(*args, **kwargs)  # pyright: ignore[reportAny]

            # Set association props in context before push_span_context
            # so child spans inherit them
            assoc_props_token = set_association_props_in_context(span)
            if assoc_props_token and isinstance(span, LaminarSpan):
                span.lmnr_assoc_props_token = assoc_props_token

            ctx_token = None
            current_task = None
            current_context_id = None
            did_push_context = False
            try:
                try:
                    current_task = asyncio.current_task()
                except Exception:
                    current_task = None
                current_context_id = id(current_task)
                new_context = push_span(span)
                did_push_context = True
                # Some auto-instrumentations are not under our control, so they
                # don't have access to our isolated context. We attach the context
                # to the OTEL global context, so that spans know their parent
                # span and trace_id.
                ctx_token = context_api.attach(new_context)
            except Exception:
                logger.debug("Failed to setup span context", exc_info=True)

            _process_input(
                span, fn, args, kwargs, ignore_input, ignore_inputs, input_formatter
            )

            try:
                res = await fn(*args, **kwargs)   # pyright: ignore[reportAny]
            except Exception as e:
                _process_exception(span, e)
                _cleanup_span(span, did_push_context)
                raise
            finally:
                # Always restore global context if we are in the same asyncio context
                current_task = None
                try:
                    current_task = asyncio.current_task()
                except Exception:
                    current_task = None
                if id(current_task) == current_context_id:
                    try:
                        if ctx_token is not None:
                            context_api.detach(ctx_token)
                    except Exception:
                        logger.debug("Failed to detach global context", exc_info=True)
                else:
                    logger.debug(
                        "Not detaching global context, not in the same context"
                    )

            # span will be ended in the generator
            if isinstance(res, types.AsyncGeneratorType):
                # probably unreachable, read the comment in the similar
                # part of the sync wrapper.
                return _ahandle_generator(
                    span,
                    res,
                    ignore_output,
                    output_formatter,
                    did_push_context,
                )

            _process_output(span, res, ignore_output, output_formatter)
            _cleanup_span(span, did_push_context)
            return res  # pyright: ignore[reportAny]

        return cast(F, wrap)

    return decorate


def _handle_generator(
    span: Span,
    res: types.GeneratorType[Any, Any, Any],  # pyright: ignore[reportExplicitAny]
    ignore_output: bool = False,
    output_formatter: Callable[..., str] | None = None,
    did_push_context: bool = True,
):
    results = []
    try:
        for part in res:  # pyright: ignore[reportAny]
            results.append(part)  # pyright: ignore[reportAny, reportUnknownMemberType]
            yield part
    except Exception as e:
        _process_exception(span, e)
        raise
    finally:
        _process_output(span, results, ignore_output, output_formatter)
        _cleanup_span(span, did_push_context)


async def _ahandle_generator(
    span: Span,
    res: types.AsyncGeneratorType[Any, Any],  # pyright: ignore[reportExplicitAny]
    ignore_output: bool = False,
    output_formatter: Callable[..., str] | None = None,
    did_push_context: bool = True,
):
    results = []
    try:
        async for part in res:  # pyright: ignore[reportAny]
            results.append(part)  # pyright: ignore[reportAny, reportUnknownMemberType]
            yield part
    except Exception as e:
        _process_exception(span, e)
        raise
    finally:
        _process_output(span, results, ignore_output, output_formatter)
        _cleanup_span(span, did_push_context)


def _process_exception(span: Span, e: Exception):
    try:
        # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
        span.record_exception(
            e, attributes=get_event_attributes_from_context(), escaped=True
        )
        span.set_status(Status(StatusCode.ERROR, str(e)))
    except Exception:
        logger.debug("Failed to process exception", exc_info=True)

from functools import wraps
import json
import logging
import pydantic
import types
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.trace import Span

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
from lmnr.opentelemetry_lib.utils.json_encoder import JSONEncoder


class CustomJSONEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, pydantic.BaseModel):
            return o.model_dump_json()
        try:
            return super().default(o)
        except TypeError:
            return str(o)  # Fallback to string representation for unsupported types


def json_dumps(data: dict) -> str:
    try:
        return json.dumps(data, cls=CustomJSONEncoder)
    except Exception:
        # Log the exception and return a placeholder if serialization completely fails
        logging.warning("Failed to serialize data to JSON, type: %s", type(data))
        return "{}"  # Return an empty JSON object as a fallback


def entity_method(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            span_name = name or fn.__name__
            wrapper = TracerWrapper()

            with get_tracer_with_context() as (tracer, isolated_context):
                # Create span in isolated context
                span = tracer.start_span(
                    span_name,
                    context=isolated_context,
                    attributes={SPAN_TYPE: span_type},
                )

                if association_properties is not None:
                    for key, value in association_properties.items():
                        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)

                # Set up context for this span and update isolated context
                new_context = trace.set_span_in_context(span, isolated_context)
                wrapper.set_isolated_context(new_context)

                # Also set up global context for nested OpenTelemetry instrumentation
                ctx_token = context_api.attach(new_context)

                try:
                    if not ignore_input:
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
                            span.set_attribute(
                                SPAN_INPUT, "Laminar: input too large to record"
                            )
                        else:
                            span.set_attribute(SPAN_INPUT, inp)
                except TypeError:
                    pass

                try:
                    res = fn(*args, **kwargs)
                except Exception as e:
                    _process_exception(span, e)
                    span.end()
                    raise e
                finally:
                    # Always restore context
                    context_api.detach(ctx_token)

                # span will be ended in the generator
                if isinstance(res, types.GeneratorType):
                    return _handle_generator(span, res)
                if isinstance(res, types.AsyncGeneratorType):
                    return _ahandle_generator(span, res)

                try:
                    if not ignore_output:
                        output = json_dumps(res)
                        if len(output) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                            span.set_attribute(
                                SPAN_OUTPUT, "Laminar: output too large to record"
                            )
                        else:
                            span.set_attribute(SPAN_OUTPUT, output)
                except TypeError:
                    pass

                span.end()
                return res

        return wrap

    return decorate


# Async Decorators
def aentity_method(
    name: str | None = None,
    ignore_input: bool = False,
    ignore_inputs: list[str] | None = None,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    association_properties: dict[str, Any] | None = None,
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__
            wrapper = TracerWrapper()

            with get_tracer_with_context() as (tracer, isolated_context):
                # Create span in isolated context
                span = tracer.start_span(
                    span_name,
                    context=isolated_context,
                    attributes={SPAN_TYPE: span_type},
                )

                if association_properties is not None:
                    for key, value in association_properties.items():
                        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)

                # Set up context for this span and update isolated context
                new_context = trace.set_span_in_context(span, isolated_context)
                wrapper.set_isolated_context(new_context)

                # Also set up global context for nested OpenTelemetry instrumentation
                ctx_token = context_api.attach(new_context)

                try:
                    if not ignore_input:
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
                            span.set_attribute(
                                SPAN_INPUT, "Laminar: input too large to record"
                            )
                        else:
                            span.set_attribute(SPAN_INPUT, inp)
                except TypeError:
                    pass

                try:
                    res = await fn(*args, **kwargs)
                except Exception as e:
                    _process_exception(span, e)
                    span.end()
                    raise e
                finally:
                    # Always restore context
                    context_api.detach(ctx_token)

                # span will be ended in the generator
                if isinstance(res, types.AsyncGeneratorType):
                    return await _ahandle_generator(span, res)

                try:
                    if not ignore_output:
                        output = json_dumps(res)
                        if len(output) > MAX_MANUAL_SPAN_PAYLOAD_SIZE:
                            span.set_attribute(
                                SPAN_OUTPUT, "Laminar: output too large to record"
                            )
                        else:
                            span.set_attribute(SPAN_OUTPUT, output)
                except TypeError:
                    pass

                span.end()
                return res

        return wrap

    return decorate


def _handle_generator(span, res):
    yield from res
    span.end()


async def _ahandle_generator(span, res):
    # async with contextlib.aclosing(res) as closing_gen:
    async for part in res:
        yield part
    span.end()


def _process_exception(span: Span, e: Exception):
    # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
    span.record_exception(e, escaped=True)

from functools import wraps
import json
import logging
import pydantic
import types
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.tracing.context_properties import (
    remove_association_properties,
    update_association_properties,
)
from lmnr.sdk.utils import get_input_from_func_args, is_method
from lmnr.opentelemetry_lib import MAX_MANUAL_SPAN_PAYLOAD_SIZE
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT, SPAN_TYPE
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

            if association_properties is not None:
                update_association_properties(association_properties)

            with get_tracer() as tracer:
                span = tracer.start_span(span_name, attributes={SPAN_TYPE: span_type})

                ctx = trace.set_span_in_context(span, context_api.get_current())
                ctx_token = context_api.attach(ctx)

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
                context_api.detach(ctx_token)
                if association_properties is not None:
                    remove_association_properties(association_properties)
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

            if association_properties is not None:
                update_association_properties(association_properties)

            with get_tracer() as tracer:
                span = tracer.start_span(span_name, attributes={SPAN_TYPE: span_type})

                ctx = trace.set_span_in_context(span, context_api.get_current())
                ctx_token = context_api.attach(ctx)

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

                # span will be ended in the generator
                if isinstance(res, types.AsyncGeneratorType):
                    # probably unreachable, read the comment in the similar
                    # part of the sync wrapper.
                    return await _ahandle_generator(span, ctx_token, res)

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
                if association_properties is not None:
                    remove_association_properties(association_properties)
                context_api.detach(ctx_token)

                return res

        return wrap

    return decorate


def _handle_generator(span, ctx_token, res):
    yield from res

    span.end()
    if ctx_token is not None:
        context_api.detach(ctx_token)


async def _ahandle_generator(span, ctx_token, res):
    # async with contextlib.aclosing(res) as closing_gen:
    async for part in res:
        yield part

    span.end()
    if ctx_token is not None:
        context_api.detach(ctx_token)


def _process_exception(span: Span, e: Exception):
    # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
    span.record_exception(e, escaped=True)

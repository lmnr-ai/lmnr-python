import json
from functools import wraps
import logging
import pydantic
import types
from typing import Any, Literal, Optional, Union

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.trace import Span

from lmnr.sdk.utils import get_input_from_func_args, is_method
from lmnr.opentelemetry_lib.tracing import get_tracer
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT, SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.tracing import TracerWrapper
from lmnr.opentelemetry_lib.utils.json_encoder import JSONEncoder
from lmnr.opentelemetry_lib.config import MAX_MANUAL_SPAN_PAYLOAD_SIZE


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
    name: Optional[str] = None,
    ignore_input: bool = False,
    ignore_inputs: Optional[list[str]] = None,
    ignore_output: bool = False,
    span_type: Union[Literal["DEFAULT"], Literal["LLM"], Literal["TOOL"]] = "DEFAULT",
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            span_name = name or fn.__name__

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
                    return _handle_generator(span, res)

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

                return res

        return wrap

    return decorate


# Async Decorators
def aentity_method(
    name: Optional[str] = None,
    ignore_input: bool = False,
    ignore_inputs: Optional[list[str]] = None,
    ignore_output: bool = False,
    span_type: Union[Literal["DEFAULT"], Literal["LLM"], Literal["TOOL"]] = "DEFAULT",
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__

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
                context_api.detach(ctx_token)

                return res

        return wrap

    return decorate


def _handle_generator(span, res):
    # for some reason the SPAN_KEY is not being set in the context of the generator, so we re-set it
    context_api.attach(trace.set_span_in_context(span))
    yield from res

    span.end()

    # Note: we don't detach the context here as this fails in some situations
    # https://github.com/open-telemetry/opentelemetry-python/issues/2606
    # This is not a problem since the context will be detached automatically during garbage collection


async def _ahandle_generator(span, ctx_token, res):
    async for part in res:
        yield part

    span.end()
    context_api.detach(ctx_token)


def _process_exception(span: Span, e: Exception):
    # Note that this `escaped` is sent as a StringValue("True"), not a boolean.
    span.record_exception(e, escaped=True)

import json
from functools import wraps
import logging
import os
import pydantic
import types
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry import context as context_api

from lmnr.sdk.utils import get_input_from_func_args, is_method
from lmnr.traceloop_sdk.tracing import get_tracer
from lmnr.traceloop_sdk.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT, SPAN_PATH
from lmnr.traceloop_sdk.tracing.tracing import TracerWrapper, get_span_path
from lmnr.traceloop_sdk.utils.json_encoder import JSONEncoder


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
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            span_name = name or fn.__name__

            with get_tracer() as tracer:
                span = tracer.start_span(span_name)

                span_path = get_span_path(span_name)
                span.set_attribute(SPAN_PATH, span_path)
                ctx = context_api.set_value("span_path", span_path)

                ctx = trace.set_span_in_context(span, ctx)
                ctx_token = context_api.attach(ctx)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SPAN_INPUT,
                            json_dumps(
                                get_input_from_func_args(
                                    fn, is_method(fn), args, kwargs
                                )
                            ),
                        )
                except TypeError:
                    pass

                res = fn(*args, **kwargs)

                # span will be ended in the generator
                if isinstance(res, types.GeneratorType):
                    return _handle_generator(span, res)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SPAN_OUTPUT,
                            json_dumps(res),
                        )
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
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return await fn(*args, **kwargs)

            span_name = name or fn.__name__

            with get_tracer() as tracer:
                span = tracer.start_span(span_name)

                span_path = get_span_path(span_name)
                span.set_attribute(SPAN_PATH, span_path)
                ctx = context_api.set_value("span_path", span_path)

                ctx = trace.set_span_in_context(span, ctx)
                ctx_token = context_api.attach(ctx)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SPAN_INPUT,
                            json_dumps(
                                get_input_from_func_args(
                                    fn, is_method(fn), args, kwargs
                                )
                            ),
                        )
                except TypeError:
                    pass

                res = await fn(*args, **kwargs)

                # span will be ended in the generator
                if isinstance(res, types.AsyncGeneratorType):
                    return await _ahandle_generator(span, ctx_token, res)

                try:
                    if _should_send_prompts():
                        span.set_attribute(SPAN_OUTPUT, json_dumps(res))
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


def _should_send_prompts():
    return (
        os.getenv("TRACELOOP_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")

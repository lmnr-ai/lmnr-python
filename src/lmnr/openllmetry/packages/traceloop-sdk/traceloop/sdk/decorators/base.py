import json
from functools import wraps
import os
import types
from typing import Any, Optional
import warnings

from opentelemetry import trace
from opentelemetry import context as context_api
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues

from traceloop.sdk.tracing import get_tracer, set_workflow_name
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_entity_path,
    get_chained_entity_path,
)
from traceloop.sdk.utils import camel_to_snake
from traceloop.sdk.utils.json_encoder import JSONEncoder


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)  # Fallback to string representation for unsupported types


def _json_dumps(data: dict) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return json.dumps(data, cls=CustomJSONEncoder)
    except Exception as e:
        # Log the exception and return a placeholder if serialization completely fails
        # Telemetry().log_exception(e)
        return "{}"  # Return an empty JSON object as a fallback


def entity_method(
    name: Optional[str] = None,
    version: Optional[int] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorate(fn):
        @wraps(fn)
        def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                return fn(*args, **kwargs)

            entity_name = name or fn.__name__
            if tlp_span_kind in [
                TraceloopSpanKindValues.WORKFLOW,
                TraceloopSpanKindValues.AGENT,
            ]:
                set_workflow_name(entity_name)
            span_name = f"{entity_name}.{tlp_span_kind.value}"

            with get_tracer() as tracer:
                span = tracer.start_span(span_name)
                ctx = trace.set_span_in_context(span)
                ctx_token = context_api.attach(ctx)

                if tlp_span_kind in [
                    TraceloopSpanKindValues.TASK,
                    TraceloopSpanKindValues.TOOL,
                ]:
                    entity_path = get_chained_entity_path(entity_name)
                    set_entity_path(entity_path)

                span.set_attribute(
                    SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value
                )
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
                if version:
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_INPUT,
                            _json_dumps({"args": args, "kwargs": kwargs}),
                        )
                except TypeError as e:
                    pass

                res = fn(*args, **kwargs)

                # span will be ended in the generator
                if isinstance(res, types.GeneratorType):
                    return _handle_generator(span, res)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                            _json_dumps(res),
                        )
                except TypeError as e:
                    pass

                span.end()
                context_api.detach(ctx_token)

                return res

        return wrap

    return decorate


def entity_class(
    name: Optional[str],
    version: Optional[int],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            entity_method(name=task_name, version=version, tlp_span_kind=tlp_span_kind)(
                method
            ),
        )
        return cls

    return decorator


# Async Decorators


def aentity_method(
    name: Optional[str] = None,
    version: Optional[int] = None,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorate(fn):
        @wraps(fn)
        async def wrap(*args, **kwargs):
            if not TracerWrapper.verify_initialized():
                print("Tracer not initialized")
                return await fn(*args, **kwargs)

            entity_name = name or fn.__name__
            if tlp_span_kind in [
                TraceloopSpanKindValues.WORKFLOW,
                TraceloopSpanKindValues.AGENT,
            ]:
                set_workflow_name(entity_name)
            span_name = f"{entity_name}.{tlp_span_kind.value}"

            with get_tracer() as tracer:
                span = tracer.start_span(span_name)
                ctx = trace.set_span_in_context(span)
                ctx_token = context_api.attach(ctx)

                if tlp_span_kind in [
                    TraceloopSpanKindValues.TASK,
                    TraceloopSpanKindValues.TOOL,
                ]:
                    entity_path = get_chained_entity_path(entity_name)
                    set_entity_path(entity_path)

                span.set_attribute(
                    SpanAttributes.TRACELOOP_SPAN_KIND, tlp_span_kind.value
                )
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, entity_name)
                if version:
                    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_VERSION, version)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_INPUT,
                            _json_dumps({"args": args, "kwargs": kwargs}),
                        )
                except TypeError as e:
                    pass

                res = await fn(*args, **kwargs)

                # span will be ended in the generator
                if isinstance(res, types.AsyncGeneratorType):
                    return await _ahandle_generator(span, ctx_token, res)

                try:
                    if _should_send_prompts():
                        span.set_attribute(
                            SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(res)
                        )
                except TypeError as e:
                    pass

                span.end()
                context_api.detach(ctx_token)

                return res

        return wrap

    return decorate


def aentity_class(
    name: Optional[str],
    version: Optional[int],
    method_name: str,
    tlp_span_kind: Optional[TraceloopSpanKindValues] = TraceloopSpanKindValues.TASK,
):
    def decorator(cls):
        task_name = name if name else camel_to_snake(cls.__name__)
        method = getattr(cls, method_name)
        setattr(
            cls,
            method_name,
            aentity_method(
                name=task_name, version=version, tlp_span_kind=tlp_span_kind
            )(method),
        )
        return cls

    return decorator


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
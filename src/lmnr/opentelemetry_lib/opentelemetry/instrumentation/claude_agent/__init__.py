from lmnr import Laminar
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import get_input_from_func_args, is_method
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing import Collection
from wrapt import wrap_function_wrapper

_instruments = ("claude-agent-sdk >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "connect",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "query",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "receive_messages",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": True,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "receive_response",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": True,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "interrupt",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "disconnect",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
    },
    {
        "package": "claude_agent_sdk",
        "object": "",
        "method": "query",
        "is_async": True,
        "is_streaming": True,
    },
]

def _with_wrapper(func):
    """Helper for providing tracer for wrapper functions. Includes metric collectors."""

    def wrapper(
        to_wrap,
    ):
        def wrapper(wrapped, instance, args, kwargs):
            return func(
                to_wrap,
                wrapped,
                instance,
                args,
                kwargs,
            )

        return wrapper

    return wrapper

def _span_name(to_wrap: dict[str, str]) -> str:
    class_name = to_wrap.get("class_name")
    method = to_wrap.get("method")
    return f"{class_name}.{method}" if class_name else method


def _record_input(span, wrapped, args, kwargs):
    try:
        span.set_attribute(
            "lmnr.span.input",
            json_dumps(
                get_input_from_func_args(
                    wrapped,
                    is_method=is_method(wrapped),
                    func_args=list(args),
                    func_kwargs=kwargs,
                )
            ),
        )
    except Exception:
        pass


def _record_output(span, to_wrap, value):
    output_formatter = lambda x: json_dumps(x)
    try:
        span.set_attribute("lmnr.span.output", output_formatter(value))
    except Exception:
        pass


@_with_wrapper
def _wrap_sync(to_wrap, wrapped, instance, args, kwargs):
    with Laminar.start_as_current_span(
        _span_name(to_wrap),
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        _record_input(span, wrapped, args, kwargs)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise

        _record_output(span, to_wrap, result)
        return result


@_with_wrapper
async def _wrap_async(to_wrap, wrapped, instance, args, kwargs):
    with Laminar.start_as_current_span(
        _span_name(to_wrap),
        span_type=to_wrap.get("span_type", "DEFAULT"),
    ) as span:
        _record_input(span, wrapped, args, kwargs)

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise

        _record_output(span, to_wrap, result)
        return result


@_with_wrapper
def _wrap_async_gen(to_wrap, wrapped, instance, args, kwargs):
    async def generator():
        span = Laminar.start_span(
            _span_name(to_wrap),
            span_type=to_wrap.get("span_type", "DEFAULT"),
        )
        collected = []
        async_iter = None

        try:
            with Laminar.use_span(span):
                _record_input(span, wrapped, args, kwargs)
                async_source = wrapped(*args, **kwargs)
                async_iter = (
                    async_source.__aiter__() if hasattr(async_source, "__aiter__") else async_source
                )

            while True:
                try:
                    with Laminar.use_span(span):
                        item = await async_iter.__anext__()
                        collected.append(item)
                except StopAsyncIteration:
                    break
                yield item
        except Exception as e:  # pylint: disable=broad-except
            with Laminar.use_span(span):
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
            raise
        finally:
            if async_iter and hasattr(async_iter, "aclose"):
                try:
                    with Laminar.use_span(span):
                        await async_iter.aclose()
                except Exception:  # pylint: disable=broad-except
                    pass
            with Laminar.use_span(span):
                _record_output(span, to_wrap, collected)
                span.end()

    return generator()

class ClaudeAgentInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            is_streaming = wrapped_method.get("is_streaming", False)
            is_async = wrapped_method.get("is_async", False)

            if is_streaming:
                wrapper_factory = _wrap_async_gen
            elif is_async:
                wrapper_factory = _wrap_async
            else:
                wrapper_factory = _wrap_sync

            # For class methods: "Class.method", for module functions: just "function_name"
            if wrap_object:
                target = f"{wrap_object}.{wrap_method}"
            else:
                target = wrap_method

            try:
                wrap_function_wrapper(
                    wrap_package,
                    target,
                    wrapper_factory(wrapped_method),
                )
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            
            # For class methods: "package.Class", for module functions: just "package"
            if wrap_object:
                module_path = f"{wrap_package}.{wrap_object}"
            else:
                module_path = wrap_package

            try:
                unwrap(module_path, wrap_method)
            except (ModuleNotFoundError, AttributeError):
                pass  # that's ok, we don't want to fail if some methods do not exist

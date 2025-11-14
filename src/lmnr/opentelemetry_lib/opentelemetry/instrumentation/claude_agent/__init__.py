import importlib
import sys

from lmnr import Laminar
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.sdk.utils import get_input_from_func_args, is_method
from opentelemetry.trace import Status, StatusCode
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from typing import Any, Collection
from wrapt import FunctionWrapper, wrap_function_wrapper

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
        # No "object" and "class_name" fields for module-level functions
        "package": "claude_agent_sdk",
        "method": "query",
        "is_async": True,
        "is_streaming": True,
    },
    {
        # No "object" and "class_name" fields for module-level functions
        "package": "claude_agent_sdk",
        "method": "create_sdk_mcp_server",
        "is_async": False,
        "is_streaming": False,
    },
]

_MODULE_FUNCTION_ORIGINALS: dict[tuple[str, str], Any] = {}

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


def _replace_function_aliases(original, wrapped):
    for module in list(sys.modules.values()):
        module_dict = getattr(module, "__dict__", None)
        if not module_dict:
            continue
        for attr, value in list(module_dict.items()):
            if value is original:
                setattr(module, attr, wrapped)


def _wrap_module_function(module_name: str, function_name: str, wrapper):
    try:
        module = sys.modules.get(module_name) or importlib.import_module(module_name)
    except ModuleNotFoundError:
        return

    try:
        original = getattr(module, function_name)
    except AttributeError:
        return

    key = (module_name, function_name)
    if key not in _MODULE_FUNCTION_ORIGINALS:
        _MODULE_FUNCTION_ORIGINALS[key] = original

    wrapped_function = FunctionWrapper(original, wrapper)
    setattr(module, function_name, wrapped_function)
    _replace_function_aliases(original, wrapped_function)


def _unwrap_module_function(module_name: str, function_name: str):
    key = (module_name, function_name)
    original = _MODULE_FUNCTION_ORIGINALS.get(key)
    if not original:
        return

    module = sys.modules.get(module_name)
    if not module:
        return

    current = getattr(module, function_name, None)
    setattr(module, function_name, original)
    if current is not None:
        _replace_function_aliases(current, original)
    del _MODULE_FUNCTION_ORIGINALS[key]

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
                    with Laminar.use_span(
                        span, record_exception=False, set_status_on_exception=False
                    ):
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

            wrapper = wrapper_factory(wrapped_method)

            if wrap_object:
                target = f"{wrap_object}.{wrap_method}"
                try:
                    wrap_function_wrapper(
                        wrap_package,
                        target,
                        wrapper,
                    )
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok, we don't want to fail if some methods do not exist
            else:
                try:
                    _wrap_module_function(
                        wrap_package,
                        wrap_method,
                        wrapper,
                    )
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            
            if wrap_object:
                module_path = f"{wrap_package}.{wrap_object}"
                try:
                    unwrap(module_path, wrap_method)
                except (ModuleNotFoundError, AttributeError):
                    pass  # that's ok, we don't want to fail if some methods do not exist
            else:
                _unwrap_module_function(wrap_package, wrap_method)

import importlib
import os
import sys

from lmnr import Laminar
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.tracing import get_current_context
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_IDS_PATH, SPAN_PATH
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import get_input_from_func_args, is_method

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import Status, StatusCode
from typing import Any, Collection
from typing_extensions import TypedDict
from wrapt import FunctionWrapper, wrap_function_wrapper

from .proxy import start_proxy, release_proxy, get_cc_proxy_base_url, set_trace_to_proxy

logger = get_default_logger(__name__)


class SpanContextPayload(TypedDict):
    trace_id: str
    span_id: str
    project_api_key: str
    span_ids_path: list[str]
    span_path: list[str]
    laminar_url: str


_instruments = ("claude-agent-sdk >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "connect",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
        # start proxy on connection
        "is_start_proxy": True,
        "is_publish_span_context": True,  # TODO: is there a need to publish span context here?
        "is_release_proxy": False,
    },
    {
        "package": "claude_agent_sdk.client",
        "object": "ClaudeSDKClient",
        "method": "query",
        "class_name": "ClaudeSDKClient",
        "is_async": True,
        "is_streaming": False,
        # only publish span context here as start/close managed by connect/disconnect
        "is_start_proxy": False,
        "is_publish_span_context": True,
        "is_release_proxy": False,
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
        # close proxy on a connection drop
        "is_start_proxy": False,
        "is_publish_span_context": False,
        "is_release_proxy": True,
    },
    {
        # No "object" and "class_name" fields for module-level functions
        "package": "claude_agent_sdk",
        "method": "query",
        "is_async": True,
        "is_streaming": True,
        # start, send span to, and release proxy here as it is a module-level function doing all on its own
        "is_start_proxy": True,
        "is_publish_span_context": True,
        "is_release_proxy": True,
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
    try:
        span.set_attribute("lmnr.span.output", json_dumps(value))
    except Exception:
        pass


def _get_span_context_payload() -> dict[str, str] | None:
    current_span: ReadableSpan = trace.get_current_span(context=get_current_context())
    if current_span is trace.INVALID_SPAN:
        return None

    span_context = current_span.get_span_context()
    if span_context is None or not span_context.is_valid:
        return None

    span_ids_path = []
    span_path = []
    if hasattr(current_span, "attributes"):
        readable_span: ReadableSpan = current_span
        span_ids_path = list(readable_span.attributes.get(SPAN_IDS_PATH, tuple()))
        span_path = list(readable_span.attributes.get(SPAN_PATH, tuple()))

    project_api_key = Laminar.get_project_api_key()
    laminar_url = Laminar.get_base_http_url()

    return {
        "trace_id": f"{span_context.trace_id:032x}",
        "span_id": f"{span_context.span_id:016x}",
        "project_api_key": project_api_key or "",
        "span_ids_path": span_ids_path,
        "span_path": span_path,
        "laminar_url": laminar_url,
    }


def _publish_span_context() -> None:
    payload = _get_span_context_payload()
    if not payload:
        return

    set_trace_to_proxy(
        trace_id=payload["trace_id"],
        span_id=payload["span_id"],
        project_api_key=payload["project_api_key"],
        span_ids_path=payload["span_ids_path"],
        span_path=payload["span_path"],
        laminar_url=payload["laminar_url"],
    )


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

        original_base_url = None
        if to_wrap.get("is_start_proxy"):
            original_base_url = os.environ.get("ANTHROPIC_BASE_URL")
            start_proxy()

        if to_wrap.get("is_publish_span_context"):
            proxy_base_url = get_cc_proxy_base_url()
            if proxy_base_url:
                _publish_span_context()
            else:
                logger.debug(
                    "No claude proxy server found. Skipping span context publication."
                )

        try:
            result = await wrapped(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            if original_base_url is not None:
                if original_base_url:
                    os.environ["ANTHROPIC_BASE_URL"] = original_base_url
                else:
                    os.environ.pop("ANTHROPIC_BASE_URL", None)
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(e)
            raise
        finally:
            if to_wrap.get("is_release_proxy"):
                release_proxy()

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

        original_base_url = None
        if to_wrap.get("is_start_proxy"):
            original_base_url = os.environ.get("ANTHROPIC_BASE_URL")
            start_proxy()

        if to_wrap.get("is_publish_span_context"):
            with Laminar.use_span(span):
                proxy_base_url = get_cc_proxy_base_url()
                if proxy_base_url:
                    _publish_span_context()
                else:
                    logger.debug(
                        "No claude proxy server found. Skipping span context publication."
                    )

        try:
            with Laminar.use_span(span):
                _record_input(span, wrapped, args, kwargs)
                async_source = wrapped(*args, **kwargs)
                async_iter = (
                    async_source.__aiter__()
                    if hasattr(async_source, "__aiter__")
                    else async_source
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
            if original_base_url is not None:
                if original_base_url:
                    os.environ["ANTHROPIC_BASE_URL"] = original_base_url
                else:
                    os.environ.pop("ANTHROPIC_BASE_URL", None)
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

            if to_wrap.get("is_release_proxy"):
                release_proxy()

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

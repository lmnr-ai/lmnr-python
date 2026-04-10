import logging
import time
from enum import Enum

from opentelemetry import context as context_api, trace
from opentelemetry.context import Context
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry._logs import LogRecord, Logger, get_logger
from opentelemetry._logs.severity import SeverityNumber

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    set_span_attribute,
    dont_throw,
)
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from lmnr.sdk.utils import json_dumps
from .version import __version__

log = logging.getLogger(__name__)


MODAL_LOG_ATTRIBUTES = {
    "modal.system": "modal",
}


class LogStream(Enum):
    """Enum for Modal log stream types."""
    STDOUT = "stdout"
    STDERR = "stderr"


def _emit_log(
    logger: Logger,
    stream: LogStream,
    content: str,
    ctx: Context,
    extra_attributes: dict[str, str] | None = None,
):
    """Emit a log event using the OpenTelemetry Logs API."""
    if not content:
        return

    try:
        event_name = f"modal.log.{stream.value}"
        severity_number = (
            SeverityNumber.INFO if stream == LogStream.STDOUT
            else SeverityNumber.ERROR
        )

        attributes = {
            **MODAL_LOG_ATTRIBUTES,
            "modal.log.stream": stream.value,
        }
        if extra_attributes:
            attributes.update(extra_attributes)

        logger.emit(
            LogRecord(
                timestamp=time.time_ns(),
                context=ctx,
                body=content,
                severity_number=severity_number,
                attributes=attributes,
                event_name=event_name,
            )
        )
    except Exception as e:
        log.debug(f"Failed to emit Modal log event: {e}")


class _StreamReaderLogProxy:
    """Proxy for Modal's StreamReader that emits OTel logs as lines are consumed.

    Wraps the original iterator so that each line yielded is also emitted as an
    OpenTelemetry log record. This preserves the lazy/streaming nature of the
    original reader while capturing logs transparently.
    """

    def __init__(
        self,
        original,
        logger: Logger,
        stream: LogStream,
        ctx: Context,
        extra_attributes: dict[str, str] | None = None,
    ):
        self._original = original
        self._logger = logger
        self._stream = stream
        self._ctx = ctx
        self._extra_attributes = extra_attributes

    def __iter__(self):
        for line in self._original:
            _emit_log(
                self._logger, self._stream, line, self._ctx,
                self._extra_attributes,
            )
            yield line

    async def __aiter__(self):
        async for line in self._original:
            _emit_log(
                self._logger, self._stream, line, self._ctx,
                self._extra_attributes,
            )
            yield line

    def __getattr__(self, name):
        return getattr(self._original, name)


class _InstrumentedContainerProcess:
    """Proxy for Modal's ContainerProcess that intercepts stdout/stderr for logging.

    Wraps stdout and stderr with _StreamReaderLogProxy so that logs are emitted
    as the user iterates over the process output. All other attributes are
    delegated to the original process object.
    """

    def __init__(
        self,
        original,
        logger: Logger,
        ctx: Context,
        extra_attributes: dict[str, str] | None = None,
    ):
        self._original = original
        self._stdout_proxy = _StreamReaderLogProxy(
            original.stdout, logger, LogStream.STDOUT, ctx, extra_attributes,
        )
        self._stderr_proxy = _StreamReaderLogProxy(
            original.stderr, logger, LogStream.STDERR, ctx, extra_attributes,
        )

    @property
    def stdout(self):
        return self._stdout_proxy

    @property
    def stderr(self):
        return self._stderr_proxy

    def __getattr__(self, name):
        return getattr(self._original, name)


@dont_throw
def _set_create_request_attributes(span: Span, args: tuple, kwargs: dict):
    """Set span attributes from Sandbox.create() arguments."""
    # Positional args are the entrypoint command parts
    if args:
        command = " ".join(str(a) for a in args)
        set_span_attribute(span, "modal.command", command)

    input_data: dict = {}
    if args:
        input_data["entrypoint"] = list(args)

    timeout = kwargs.get("timeout")
    if timeout is not None:
        set_span_attribute(span, "modal.timeout", timeout)
        input_data["timeout"] = timeout

    image = kwargs.get("image")
    if image is not None:
        set_span_attribute(span, "modal.image", str(image))
        input_data["image"] = str(image)

    gpu = kwargs.get("gpu")
    if gpu is not None:
        set_span_attribute(span, "modal.gpu", str(gpu))
        input_data["gpu"] = str(gpu)

    cpu = kwargs.get("cpu")
    if cpu is not None:
        set_span_attribute(span, "modal.cpu", str(cpu))
        input_data["cpu"] = str(cpu)

    memory = kwargs.get("memory")
    if memory is not None:
        set_span_attribute(span, "modal.memory", str(memory))
        input_data["memory"] = str(memory)

    set_span_attribute(span, SPAN_INPUT, json_dumps(input_data))


@dont_throw
def _set_create_response_attributes(span: Span, response):
    """Set span attributes from Sandbox.create() response."""
    sandbox_id = getattr(response, "object_id", None)
    if sandbox_id:
        set_span_attribute(span, "modal.sandbox_id", sandbox_id)
    set_span_attribute(span, SPAN_OUTPUT, json_dumps({"sandbox_id": sandbox_id}))


@dont_throw
def _set_exec_request_attributes(span: Span, args: tuple, kwargs: dict):
    """Set span attributes from Sandbox.exec() arguments."""
    # args are the command parts: sb.exec("bash", "-c", "echo hello")
    if args:
        command = " ".join(str(a) for a in args)
        set_span_attribute(span, "modal.command", command)

    input_data: dict = {}
    if args:
        input_data["command"] = list(args)

    workdir = kwargs.get("workdir")
    if workdir is not None:
        set_span_attribute(span, "modal.workdir", workdir)
        input_data["workdir"] = workdir

    timeout = kwargs.get("timeout")
    if timeout is not None:
        set_span_attribute(span, "modal.timeout", timeout)
        input_data["timeout"] = timeout

    set_span_attribute(span, SPAN_INPUT, json_dumps(input_data))


@dont_throw
def _set_exec_response_attributes(span: Span, response):
    """Set span attributes from the exec response (ContainerProcess)."""
    returncode = getattr(response, "returncode", None)
    if returncode is not None:
        set_span_attribute(span, "modal.exit_code", returncode)
    set_span_attribute(
        span, SPAN_OUTPUT, json_dumps({"returncode": returncode})
    )


def _wrap_create(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for Sandbox.create().

    Creates a span around sandbox creation.
    """
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = Laminar.start_active_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
    )

    if span.is_recording():
        _set_create_request_attributes(span, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)

        if span.is_recording():
            _set_create_response_attributes(span, response)

        span.end()

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    return response


async def _awrap_create(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for async Sandbox.create()."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = Laminar.start_active_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
    )

    if span.is_recording():
        _set_create_request_attributes(span, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)

        if span.is_recording():
            _set_create_response_attributes(span, response)

        span.end()

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    return response


def _wrap_exec(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for Sandbox.exec().

    Creates a span, executes the command, and wraps the returned
    ContainerProcess to capture stdout/stderr as OTel logs.
    The span is left open and ended when process.wait() is called,
    so that logs are associated with the correct span.
    """
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    otel_logger: Logger | None = get_logger(__name__, __version__)

    # Use start_span (not start_active_span) so the exec span does NOT
    # become the active context span.  This prevents unrelated spans
    # created between exec() and wait() from being mis-parented under
    # the exec span.
    span = Laminar.start_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
    )

    if span.is_recording():
        _set_exec_request_attributes(span, args, kwargs)

    # Build a context that carries the exec span so that log records
    # emitted by the stream proxies are associated with this span.
    ctx = trace.set_span_in_context(span, get_current_context())

    try:
        process = wrapped(*args, **kwargs)

        sandbox_id = getattr(instance, "object_id", None)
        extra_attributes: dict[str, str] = {}
        if sandbox_id:
            extra_attributes["modal.sandbox_id"] = sandbox_id
            set_span_attribute(span, "modal.sandbox_id", sandbox_id)
        if args:
            extra_attributes["modal.command"] = " ".join(str(a) for a in args)

        if otel_logger is not None:
            instrumented = _InstrumentedContainerProcess(
                process, otel_logger, ctx, extra_attributes,
            )
            # Wrap wait() to end the span with exit code
            original_wait = process.wait

            def instrumented_wait(*a, **kw):
                result = original_wait(*a, **kw)
                try:
                    if span.is_recording():
                        _set_exec_response_attributes(span, process)
                    span.end()
                except Exception:
                    pass
                return result

            instrumented.wait = instrumented_wait
            return instrumented

        span.end()
        return process

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


async def _awrap_exec(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for async Sandbox.exec()."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    otel_logger: Logger | None = get_logger(__name__, __version__)

    # Use start_span (not start_active_span) — see _wrap_exec for rationale.
    span = Laminar.start_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
    )

    if span.is_recording():
        _set_exec_request_attributes(span, args, kwargs)

    ctx = trace.set_span_in_context(span, get_current_context())

    try:
        process = await wrapped(*args, **kwargs)

        sandbox_id = getattr(instance, "object_id", None)
        extra_attributes: dict[str, str] = {}
        if sandbox_id:
            extra_attributes["modal.sandbox_id"] = sandbox_id
            set_span_attribute(span, "modal.sandbox_id", sandbox_id)
        if args:
            extra_attributes["modal.command"] = " ".join(str(a) for a in args)

        if otel_logger is not None:
            instrumented = _InstrumentedContainerProcess(
                process, otel_logger, ctx, extra_attributes,
            )
            original_wait = process.wait

            async def instrumented_wait(*a, **kw):
                result = await original_wait(*a, **kw)
                try:
                    if span.is_recording():
                        _set_exec_response_attributes(span, process)
                    span.end()
                except Exception:
                    pass
                return result

            instrumented.wait = instrumented_wait
            return instrumented

        span.end()
        return process

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

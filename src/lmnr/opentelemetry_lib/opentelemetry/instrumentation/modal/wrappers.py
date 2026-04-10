"""Wrapper functions for Modal Sandbox instrumentation.

Instruments:
- Sandbox.create: Creates a span tracking sandbox creation
- Sandbox.exec: Creates a span tracking command execution, emits stdout/stderr logs

For exec, the returned ContainerProcess's stdout/stderr iterators are wrapped
to emit OTel logs as the user reads them, without consuming the data.
"""

import logging
import time
from enum import Enum

from opentelemetry import context as context_api
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
    STDOUT = "stdout"
    STDERR = "stderr"


# --- Attribute setters ---

@dont_throw
def _set_create_request_attributes(span: Span, args: tuple, kwargs: dict):
    cmd = list(args) if args else []
    name = kwargs.get("name")
    timeout = kwargs.get("timeout")
    gpu = kwargs.get("gpu")
    workdir = kwargs.get("workdir")
    app = kwargs.get("app")
    image = kwargs.get("image")

    input_data: dict = {}
    if cmd:
        set_span_attribute(span, "modal.sandbox.cmd", json_dumps(cmd))
        input_data["cmd"] = cmd
    if name:
        set_span_attribute(span, "modal.sandbox.name", name)
        input_data["name"] = name
    if timeout is not None:
        set_span_attribute(span, "modal.sandbox.timeout", timeout)
        input_data["timeout"] = timeout
    if gpu:
        set_span_attribute(span, "modal.sandbox.gpu", str(gpu))
        input_data["gpu"] = str(gpu)
    if workdir:
        set_span_attribute(span, "modal.sandbox.workdir", workdir)
        input_data["workdir"] = workdir
    if image:
        input_data["image"] = str(image)
    if app:
        input_data["app"] = str(app)

    set_span_attribute(span, SPAN_INPUT, json_dumps(input_data))


@dont_throw
def _set_create_response_attributes(span: Span, sandbox):
    sandbox_id = getattr(sandbox, "object_id", None)
    if sandbox_id:
        set_span_attribute(span, "modal.sandbox.id", sandbox_id)
    set_span_attribute(span, SPAN_OUTPUT, json_dumps({"sandbox_id": sandbox_id}))


@dont_throw
def _set_exec_request_attributes(span: Span, command: str, kwargs: dict):
    workdir = kwargs.get("workdir")

    set_span_attribute(span, "modal.command", command)
    input_data: dict = {"command": command}
    if workdir:
        set_span_attribute(span, "modal.workdir", workdir)
        input_data["workdir"] = workdir
    set_span_attribute(span, SPAN_INPUT, json_dumps(input_data))


# --- Log emission ---

def _emit_log(
    logger: Logger,
    stream: LogStream,
    content: str,
    ctx: Context,
    extra_attributes: dict[str, str] | None = None,
):
    if not content:
        return

    try:
        event_name = f"modal.log.{stream.value}"
        severity = SeverityNumber.INFO if stream == LogStream.STDOUT else SeverityNumber.ERROR

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
                severity_number=severity,
                attributes=attributes,
                event_name=event_name,
            )
        )
    except Exception as e:
        log.debug(f"Failed to emit Modal log event: {e}")


# --- Stream wrapper for tee-ing output ---

class _TeeStreamIterator:
    """Wraps a Modal StreamReader to emit OTel logs as the user reads output.

    Both iteration (for line in stream) and read() are intercepted to emit
    OTel log records. Other methods are delegated to the original stream.
    """

    def __init__(
        self,
        original_stream,
        logger: Logger,
        stream_type: LogStream,
        ctx: Context,
        extra_attributes: dict[str, str] | None = None,
    ):
        self._original = original_stream
        self._logger = logger
        self._stream_type = stream_type
        self._ctx = ctx
        self._extra_attributes = extra_attributes

    def __getattr__(self, name):
        return getattr(self._original, name)

    def read(self):
        content = self._original.read()
        try:
            _emit_log(
                self._logger, self._stream_type, content, self._ctx,
                self._extra_attributes,
            )
        except Exception:
            pass
        return content

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._original)
        try:
            _emit_log(
                self._logger, self._stream_type, line, self._ctx,
                self._extra_attributes,
            )
        except Exception:
            pass
        return line

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self._original.__anext__()
        try:
            _emit_log(
                self._logger, self._stream_type, line, self._ctx,
                self._extra_attributes,
            )
        except Exception:
            pass
        return line


def _wrap_process_streams(process, logger: Logger, command: str, ctx: Context):
    """Wrap a ContainerProcess's stdout/stderr to emit logs on iteration."""
    extra_attributes = {"modal.command": command}

    original_stdout = process.stdout
    original_stderr = process.stderr

    stdout_wrapper = _TeeStreamIterator(
        original_stdout, logger, LogStream.STDOUT, ctx, extra_attributes,
    )
    stderr_wrapper = _TeeStreamIterator(
        original_stderr, logger, LogStream.STDERR, ctx, extra_attributes,
    )

    # Replace the stdout/stderr properties with our wrappers.
    # ContainerProcess.stdout/stderr are properties, so we patch at the instance level
    # by overriding __class__ or using object.__setattr__.
    # Since these are properties on the class, we need to set them on the instance's __dict__
    # or use a wrapper object.
    try:
        process._lmnr_original_stdout = original_stdout
        process._lmnr_original_stderr = original_stderr
        # Override the property access by patching at instance level
        # Properties don't support instance-level override, so we wrap the class
        original_class = process.__class__

        class InstrumentedContainerProcess(original_class):
            @property
            def stdout(self):
                return self._lmnr_stdout_wrapper

            @property
            def stderr(self):
                return self._lmnr_stderr_wrapper

        process._lmnr_stdout_wrapper = stdout_wrapper
        process._lmnr_stderr_wrapper = stderr_wrapper
        process.__class__ = InstrumentedContainerProcess
    except Exception as e:
        log.debug(f"Failed to wrap Modal process streams: {e}")

    return process


# --- Wrappers ---

def _wrap_create(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = Laminar.start_active_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
        user_id=(kwargs.get("metadata") or {}).get("user_id"),
        session_id=(kwargs.get("metadata") or {}).get("session_id"),
        tags=(kwargs.get("metadata") or {}).get("tags", []),
        metadata=(kwargs.get("metadata") or {}),
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


def _wrap_exec(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    logger: Logger | None = get_logger(__name__, __version__)

    span = Laminar.start_active_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
        user_id=(kwargs.get("metadata") or {}).get("user_id"),
        session_id=(kwargs.get("metadata") or {}).get("session_id"),
        tags=(kwargs.get("metadata") or {}).get("tags", []),
        metadata=(kwargs.get("metadata") or {}),
    )

    command = " ".join(str(a) for a in args) if args else ""

    if span.is_recording():
        _set_exec_request_attributes(span, command, kwargs)

    ctx = get_current_context()

    try:
        response = wrapped(*args, **kwargs)
        span.end()
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise

    # Wrap stdout/stderr iterators to emit logs as user reads them
    try:
        if logger is not None:
            response = _wrap_process_streams(response, logger, command, ctx)
    except Exception as log_error:
        log.debug(f"Failed to wrap Modal process streams: {log_error}")

    return response



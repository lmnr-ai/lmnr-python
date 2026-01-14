"""OpenTelemetry Daytona SDK instrumentation

This module instruments the Daytona SDK to capture traces and logs for
execute_session_command calls.

The instrumentation handles both synchronous and asynchronous commands:

Synchronous commands (run_async/var_async=False):
1. Create a span when execute_session_command is called
2. Execute the command and wait for completion
3. End the span after the command returns
4. Emit logs immediately from response.stdout/stderr

Asynchronous commands (run_async/var_async=True):
1. Create a span when execute_session_command is called
2. Execute the command (returns immediately with cmd_id)
3. End the span after execute_session_command returns
4. Start background log streaming to capture stdout/stderr as they arrive
5. Emit OpenTelemetry logs for each line using the Logs API

Note: The module path for the Daytona SDK may need adjustment based on the
actual package structure. Update WRAPPED_METHODS if the module path differs.
"""

import asyncio
import logging
from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry import context as context_api
from opentelemetry.trace import get_tracer, SpanKind, Status, StatusCode, Span, Tracer
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY, unwrap
from opentelemetry._events import Event, EventLogger, get_event_logger

from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)

from .config import Config
from .utils import (
    dont_throw,
    set_span_attribute,
    with_tracer_wrapper,
    async_with_tracer_wrapper,
)

logger = logging.getLogger(__name__)

_instruments = ("daytona >= 0.1.0",)

# Note: Update the package path based on the actual Daytona SDK structure.
# Common patterns:
#   - "daytona._process" for internal modules
#   - "daytona.process" for public modules
#   - "daytona.sandbox.process" for nested modules
WRAPPED_METHODS = [
    {
        "package": "daytona._async.process",
        "object": "AsyncProcess",
        "method": "execute_session_command",
        "span_name": "daytona.sandbox.process.execute_session_command",
        "is_async": True,
    },
     {
        "package": "daytona._sync.process",
        "object": "Process",
        "method": "execute_session_command",
        "span_name": "daytona.sandbox.process.execute_session_command",
        "is_async": False,
    },
]

# Event attributes for Daytona logs
DAYTONA_LOG_ATTRIBUTES = {
    "daytona.system": "daytona",
}


@dont_throw
def _set_request_attributes(span: Span, session_id: str, request):
    """Set span attributes from the execute_session_command request."""
    set_span_attribute(span, "daytona.session_id", session_id)

    # Extract attributes from the request object
    if hasattr(request, "command"):
        set_span_attribute(span, "daytona.command", request.command)
    if getattr(request, "run_async", False) or getattr(request, "var_async", False):
        set_span_attribute(span, "daytona.async", True)
    else:
        set_span_attribute(span, "daytona.async", False)


@dont_throw
def _set_response_attributes(span: Span, response):
    """Set span attributes from the execute_session_command response."""
    if hasattr(response, "cmd_id"):
        set_span_attribute(span, "daytona.cmd_id", response.cmd_id)
    if hasattr(response, "exit_code"):
        set_span_attribute(span, "daytona.exit_code", response.exit_code)
    if hasattr(response, "output"):
        set_span_attribute(span, "daytona.output", response.output)


def _emit_log_event(
    event_logger: EventLogger | None,
    stream: str,
    content: str,
    session_id: str,
    cmd_id: str,
):
    """Emit a log event using the OpenTelemetry Logs API.

    This emits a proper OTel log record that is independent of any span,
    allowing logs to be captured even after the command span has ended.
    """
    if event_logger is None or not content:
        return

    try:
        event_name = f"daytona.log.{stream}"
        event_body = {
            "content": content,
            "stream": stream,
            "session_id": session_id,
            "cmd_id": cmd_id,
        }

        event_logger.emit(
            Event(
                name=event_name,
                body=event_body,
                attributes={
                    **DAYTONA_LOG_ATTRIBUTES,
                    "daytona.session_id": session_id,
                    "daytona.cmd_id": cmd_id,
                    "daytona.log.stream": stream,
                },
            )
        )
    except Exception as e:
        logger.debug(f"Failed to emit Daytona log event: {e}")


def _emit_logs_from_response(
    event_logger: EventLogger | None,
    response,
    session_id: str,
    cmd_id: str,
):
    """Emit logs from a synchronous command response.

    For synchronous commands, the stdout and stderr are already in the response,
    so we emit them immediately as log events.
    """
    if event_logger is None:
        return

    # Emit stdout logs
    if hasattr(response, "stdout") and response.stdout:
        _emit_log_event(event_logger, "stdout", response.stdout, session_id, cmd_id)

    # Emit stderr logs
    if hasattr(response, "stderr") and response.stderr:
        _emit_log_event(event_logger, "stderr", response.stderr, session_id, cmd_id)


def _create_log_callbacks(
    event_logger: EventLogger | None,
    session_id: str,
    cmd_id: str,
):
    """Create stdout/stderr callbacks that emit OTel log events.

    These callbacks will emit OpenTelemetry log records for each log line
    received from the Daytona sandbox, independent of any span.
    """

    def on_stdout(content: str):
        """Callback for stdout log lines."""
        _emit_log_event(event_logger, "stdout", content, session_id, cmd_id)

    def on_stderr(content: str):
        """Callback for stderr log lines."""
        _emit_log_event(event_logger, "stderr", content, session_id, cmd_id)

    return on_stdout, on_stderr


async def _stream_logs_async(
    event_logger: EventLogger | None,
    process,
    session_id: str,
    cmd_id: str,
):
    """Stream logs asynchronously and emit them as OTel log events.

    This function starts the log streaming in the background and emits
    OpenTelemetry log records for each stdout/stderr log line.
    The logs are emitted independently of any span.
    """
    on_stdout, on_stderr = _create_log_callbacks(event_logger, session_id, cmd_id)

    try:
        await process.get_session_command_logs_async(
            session_id,
            cmd_id,
            on_stdout,
            on_stderr,
        )
    except Exception as e:
        logger.debug(f"Failed to stream Daytona logs: {e}")


def _start_log_streaming(
    event_logger: EventLogger | None,
    instance,
    session_id: str,
    cmd_id: str,
):
    """Start log streaming in the background.

    This function creates an asyncio task to stream logs from the Daytona
    sandbox. The logs are emitted as OpenTelemetry log records.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop - try to get the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop at all, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

    # Create the log streaming task
    try:
        # If we're in a running loop, create a task
        if loop.is_running():
            asyncio.create_task(
                _stream_logs_async(event_logger, instance, session_id, cmd_id)
            )
        else:
            # If no loop is running, run the coroutine in a new thread
            import threading

            def run_in_thread():
                asyncio.run(
                    _stream_logs_async(event_logger, instance, session_id, cmd_id)
                )

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
    except Exception as e:
        logger.debug(f"Failed to start Daytona log streaming task: {e}")


@with_tracer_wrapper
def _wrap(
    tracer: Tracer,
    event_logger: EventLogger | None,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for sync execute_session_command.

    This wrapper:
    1. Creates a span for the command execution
    2. Sets request attributes
    3. Executes the command
    4. Sets response attributes
    5. Ends the span
    6. Emits logs:
       - For async commands (run_async/var_async=True): starts background log streaming
       - For sync commands: emits logs immediately from response.stdout/stderr
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            "lmnr.span.type": "TOOL",
        },
        context=get_current_context(),
    )

    # Extract session_id and request from args/kwargs
    # execute_session_command(session_id, request)
    session_id = args[0] if len(args) > 0 else kwargs.get("session_id")
    request = args[1] if len(args) > 1 else kwargs.get("request")

    if span.is_recording():
        _set_request_attributes(span, session_id, request)

    try:
        response = wrapped(*args, **kwargs)

        if span.is_recording():
            _set_response_attributes(span, response)

        # End the span immediately - logs will be emitted separately
        span.end()

        cmd_id = getattr(response, "cmd_id", None)
        if cmd_id and session_id:
            # Check if this is an async command
            is_async_command = request is not None and (
                getattr(request, "run_async", False) or getattr(request, "var_async", False)
            )

            if is_async_command:
                # For async commands, start background log streaming
                _start_log_streaming(event_logger, instance, session_id, cmd_id)
            else:
                # For sync commands, emit logs immediately from response
                _emit_logs_from_response(event_logger, response, session_id, cmd_id)

        return response

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


@async_with_tracer_wrapper
async def _awrap(
    tracer: Tracer,
    event_logger: EventLogger | None,
    to_wrap,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for async execute_session_command.

    This wrapper:
    1. Creates a span for the command execution
    2. Sets request attributes
    3. Executes the command
    4. Sets response attributes
    5. Ends the span
    6. Emits logs:
       - For async commands (run_async/var_async=True): starts background log streaming
       - For sync commands: emits logs immediately from response.stdout/stderr
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = tracer.start_span(
        to_wrap.get("span_name"),
        kind=SpanKind.CLIENT,
        attributes={
            "lmnr.span.type": "TOOL",
        },
        context=get_current_context(),
    )

    # Extract session_id and request from args/kwargs
    # execute_session_command(session_id, request)
    session_id = args[0] if len(args) > 0 else kwargs.get("session_id")
    request = args[1] if len(args) > 1 else kwargs.get("request")

    if span.is_recording():
        _set_request_attributes(span, session_id, request)

    try:
        response = await wrapped(*args, **kwargs)

        if span.is_recording():
            _set_response_attributes(span, response)

        # End the span immediately - logs will be emitted separately
        span.end()

        cmd_id = getattr(response, "cmd_id", None)
        if cmd_id and session_id:
            # Check if this is an async command
            is_async_command = request is not None and (
                getattr(request, "run_async", False) or getattr(request, "var_async", False)
            )

            if is_async_command:
                # For async commands, start background log streaming
                _start_log_streaming(event_logger, instance, session_id, cmd_id)
            else:
                # For sync commands, emit logs immediately from response
                _emit_logs_from_response(event_logger, response, session_id, cmd_id)

        return response

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


class DaytonaSdkInstrumentor(BaseInstrumentor):
    """An instrumentor for Daytona SDK.

    This instrumentor wraps the execute_session_command method to:
    1. Create OpenTelemetry spans for command execution
    2. Stream command logs as OpenTelemetry log records

    The logs are emitted using the OpenTelemetry Logs API, which is separate
    from the tracing API. This allows logs to be captured even after the
    command span has ended.

    Usage:
        from lmnr import Laminar
        from lmnr.opentelemetry_lib.opentelemetry.instrumentation.daytona import (
            DaytonaSdkInstrumentor,
        )

        Laminar.initialize()
        DaytonaSdkInstrumentor().instrument()

    With custom event logger provider:
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._events import EventLoggerProvider

        logger_provider = LoggerProvider()
        event_logger_provider = EventLoggerProvider(logger_provider)

        DaytonaSdkInstrumentor().instrument(
            event_logger_provider=event_logger_provider
        )
    """

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, "0.0.1a1", tracer_provider)

        # Get event logger for emitting logs
        event_logger_provider = kwargs.get("event_logger_provider")
        event_logger = None
        if event_logger_provider:
            event_logger = get_event_logger(
                __name__, "0.0.1a1", event_logger_provider=event_logger_provider
            )

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            target = f"{wrap_object}.{wrap_method}" if wrap_object else wrap_method

            try:
                wrap_function_wrapper(
                    wrap_package,
                    target,
                    (
                        _awrap(tracer, event_logger, wrapped_method)
                        if wrapped_method.get("is_async")
                        else _wrap(tracer, event_logger, wrapped_method)
                    ),
                )
            except ModuleNotFoundError:
                logger.debug(f"Could not instrument {wrap_package}.{target}")

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            module_path = (
                f"{wrap_package}.{wrap_object}" if wrap_object else wrap_package
            )

            try:
                unwrap(module_path, wrap_method)
            except Exception:
                pass

import asyncio
import logging
import threading
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
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from .version import __version__

log = logging.getLogger(__name__)


DAYTONA_LOG_ATTRIBUTES = {
    "daytona.system": "daytona",
}


class LogStream(Enum):
    """Enum for Daytona log stream types."""
    STDOUT = "stdout"
    STDERR = "stderr"


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


def _emit_log(
    logger: Logger,
    stream: LogStream,
    content: str,
    session_id: str,
    cmd_id: str,
    ctx: Context,
):
    """Emit a log event using the OpenTelemetry Logs API.

    This emits a proper OTel log record that is independent of any span,
    allowing logs to be captured even after the command span has ended.

    Args:
        ctx: The OpenTelemetry Context containing the span to associate with this log.
    """
    if not content:
        return

    try:
        event_name = f"daytona.log.{stream.value}"
        event_body = {
            "content": content,
            "stream": stream.value,
            "session_id": session_id,
            "cmd_id": cmd_id,
        }
        event_severity_number = SeverityNumber.INFO if stream == LogStream.STDOUT else SeverityNumber.ERROR

        logger.emit(
            LogRecord(
                timestamp=time.time_ns(),
                context=ctx,
                body=event_body,
                severity_number=event_severity_number,
                attributes={
                    **DAYTONA_LOG_ATTRIBUTES,
                    "daytona.session_id": session_id,
                    "daytona.cmd_id": cmd_id,
                    "daytona.log.stream": stream.value,
                },
                event_name=event_name,
            )
        )
    except Exception as e:
        log.debug(f"Failed to emit Daytona log event: {e}")


def _emit_logs_from_response(
    logger: Logger,
    response,
    session_id: str,
    cmd_id: str,
    ctx: Context,
):
    """Emit logs from a synchronous command response.

    For synchronous commands, the stdout and stderr are already in the response,
    so we emit them immediately as log events.
    """
    # Emit stdout logs
    if hasattr(response, "stdout") and response.stdout:
        _emit_log(logger, LogStream.STDOUT, response.stdout, session_id, cmd_id, ctx)

    # Emit stderr logs
    if hasattr(response, "stderr") and response.stderr:
        _emit_log(logger, LogStream.STDERR, response.stderr, session_id, cmd_id, ctx)


def _create_log_callbacks(
    logger: Logger,
    session_id: str,
    cmd_id: str,
    ctx: Context,
):
    """Create stdout/stderr callbacks that emit OTel log events.

    These callbacks will emit OpenTelemetry log records for each log line
    received from the Daytona sandbox, independent of any span.

    The context is captured at callback creation time (when the command
    span is active) and closed over, so logs emitted later in background
    threads are still correctly associated with the original command span.
    """

    def on_stdout(content: str):
        """Callback for stdout log lines."""
        _emit_log(logger, LogStream.STDOUT, content, session_id, cmd_id, ctx)

    def on_stderr(content: str):
        """Callback for stderr log lines."""
        _emit_log(logger, LogStream.STDERR, content, session_id, cmd_id, ctx)

    return on_stdout, on_stderr


async def _stream_logs_async(
    logger: Logger,
    process,
    session_id: str,
    cmd_id: str,
    ctx: Context,
):
    """Stream logs asynchronously and emit them as OTel logs.

    This function starts the log streaming in the background and emits
    OpenTelemetry log records for each stdout/stderr log line.
    The logs are associated with the span that was active when streaming started.
    """
    on_stdout, on_stderr = _create_log_callbacks(logger, session_id, cmd_id, ctx)

    try:
        await process.get_session_command_logs_async(
            session_id,
            cmd_id,
            on_stdout,
            on_stderr,
        )
    except asyncio.CancelledError:
        log.debug("Daytona log streaming was cancelled")
    except Exception as e:
        log.debug(f"Failed to stream Daytona logs: {e}")


def _start_log_streaming(
    logger: Logger,
    instance,
    session_id: str,
    cmd_id: str,
    ctx: Context,
):
    """Start log streaming in the background.

    This function streams logs from the Daytona sandbox and emits them as
    OpenTelemetry log records.

    The context is captured before calling this function so that 
    logs arriving later are correctly associated with the original command span.

    For async contexts (when there's a running event loop), we use
    asyncio.create_task() to avoid cross-event-loop issues with aiohttp.
    For sync contexts, we spawn a separate thread with its own event loop.
    """

    async def stream_wrapper():
        try:
            await _stream_logs_async(logger, instance, session_id, cmd_id, ctx)
        except Exception as e:
            log.debug(f"Log streaming error: {e}")

    # Try to use existing event loop first (for async contexts)
    # This avoids cross-event-loop issues with aiohttp clients
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - create a task in the existing loop
        loop.create_task(stream_wrapper())
        return
    except RuntimeError:
        # No running event loop - fall back to thread approach (for sync contexts)
        pass

    # Sync context: use thread with its own event loop
    def run_in_thread():
        try:
            asyncio.run(
                _stream_logs_async(logger, instance, session_id, cmd_id, ctx)
            )
        except Exception as e:
            log.debug(f"Log streaming thread error: {e}")

    try:
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
    except Exception as e:
        log.debug(f"Failed to start Daytona log streaming thread: {e}")


def _process_command_response(
    logger: Logger | None,
    instance,
    response,
    session_id: str,
    request,
    ctx: Context,
):
    """Handle response and emit logs or start log streaming.

    For async commands (run_async/var_async=True): starts background log streaming.
    For sync commands: emits logs immediately from response.stdout/stderr.
    """
    cmd_id = getattr(response, "cmd_id", None)
    if not (cmd_id and session_id):
        return

    if logger is None:
        return

    is_async_command = request is not None and (
        getattr(request, "run_async", False) or getattr(request, "var_async", False)
    )

    if is_async_command:
         _start_log_streaming(logger, instance, session_id, cmd_id, ctx)
    else:
        _emit_logs_from_response(logger, response, session_id, cmd_id, ctx)


def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for sync execute_session_command.

    Creates a span, executes the command, sets attributes, ends span, then emits logs.
    """
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

    # Extract session_id and request from args/kwargs
    # execute_session_command(session_id, request)
    session_id = args[0] if len(args) > 0 else kwargs.get("session_id")
    request = args[1] if len(args) > 1 else kwargs.get("request")

    if span.is_recording():
        _set_request_attributes(span, session_id, request)

    # Capture updated context with span for logs to be associated with it
    ctx = get_current_context()

    try:
        response = wrapped(*args, **kwargs)

        if span.is_recording():
            _set_response_attributes(span, response)

        # End the span immediately - logs will be emitted separately
        span.end()

        _process_command_response(
            logger, instance, response, session_id, request, ctx
        )

        return response

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


async def _awrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance,
    args,
    kwargs,
):
    """Wrapper for async execute_session_command.

    Creates a span, executes the command, sets attributes, ends span, then emits logs.
    """
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    logger: Logger | None = get_logger(__name__, __version__)

    span = Laminar.start_active_span(
        name=to_wrap["span_name"],
        span_type=to_wrap["span_type"],
        user_id=(kwargs.get("metadata") or {}).get("user_id"),
        session_id=(kwargs.get("metadata") or {}).get("session_id"),
        tags=(kwargs.get("metadata") or {}).get("tags", []),
        metadata=(kwargs.get("metadata") or {}),
    )

    # Extract session_id and request from args/kwargs
    # execute_session_command(session_id, request)
    session_id = args[0] if len(args) > 0 else kwargs.get("session_id")
    request = args[1] if len(args) > 1 else kwargs.get("request")

    if span.is_recording():
        _set_request_attributes(span, session_id, request)

    # Capture updated context with span for logs to be associated with it
    ctx = get_current_context()

    try:
        response = await wrapped(*args, **kwargs)

        if span.is_recording():
            _set_response_attributes(span, response)

        # End the span immediately - logs will be emitted separately
        span.end()

        _process_command_response(
            logger, instance, response, session_id, request, ctx
        )

        return response

    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise
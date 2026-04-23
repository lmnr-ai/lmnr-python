"""Wrapper functions for Modal Sandbox instrumentation.

Modal uses the ``synchronicity`` library, which exposes every public SDK
method as a ``FunctionWithAio`` or ``MethodWithAio`` object. These objects
hold *two* independent references to the underlying implementation:

* a blocking callable (used when calling ``Sandbox.create(...)``)
* an async callable, exposed as ``.aio`` (used when calling
  ``await Sandbox.create.aio(...)``)

Because the two references are completely independent, instrumenting only
the blocking attribute would silently miss the entire async code path.
The instrumentation therefore patches both the blocking and async
underlying functions on the ``FunctionWithAio`` / ``MethodWithAio``
instances directly (see ``__init__.py``).

This module provides four wrappers:

* ``_wrap_create``       - blocking ``Sandbox.create``
* ``_wrap_create_async`` - ``Sandbox.create.aio``
* ``_wrap_exec``         - blocking ``sandbox.exec``
* ``_wrap_exec_async``   - ``sandbox.exec.aio``

For ``exec``, the returned ``ContainerProcess``'s ``stdout``/``stderr``
readers are replaced with tee-ing wrappers that emit OTel log records
as the user iterates over or reads the streams, without consuming the
data.
"""

import logging
from functools import partial

from opentelemetry import context as context_api
from opentelemetry.context import Context
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry._logs import Logger, get_logger

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.log_emission import (
    LogStream,
    emit_log,
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


SPAN_NAME_CREATE = "modal.sandbox.create"
SPAN_NAME_EXEC = "modal.sandbox.exec"

_emit_log = partial(emit_log, "modal")


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


def _extract_process_id(process) -> str | None:
    """Extract a Modal ContainerProcess's process_id.

    Modal wraps the public ``ContainerProcess`` with ``synchronicity``, whose
    proxy class only exposes explicitly-translated public methods (``stdout``,
    ``stderr``, ``wait``, ``poll``, ``returncode``, ``attach``). The actual
    ``_impl`` / ``_process_id`` live on the underlying async object, which is
    stashed in the wrapper's ``__dict__`` under a synthesized name like
    ``_sync_original_<id>`` and reachable via the ``_sync_synchronizer``
    attribute.

    We try, in order:
    1. A direct ``process_id`` attribute (forward-compat if Modal exposes it).
    2. Walking through the synchronicity proxy to ``_impl._process_id``.
    """
    pid = getattr(process, "process_id", None)
    if pid:
        return pid

    try:
        synchronizer = getattr(process, "_sync_synchronizer", None)
        if synchronizer is not None:
            original_attr = getattr(synchronizer, "_original_attr", None)
            if original_attr:
                original = process.__dict__.get(original_attr)
                if original is not None:
                    impl = getattr(original, "_impl", None) or original
                    pid = getattr(impl, "_process_id", None) or getattr(
                        impl, "process_id", None
                    )
                    if pid:
                        return pid
    except Exception:
        pass

    return None


@dont_throw
def _set_exec_response_attributes(span: Span, process):
    process_id = _extract_process_id(process)

    output_data: dict = {}
    if process_id:
        set_span_attribute(span, "modal.process.id", process_id)
        output_data["process_id"] = process_id

    set_span_attribute(span, SPAN_OUTPUT, json_dumps(output_data))


# --- Span helpers ---


def _start_span(name: str, kwargs: dict):
    metadata = kwargs.get("metadata") or {}
    return Laminar.start_active_span(
        name=name,
        span_type="DEFAULT",
        user_id=metadata.get("user_id"),
        session_id=metadata.get("session_id"),
        tags=metadata.get("tags", []),
        metadata=metadata,
    )


@dont_throw
def _record_exception(span: Span, exc: Exception):
    attributes = get_event_attributes_from_context()
    span.set_attribute(ERROR_TYPE, exc.__class__.__name__)
    span.record_exception(exc, attributes=attributes)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


# --- Stream wrapper for tee-ing output ---


class _ReadProxy:
    """Callable proxy for ``StreamReader.read`` that preserves the ``.aio`` API.

    Modal exposes ``stream.read`` as a ``MethodWithAio`` instance, so user
    code may do either::

        data = stream.read()            # blocking
        data = await stream.read.aio()  # async

    ``__call__`` forwards to the blocking implementation, ``aio`` to the
    async one, and both emit a single OTel log record containing the
    complete output.
    """

    def __init__(self, original_read, tee: "_TeeStreamIterator"):
        self._original_read = original_read
        self._tee = tee

    def __call__(self, *args, **kwargs):
        content = self._original_read(*args, **kwargs)
        self._tee._emit(content)
        return content

    async def aio(self, *args, **kwargs):
        aio_fn = getattr(self._original_read, "aio", None)
        if aio_fn is None:
            # Fallback: some modal versions may expose read() as a coroutine directly
            result = self._original_read(*args, **kwargs)
            if hasattr(result, "__await__"):
                content = await result
            else:
                content = result
        else:
            content = await aio_fn(*args, **kwargs)
        self._tee._emit(content)
        return content


class _TeeStreamIterator:
    """Wraps a Modal StreamReader to emit OTel logs as the user reads output.

    Iteration (both sync ``for`` and ``async for``) as well as ``read()`` and
    ``await read.aio()`` are intercepted to emit OTel log records. Any other
    attribute access is delegated to the underlying stream.
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
        self._sync_iter = None
        self._async_iter = None

    def __getattr__(self, name):
        return getattr(self._original, name)

    def _emit(self, content: str):
        try:
            _emit_log(
                self._logger,
                self._stream_type,
                content,
                self._ctx,
                self._extra_attributes,
            )
        except Exception:
            pass

    @property
    def read(self):
        return _ReadProxy(self._original.read, self)

    def __iter__(self):
        self._sync_iter = iter(self._original)
        return self

    def __next__(self):
        if self._sync_iter is None:
            self._sync_iter = iter(self._original)
        line = next(self._sync_iter)
        self._emit(line)
        return line

    def __aiter__(self):
        self._async_iter = self._original.__aiter__()
        return self

    async def __anext__(self):
        if self._async_iter is None:
            self._async_iter = self._original.__aiter__()
        line = await self._async_iter.__anext__()
        self._emit(line)
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

    # ContainerProcess.stdout / .stderr are class-level properties. We can't
    # simply assign on the instance, so we swap __class__ to a subclass whose
    # property getters return the tee-ing wrappers we stashed on the instance.
    try:
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


def _wrap_create(wrapped, instance, args, kwargs):
    """Blocking ``Sandbox.create``."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = _start_span(SPAN_NAME_CREATE, kwargs)

    if span.is_recording():
        _set_create_request_attributes(span, args, kwargs)

    try:
        response = wrapped(*args, **kwargs)
        if span.is_recording():
            _set_create_response_attributes(span, response)
        span.end()
    except Exception as e:
        _record_exception(span, e)
        span.end()
        raise

    return response


async def _wrap_create_async(wrapped, instance, args, kwargs):
    """``await Sandbox.create.aio(...)``."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = _start_span(SPAN_NAME_CREATE, kwargs)

    if span.is_recording():
        _set_create_request_attributes(span, args, kwargs)

    try:
        response = await wrapped(*args, **kwargs)
        if span.is_recording():
            _set_create_response_attributes(span, response)
        span.end()
    except Exception as e:
        _record_exception(span, e)
        span.end()
        raise

    return response


def _split_exec_args(args: tuple) -> tuple:
    """Strip the leading ``self`` injected by MethodWithAio's partial binding.

    MethodWithAio's ``__get__`` returns ``functools.partial(_func, sandbox)``
    (see ``synchronicity/combined_types.py``), so when we wrap ``_func``
    directly the sandbox instance arrives as ``args[0]``. Every remaining
    positional argument is a command token.
    """
    if not args:
        return (), ()
    return args[0], tuple(args[1:])


def _wrap_exec(wrapped, instance, args, kwargs):
    """Blocking ``sandbox.exec``."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    logger: Logger | None = get_logger(__name__, __version__)

    span = _start_span(SPAN_NAME_EXEC, kwargs)

    _, cmd_tokens = _split_exec_args(args)
    command = " ".join(str(a) for a in cmd_tokens)

    if span.is_recording():
        _set_exec_request_attributes(span, command, kwargs)

    ctx = get_current_context()

    try:
        response = wrapped(*args, **kwargs)
        if span.is_recording():
            _set_exec_response_attributes(span, response)
        span.end()
    except Exception as e:
        _record_exception(span, e)
        span.end()
        raise

    try:
        if logger is not None:
            response = _wrap_process_streams(response, logger, command, ctx)
    except Exception as log_error:
        log.debug(f"Failed to wrap Modal process streams: {log_error}")

    return response


async def _wrap_exec_async(wrapped, instance, args, kwargs):
    """``await sandbox.exec.aio(...)``."""
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = ()

    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    logger: Logger | None = get_logger(__name__, __version__)

    span = _start_span(SPAN_NAME_EXEC, kwargs)

    _, cmd_tokens = _split_exec_args(args)
    command = " ".join(str(a) for a in cmd_tokens)

    if span.is_recording():
        _set_exec_request_attributes(span, command, kwargs)

    ctx = get_current_context()

    try:
        response = await wrapped(*args, **kwargs)
        if span.is_recording():
            _set_exec_response_attributes(span, response)
        span.end()
    except Exception as e:
        _record_exception(span, e)
        span.end()
        raise

    try:
        if logger is not None:
            response = _wrap_process_streams(response, logger, command, ctx)
    except Exception as log_error:
        log.debug(f"Failed to wrap Modal process streams: {log_error}")

    return response

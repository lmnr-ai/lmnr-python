"""Shared helpers for emitting OTel log records from sandbox instrumentations.

Sandbox-style providers (Daytona, Modal, ...) all emit stdout/stderr as OTel
log events using a common shape. This module centralizes the ``LogStream``
enum and the ``emit_log`` helper, parameterized by provider name, so new
sandbox instrumentations can reuse the same behavior without copying it.
"""

import logging
import time
from enum import Enum

from opentelemetry.context import Context
from opentelemetry._logs import LogRecord, Logger
from opentelemetry._logs.severity import SeverityNumber


log = logging.getLogger(__name__)


class LogStream(Enum):
    STDOUT = "stdout"
    STDERR = "stderr"


def emit_log(
    provider: str,
    logger: Logger,
    stream: LogStream,
    content: str,
    ctx: Context,
    extra_attributes: dict[str, str] | None = None,
):
    """Emit a provider-scoped OTel log record for a stdout/stderr chunk.

    The emitted record uses ``{provider}.log.{stream}`` as the event name,
    ``{provider}.system`` / ``{provider}.log.stream`` as attributes, and a
    severity derived from the stream (STDOUT=INFO, STDERR=ERROR).
    """
    if not content:
        return

    try:
        event_name = f"{provider}.log.{stream.value}"
        severity = (
            SeverityNumber.INFO if stream == LogStream.STDOUT else SeverityNumber.ERROR
        )

        attributes: dict[str, str] = {
            f"{provider}.system": provider,
            f"{provider}.log.stream": stream.value,
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
        log.debug(f"Failed to emit {provider} log event: {e}")

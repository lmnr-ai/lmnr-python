"""Span utilities for Claude Agent instrumentation."""

from typing import Any

from lmnr import Laminar
from lmnr.opentelemetry_lib.tracing import get_current_context
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_IDS_PATH, SPAN_PATH
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import get_input_from_func_args, is_method, json_dumps

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan

logger = get_default_logger(__name__)


def span_name(to_wrap: dict[str, str]) -> str:
    """Generate span name from method metadata."""
    class_name = to_wrap.get("class_name")
    method = to_wrap.get("method")
    return f"{class_name}.{method}" if class_name else method


def record_input(span, wrapped, args, kwargs):
    """Record function input as span attribute."""
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


def record_output(span, to_wrap, value):
    """Record function output as span attribute."""
    try:
        span.set_attribute("lmnr.span.output", json_dumps(value))
    except Exception:
        pass


def get_span_context_payload() -> dict[str, str] | None:
    """Extract current span context for publishing to proxy."""
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


def publish_span_context_for_transport(transport) -> None:
    """Publish span context to transport's dedicated proxy."""
    if transport is None:
        logger.debug("No transport found")
        return

    # Get context from transport instance
    context: dict[str, Any] | None = getattr(transport, "__lmnr_context", None)
    if not context or not context.get("proxy"):
        logger.debug("No proxy found for transport")
        return

    # Get span context
    payload = get_span_context_payload()
    if not payload:
        return

    try:
        context["proxy"].set_current_trace(
            trace_id=payload["trace_id"],
            span_id=payload["span_id"],
            project_api_key=payload["project_api_key"],
            span_path=payload["span_path"],
            span_ids_path=payload["span_ids_path"],
            laminar_url=payload["laminar_url"],
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.debug("Failed to publish span context to proxy: %s", e)

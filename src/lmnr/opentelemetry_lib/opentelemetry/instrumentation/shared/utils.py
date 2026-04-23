from copy import deepcopy
from typing import Any

import traceback

from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from lmnr.opentelemetry_lib.tracing.context import get_current_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.laminar import Laminar

logger = get_default_logger(__name__)


def dont_throw(func):
    def wrapper(*args, **kwargs):
        logger = get_default_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.debug(
                "Laminar failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            return None

    return wrapper


def set_span_attribute(
    span: Span, attribute_name: str, attribute_value: AttributeValue
):
    if attribute_value is not None and attribute_value != "":
        span.set_attribute(attribute_name, attribute_value)


def to_dict(obj: Any) -> dict:
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return deepcopy(obj)
        elif obj is None:
            return {}
        else:
            return dict(obj)
    except Exception as e:
        logger.debug(f"Error converting to dict: {obj}, error: {e}")
        return {}


def extract_json_schema(schema: dict | BaseModel) -> dict:
    if isinstance(schema, dict):
        return schema
    elif hasattr(schema, "model_json_schema") and callable(schema.model_json_schema):
        return schema.model_json_schema()
    else:
        return {}


def safe_start_span(
    name: str,
    context: Context | None = None,
    attributes: dict[str, AttributeValue] | None = None,
    span_type: str = "DEFAULT",
) -> Span | None:
    if not Laminar.is_initialized():
        return None
    try:
        return Laminar.start_span(
            name, context=context, attributes=attributes, span_type=span_type
        )
    except Exception:
        logger.debug(f"Failed to start span: {name}", exc_info=True)
        return None


def _non_recording_span() -> Span:
    """Return a no-op span. All Span methods (set_attribute, end, ...) are
    safe no-ops; ``is_recording()`` returns False."""
    return NonRecordingSpan(
        SpanContext(trace_id=0, span_id=0, is_remote=False)
    )


def safe_start_active_span(name: str, **kwargs: Any) -> Span:
    """Like ``Laminar.start_active_span`` but never raises.

    On any failure (Laminar not initialised, internal error, etc.) returns a
    ``NonRecordingSpan`` so callers can always treat the result as a valid
    Span without null-checks. ``is_recording()`` will be False, which lets the
    standard ``if span.is_recording(): set_attrs(...)`` pattern naturally
    skip work for the no-op span.
    """
    try:
        return Laminar.start_active_span(name=name, **kwargs)
    except Exception:
        logger.debug(f"Failed to start active span: {name}", exc_info=True)
        return _non_recording_span()


def safe_end_span(span: Span) -> None:
    """End a span, swallowing any exception so caller code can't break."""
    try:
        span.end()
    except Exception:
        logger.debug("Failed to end span", exc_info=True)


def safe_is_recording(span: Span) -> bool:
    """Return ``span.is_recording()``, defaulting to False on any error."""
    try:
        return span.is_recording()
    except Exception:
        return False


def safe_get_current_context() -> Context:
    """Return Laminar's current isolated context, or an empty ``Context`` on
    any error."""
    try:
        return get_current_context()
    except Exception:
        logger.debug("Failed to get current context", exc_info=True)
        return Context()

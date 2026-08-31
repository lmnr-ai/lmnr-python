from copy import deepcopy
from typing import Any

import traceback

from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from lmnr.opentelemetry_lib.tracing.attributes import SPAN_TYPE
from lmnr.opentelemetry_lib.tracing.tracer import get_tracer_with_context
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.types import LaminarSpanType

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


def to_dict(obj: Any) -> dict[str, Any]:
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


def extract_json_schema(schema: dict[str, Any] | BaseModel) -> dict[str, Any]:
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
    span_type: LaminarSpanType = "DEFAULT",
    start_time: int | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Span | None:
    """Start a span, returning None instead of raising if that is not possible.

    `start_time` (ns) and `kind` are deliberately NOT exposed on the public
    `Laminar.start_span`, but some instrumentations genuinely need them: the
    OpenAI responses/assistants wrappers only learn a call happened once it has
    finished, so they open the span retroactively at the recorded start time.
    When either is requested we go through the tracer directly and stamp the
    Laminar-specific attributes ourselves, so the public API stays unchanged.
    """
    if not Laminar.is_initialized():
        return None
    try:
        if start_time is None and kind is SpanKind.INTERNAL:
            return Laminar.start_span(
                name, context=context, attributes=attributes, span_type=span_type
            )
        with get_tracer_with_context() as (tracer, isolated_context):
            return tracer.start_span(
                name,
                context=context or isolated_context,
                kind=kind,
                start_time=start_time,
                attributes={**(attributes or {}), SPAN_TYPE: span_type},
            )
    except Exception:
        logger.debug(f"Failed to start span: {name}", exc_info=True)
        return None

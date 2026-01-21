from copy import deepcopy
from typing import Any

import traceback

from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

from lmnr.sdk.log import get_default_logger

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

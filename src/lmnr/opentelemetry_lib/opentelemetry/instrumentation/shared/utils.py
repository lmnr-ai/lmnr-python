import traceback

from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from lmnr.sdk.log import get_default_logger


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

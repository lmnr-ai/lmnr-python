from pydantic import BaseModel
from opentelemetry.sdk.trace import Span
from opentelemetry.util.types import AttributeValue


def model_as_dict(model: BaseModel | dict) -> dict:
    if isinstance(model, BaseModel) and hasattr(model, "model_dump"):
        return model.model_dump()
    elif isinstance(model, dict):
        return model
    else:
        return dict(model)


def set_span_attribute(span: Span, key: str, value: AttributeValue | None):
    if value is None or value == "":
        return
    span.set_attribute(key, value)

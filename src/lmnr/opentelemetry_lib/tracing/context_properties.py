import copy

from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    TRACING_LEVEL,
)

from opentelemetry.context import Context, attach, set_value, get_value
from opentelemetry.sdk.trace import Span
from opentelemetry import trace


# TODO: delete this once deprecated Laminar.with_labels is removed. The logic
# should be moved into Laminar.set_tracing_level
def set_association_properties(properties: dict) -> None:
    attach(set_value("association_properties", properties))

    span = trace.get_current_span()
    _set_association_properties_attributes(span, properties)


# TODO: delete this once deprecated Laminar.with_labels is removed
def get_association_properties(context: Context | None = None) -> dict:
    return get_value("association_properties", context) or {}


# TODO: delete this once deprecated Laminar.with_labels is removed. The logic
# should be moved into Laminar.set_tracing_level
def update_association_properties(
    properties: dict,
    set_on_current_span: bool = True,
    context: Context | None = None,
) -> None:
    """Only adds or updates properties that are not already present"""
    association_properties = get_value("association_properties", context) or {}
    association_properties.update(properties)

    attach(set_value("association_properties", association_properties, context))

    if set_on_current_span:
        span = trace.get_current_span()
        _set_association_properties_attributes(span, properties)


# TODO: this logic should be moved into Laminar.set_tracing_level
def remove_association_properties(properties: dict) -> None:
    props: dict = copy.copy(get_value("association_properties") or {})
    for k in properties.keys():
        props.pop(k, None)
    set_association_properties(props)


def _set_association_properties_attributes(span: Span, properties: dict) -> None:
    if not span.is_recording():
        return
    for key, value in properties.items():
        if key == TRACING_LEVEL:
            span.set_attribute(f"lmnr.internal.{TRACING_LEVEL}", value)
            continue
        if (
            key in ["langgraph.edges", "langgraph.nodes"]
            and span.name != "LangGraph.workflow"
        ):
            continue
        span.set_attribute(f"{ASSOCIATION_PROPERTIES}.{key}", value)

from opentelemetry.trace import Span
from lmnr.opentelemetry_lib.tracing.span import LaminarSpan
from lmnr.opentelemetry_lib.tracing.attributes import (
    ASSOCIATION_PROPERTIES,
    ROLLOUT_SESSION_ID,
    USER_ID,
    SESSION_ID,
    TRACE_TYPE,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    attach_context,
    set_value,
    CONTEXT_USER_ID_KEY,
    CONTEXT_SESSION_ID_KEY,
    CONTEXT_ROLLOUT_SESSION_ID_KEY,
    CONTEXT_TRACE_TYPE_KEY,
    CONTEXT_METADATA_KEY,
)


def set_association_props_in_context(span: Span):
    """Set association properties from span in context before push_span_context.

    Returns the token that needs to be detached when the span ends.
    """
    if not isinstance(span, LaminarSpan):
        return None

    props = span.laminar_association_properties
    user_id_key = f"{ASSOCIATION_PROPERTIES}.{USER_ID}"
    session_id_key = f"{ASSOCIATION_PROPERTIES}.{SESSION_ID}"
    rollout_session_id_key = ROLLOUT_SESSION_ID
    trace_type_key = f"{ASSOCIATION_PROPERTIES}.{TRACE_TYPE}"

    # Extract values from props
    extracted_user_id = props.get(user_id_key)
    extracted_session_id = props.get(session_id_key)
    extracted_rollout_session_id = props.get(rollout_session_id_key)
    extracted_trace_type = props.get(trace_type_key)

    # Extract metadata from props (keys without ASSOCIATION_PROPERTIES prefix)
    metadata_dict = {}
    for key, value in props.items():
        if not key.startswith(f"{ASSOCIATION_PROPERTIES}."):
            metadata_dict[key] = value

    # Set context with association props
    current_ctx = get_current_context()
    ctx_with_props = current_ctx
    if extracted_user_id:
        ctx_with_props = set_value(
            CONTEXT_USER_ID_KEY, extracted_user_id, ctx_with_props
        )
    if extracted_session_id:
        ctx_with_props = set_value(
            CONTEXT_SESSION_ID_KEY, extracted_session_id, ctx_with_props
        )
    if extracted_rollout_session_id:
        ctx_with_props = set_value(
            CONTEXT_ROLLOUT_SESSION_ID_KEY, extracted_rollout_session_id, ctx_with_props
        )
    if extracted_trace_type:
        ctx_with_props = set_value(
            CONTEXT_TRACE_TYPE_KEY, extracted_trace_type, ctx_with_props
        )
    if metadata_dict:
        ctx_with_props = set_value(CONTEXT_METADATA_KEY, metadata_dict, ctx_with_props)

    return attach_context(ctx_with_props)

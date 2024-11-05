from enum import Enum
from opentelemetry.semconv_ai import SpanAttributes

SPAN_INPUT = "lmnr.span.input"
SPAN_OUTPUT = "lmnr.span.output"
SPAN_TYPE = "lmnr.span.type"
SPAN_PATH = "lmnr.span.path"
SPAN_INSTRUMENTATION_SOURCE = "lmnr.span.instrumentation_source"
OVERRIDE_PARENT_SPAN = "lmnr.internal.override_parent_span"

ASSOCIATION_PROPERTIES = "lmnr.association.properties"
SESSION_ID = "session_id"
USER_ID = "user_id"
TRACE_TYPE = "trace_type"


# exposed to the user, configurable
class Attributes(Enum):
    # == This is the minimum set of attributes for a proper LLM span ==
    #
    # not SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
    INPUT_TOKEN_COUNT = "gen_ai.usage.input_tokens"
    # not SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
    OUTPUT_TOKEN_COUNT = "gen_ai.usage.output_tokens"
    TOTAL_TOKEN_COUNT = SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    PROVIDER = SpanAttributes.LLM_SYSTEM
    REQUEST_MODEL = SpanAttributes.LLM_REQUEST_MODEL
    RESPONSE_MODEL = SpanAttributes.LLM_RESPONSE_MODEL
    #
    ## == End of minimum set ==
    # == Additional attributes ==
    #
    INPUT_COST = "gen_ai.usage.input_cost"
    OUTPUT_COST = "gen_ai.usage.output_cost"
    TOTAL_COST = "gen_ai.usage.cost"
    #
    # == End of additional attributes ==

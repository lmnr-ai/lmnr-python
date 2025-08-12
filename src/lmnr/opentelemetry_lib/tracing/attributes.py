from enum import Enum
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_SYSTEM,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_RESPONSE_ID,
)
from opentelemetry.semconv_ai import SpanAttributes

SPAN_INPUT = "lmnr.span.input"
SPAN_OUTPUT = "lmnr.span.output"
SPAN_TYPE = "lmnr.span.type"
SPAN_PATH = "lmnr.span.path"
SPAN_IDS_PATH = "lmnr.span.ids_path"
PARENT_SPAN_PATH = "lmnr.span.parent_path"
PARENT_SPAN_IDS_PATH = "lmnr.span.parent_ids_path"
SPAN_INSTRUMENTATION_SOURCE = "lmnr.span.instrumentation_source"
SPAN_SDK_VERSION = "lmnr.span.sdk_version"
SPAN_LANGUAGE_VERSION = "lmnr.span.language_version"
HUMAN_EVALUATOR_OPTIONS = "lmnr.span.human_evaluator_options"

ASSOCIATION_PROPERTIES = "lmnr.association.properties"
SESSION_ID = "session_id"
USER_ID = "user_id"
TRACE_TYPE = "trace_type"
TRACING_LEVEL = "tracing_level"


# exposed to the user, configurable
class Attributes(Enum):
    # == This is the minimum set of attributes for a proper LLM span ==
    #
    INPUT_TOKEN_COUNT = GEN_AI_USAGE_INPUT_TOKENS
    OUTPUT_TOKEN_COUNT = GEN_AI_USAGE_OUTPUT_TOKENS
    TOTAL_TOKEN_COUNT = SpanAttributes.LLM_USAGE_TOTAL_TOKENS
    PROVIDER = GEN_AI_SYSTEM
    REQUEST_MODEL = GEN_AI_REQUEST_MODEL
    RESPONSE_MODEL = GEN_AI_RESPONSE_MODEL
    #
    ## == End of minimum set ==
    # == Additional attributes ==
    #
    INPUT_COST = "gen_ai.usage.input_cost"
    OUTPUT_COST = "gen_ai.usage.output_cost"
    TOTAL_COST = "gen_ai.usage.cost"
    RESPONSE_ID = GEN_AI_RESPONSE_ID
    #
    # == End of additional attributes ==

"""Attribute-name constants for the Langfuse -> Laminar bridge. Pure data."""

from __future__ import annotations

LANGFUSE_TRACER_NAME = "langfuse-sdk"

# Langfuse attribute names (mirrors langfuse._client.attributes.LangfuseOtelSpanAttributes)
# Duplicated here so this module has no hard import dependency on langfuse.
LF_TRACE_INPUT = "langfuse.trace.input"
LF_TRACE_OUTPUT = "langfuse.trace.output"
LF_TRACE_TAGS = "langfuse.trace.tags"
LF_TRACE_METADATA_PREFIX = "langfuse.trace.metadata"
LF_TRACE_USER_ID = "user.id"
LF_TRACE_SESSION_ID = "session.id"
# Older names Langfuse's server still reads (before `user.id` / `session.id`).
LF_TRACE_COMPAT_USER_ID = "langfuse.user.id"
LF_TRACE_COMPAT_SESSION_ID = "langfuse.session.id"
# Set on a span Langfuse must treat as the trace root despite its parent.
LF_AS_ROOT = "langfuse.internal.as_root"

# Metadata keys Langfuse's server reads as trace session / user / tags, from
# observation or trace metadata. The `langfuse.openai` integration documents
# them (`metadata={"langfuse_session_id": ...}`) and records them as plain
# observation metadata.
LF_METADATA_SESSION_ID = "langfuse_session_id"
LF_METADATA_USER_ID = "langfuse_user_id"
LF_METADATA_TAGS = "langfuse_tags"
LF_METADATA_TRACE_KEYS = {LF_METADATA_SESSION_ID, LF_METADATA_USER_ID, LF_METADATA_TAGS}

LF_OBSERVATION_TYPE = "langfuse.observation.type"
LF_OBSERVATION_INPUT = "langfuse.observation.input"
LF_OBSERVATION_OUTPUT = "langfuse.observation.output"
LF_OBSERVATION_MODEL = "langfuse.observation.model.name"
LF_OBSERVATION_MODEL_PARAMETERS = "langfuse.observation.model.parameters"
LF_OBSERVATION_METADATA_PREFIX = "langfuse.observation.metadata"
LF_OBSERVATION_USAGE_DETAILS = "langfuse.observation.usage_details"
LF_OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
LF_OBSERVATION_LEVEL = "langfuse.observation.level"
LF_OBSERVATION_STATUS_MESSAGE = "langfuse.observation.status_message"

# Laminar keys with no OTel semconv constant. The others the bridge writes
# (`gen_ai.input.messages`, `gen_ai.usage.input_tokens`, ...) come from
# `opentelemetry.semconv._incubating.attributes.gen_ai_attributes`.
GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
GEN_AI_USAGE_INPUT_COST = "gen_ai.usage.input_cost"
GEN_AI_USAGE_OUTPUT_COST = "gen_ai.usage.output_cost"
GEN_AI_USAGE_COST = "gen_ai.usage.cost"
GEN_AI_REQUEST_REASONING_LEVEL = "gen_ai.request.reasoning.level"
GEN_AI_REQUEST_REASONING_SUMMARY = "gen_ai.request.reasoning_summary"
GEN_AI_REQUEST_SERVICE_TIER = "gen_ai.request.service_tier"

# LangChain's `ls_provider` values that differ from the provider names in
# Laminar's model prices.
LS_PROVIDER_ALIASES = {
    "amazon_bedrock": "bedrock",
    "google_vertexai": "vertex_ai",
    "google_genai": "gemini",
}

# Langfuse "observation types" that represent LLM calls. See
# langfuse._client.constants.ObservationTypeGenerationLike. Any of these maps
# to Laminar's LLM span type.
LF_LLM_OBSERVATION_TYPES = {"generation", "completion", "embedding"}
LF_TOOL_OBSERVATION_TYPES = {"tool"}

# --- OpenInference semantic conventions -------------------------------------
# Langfuse's docs recommend openinference instrumentations for groq and
# google_genai (e.g. `openinference-instrumentation-google-genai`). Those emit
# a flat, indexed attribute layout — `llm.input_messages.0.message.role`,
# `llm.input_messages.0.message.content`, `llm.output_messages.0...`,
# `llm.token_count.prompt`, `llm.tools.0.tool.json_schema`, etc. — completely
# different from both Langfuse's `langfuse.*` blobs and Laminar's GenAI shape.
# See the openinference-semantic-conventions package in the Arize-ai/
# openinference repo. We translate the flat layout into Laminar / OTel GenAI
# conventions.
OI_SPAN_KIND = "openinference.span.kind"
OI_LLM_MODEL_NAME = "llm.model_name"
OI_LLM_PROVIDER = "llm.provider"
OI_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
OI_LLM_INPUT_MESSAGES = "llm.input_messages"
OI_LLM_OUTPUT_MESSAGES = "llm.output_messages"
OI_LLM_TOOLS = "llm.tools"
OI_TOKEN_PROMPT = "llm.token_count.prompt"
OI_TOKEN_COMPLETION = "llm.token_count.completion"
OI_TOKEN_TOTAL = "llm.token_count.total"
OI_INPUT_VALUE = "input.value"
OI_OUTPUT_VALUE = "output.value"
# openinference.span.kind values that mean "LLM call".
OI_LLM_SPAN_KINDS = {"LLM"}

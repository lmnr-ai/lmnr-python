"""Attribute-name constants for the Langfuse <-> Laminar bridge.

Pure data — no behavior. Split out of the original monolithic
`langfuse/__init__.py` so `translate.py` / `processor.py` don't have to import
the whole orchestrator module just to reach a constant.
"""

from __future__ import annotations

LANGFUSE_TRACER_NAME = "langfuse-sdk"

# Langfuse attribute names (mirrors langfuse._client.attributes.LangfuseOtelSpanAttributes)
# Duplicated here so this module has no hard import dependency on langfuse.
_TRACE_INPUT = "langfuse.trace.input"
_TRACE_OUTPUT = "langfuse.trace.output"
_TRACE_TAGS = "langfuse.trace.tags"
_TRACE_METADATA_PREFIX = "langfuse.trace.metadata"
_TRACE_USER_ID = "user.id"
_TRACE_SESSION_ID = "session.id"

_OBSERVATION_TYPE = "langfuse.observation.type"
_OBSERVATION_INPUT = "langfuse.observation.input"
_OBSERVATION_OUTPUT = "langfuse.observation.output"
_OBSERVATION_MODEL = "langfuse.observation.model.name"
_OBSERVATION_USAGE_DETAILS = "langfuse.observation.usage_details"
_OBSERVATION_COST_DETAILS = "langfuse.observation.cost_details"
_OBSERVATION_METADATA_PREFIX = "langfuse.observation.metadata"

_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"
_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GEN_AI_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
_GEN_AI_USAGE_INPUT_COST = "gen_ai.usage.input_cost"
_GEN_AI_USAGE_OUTPUT_COST = "gen_ai.usage.output_cost"
_GEN_AI_USAGE_TOTAL_COST = "gen_ai.usage.cost"

# Langfuse "observation types" that represent LLM calls. See
# langfuse._client.constants.ObservationTypeGenerationLike. Any of these maps
# to Laminar's LLM span type.
_LLM_OBSERVATION_TYPES = {"generation", "completion", "embedding"}
_TOOL_OBSERVATION_TYPES = {"tool"}

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
_OI_SPAN_KIND = "openinference.span.kind"
_OI_LLM_MODEL_NAME = "llm.model_name"
_OI_LLM_INPUT_MESSAGES = "llm.input_messages"
_OI_LLM_OUTPUT_MESSAGES = "llm.output_messages"
_OI_LLM_TOOLS = "llm.tools"
_OI_TOKEN_PROMPT = "llm.token_count.prompt"
_OI_TOKEN_COMPLETION = "llm.token_count.completion"
_OI_TOKEN_TOTAL = "llm.token_count.total"
_OI_INPUT_VALUE = "input.value"
_OI_OUTPUT_VALUE = "output.value"
# openinference.span.kind values that mean "LLM call".
_OI_LLM_SPAN_KINDS = {"LLM"}

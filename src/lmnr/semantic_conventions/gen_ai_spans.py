# source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
# last updated: 2024-08-26

REQUEST_MODEL: str = "gen_ai.request.model"
RESPONSE_MODEL: str = "gen_ai.response.model"
PROVIDER: str = "gen_ai.system"
INPUT_TOKEN_COUNT: str = "gen_ai.usage.input_tokens"
OUTPUT_TOKEN_COUNT: str = "gen_ai.usage.output_tokens"
TOTAL_TOKEN_COUNT: str = "gen_ai.usage.total_tokens"  # custom, not in the spec
# https://github.com/openlit/openlit/blob/main/sdk/python/src/openlit/semcov/__init__.py
COST: str = "gen_ai.usage.cost"

OPERATION: str = "gen_ai.operation.name"

FREQUENCY_PENALTY: str = "gen_ai.request.frequency_penalty"
TEMPERATURE: str = "gen_ai.request.temperature"
MAX_TOKENS: str = "gen_ai.request.max_tokens"
PRESENCE_PENALTY: str = "gen_ai.request.presence_penalty"
STOP_SEQUENCES: str = "gen_ai.request.stop_sequences"
TEMPERATURE: str = "gen_ai.request.temperature"
TOP_P: str = "gen_ai.request.top_p"
TOP_K: str = "gen_ai.request.top_k"

# https://github.com/openlit/openlit/blob/main/sdk/python/src/openlit/semcov/__init__.py
STREAM: str = "gen_ai.request.is_stream"

FINISH_REASONS = "gen_ai.response.finish_reasons"

__all__ = [
    "REQUEST_MODEL",
    "RESPONSE_MODEL",
    "PROVIDER",
    "INPUT_TOKEN_COUNT",
    "OUTPUT_TOKEN_COUNT",
    "TOTAL_TOKEN_COUNT",
    "COST",
    "OPERATION",
    "FREQUENCY_PENALTY",
    "TEMPERATURE",
    "MAX_TOKENS",
    "PRESENCE_PENALTY",
    "STOP_SEQUENCES",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "STREAM",
    "FINISH_REASONS",
]

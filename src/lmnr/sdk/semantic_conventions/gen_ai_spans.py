# source: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
# last updated: 2024-08-26

REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
PROVIDER = "gen_ai.system"
INPUT_TOKEN_COUNT = "gen_ai.usage.input_tokens"
OUTPUT_TOKEN_COUNT = "gen_ai.usage.output_tokens"
TOTAL_TOKEN_COUNT = "gen_ai.usage.total_tokens"  # custom, not in the spec
COST = "gen_ai.usage.cost"  # https://github.com/openlit/openlit/blob/main/sdk/python/src/openlit/semcov/__init__.py

OPERATION = "gen_ai.operation.name"

FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
TEMPERATURE = "gen_ai.request.temperature"
MAX_TOKENS = "gen_ai.request.max_tokens"
PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
STOP_SEQUENCES = "gen_ai.request.stop_sequences"
TEMPERATURE = "gen_ai.request.temperature"
TOP_P = "gen_ai.request.top_p"
TOP_K = "gen_ai.request.top_k"
STREAM = "gen_ai.request.is_stream"  # https://github.com/openlit/openlit/blob/main/sdk/python/src/openlit/semcov/__init__.py

FINISH_REASONS = "gen_ai.response.finish_reasons"

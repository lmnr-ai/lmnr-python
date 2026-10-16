"""OpenTelemetry context keys shared across instrumentations.

Kept as a leaf module (no provider SDK imports) so that instrumentations which
only need the key, e.g. the OpenAI Responses wrapper, don't pull in optional
packages such as `agents` (openai-agents) through their parent package.
"""

from opentelemetry.context import create_key

DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW = (
    "LMNR_DISABLE_OPENAI_RESPONSES_INSTRUMENTATION"
)
DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY = create_key(
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW
)

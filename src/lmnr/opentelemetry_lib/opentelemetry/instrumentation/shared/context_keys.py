"""OpenTelemetry context keys shared across instrumentations.

Kept as a leaf module (no provider SDK imports) so that instrumentations which
only need the key, e.g. the OpenAI Responses wrapper, don't pull in optional
packages such as `agents` (openai-agents) through their parent package.
"""

from __future__ import annotations

from contextvars import ContextVar

from opentelemetry.context import create_key

DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW = (
    "LMNR_DISABLE_OPENAI_RESPONSES_INSTRUMENTATION"
)
DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY = create_key(
    DISABLE_OPENAI_RESPONSES_INSTRUMENTATION_CONTEXT_KEY_RAW
)

# Same idea as the key above, for `LitellmModel`: the `agents.generation` span
# the openai-agents processor emits already carries the whole model call, so the
# nested `litellm.completion` span the litellm instrumentor would open is a
# duplicate of it - same tokens, same cost, counted twice.
DISABLE_LITELLM_INSTRUMENTATION_CONTEXT_KEY_RAW = "LMNR_DISABLE_LITELLM_INSTRUMENTATION"
DISABLE_LITELLM_INSTRUMENTATION_CONTEXT_KEY = create_key(
    DISABLE_LITELLM_INSTRUMENTATION_CONTEXT_KEY_RAW
)

# Topology of the LangGraph graph whose `Pregel.stream` step is running, as
# `{"langgraph.nodes": <json>, "langgraph.edges": <json>}`. The langgraph wrapper
# sets it around each step of the stream; the LangChain callback handler
# consumes (and clears) it on the first chain run it starts, the graph's own,
# and records it on that span. Deliberately a plain ContextVar, not the OTel
# context: the wrapper resets it after every step, and resetting the OTel
# context there would also undo the run spans the handler attached.
LANGGRAPH_TOPOLOGY: ContextVar[dict[str, str] | None] = ContextVar(
    "lmnr_langgraph_topology", default=None
)

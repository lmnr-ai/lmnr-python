"""OpenTelemetry deepagents instrumentation for Laminar.

Auto-injects a `LaminarMiddleware` into every deep agent built by
`deepagents.create_deep_agent`. The middleware opens:

- one `DEFAULT` root span per top-level agent invocation (`before_agent` /
  `after_agent`), and
- one `TOOL` span per tool call (`wrap_tool_call` / `awrap_tool_call`).

LLM spans come from the existing Anthropic / OpenAI auto-instrumentors and
attach under whatever span is active when the model is called — so they
automatically nest under the root span, and tool calls that invoke a
subagent (e.g. `task`) become the natural parent of that subagent's own LLM
and tool spans. No extra subagent-specific span machinery is needed.
"""

from .instrumentor import DeepagentsInstrumentor, _instruments
from .middleware import LaminarMiddleware

__all__ = [
    "DeepagentsInstrumentor",
    "LaminarMiddleware",
    "_instruments",
]

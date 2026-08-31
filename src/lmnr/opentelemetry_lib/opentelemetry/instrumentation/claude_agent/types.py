from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    WrappedFunctionSpec,
)


class ClaudeAgentSpec(WrappedFunctionSpec, total=False):
    """claude-agent-sdk's per-method extras.

    `class_name` is what the span is named after (`"{class_name}.{method_name}"`),
    which is not always the same as `object_name` — the module-level `query` and
    `create_sdk_mcp_server` rows have no class at all and fall back to the bare
    method name. `should_publish_span_context` marks the calls that must hand the
    current span context to the CLI subprocess so its spans nest under ours.
    """

    class_name: str
    should_publish_span_context: bool

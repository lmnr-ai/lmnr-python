# lmnr-atif writer golden cases

Conformance cases for SDK trajectory **writers**: given the spans of a run, a
writer must produce exactly `trajectory.json`. Every SDK that writes lmnr-atif
runs these, so writers in different languages produce the same files.

Each directory holds:

- `spans.json`: `{"options": {...}, "spans": [SpanRecord, ...]}`
- `trajectory.json`: the expected output

Writers build with `producer = {"name": "lmnr-golden", "version": "0"}` plus
the case's `options` (`release`, `project_id`, `exported`, `complete`; all
optional). No media sink is used, so inline images become text placeholders.

## SpanRecord

One finished span, with only what the writer needs:

```jsonc
{
  "trace_id": "4f1c2b7e-9a3d-4e5f-8b6a-1c2d3e4f5a6b",           // Laminar UUID
  "span_id": "00000000-0000-0000-0000-000000000002",            // Laminar UUID
  "parent_span_id": "00000000-0000-0000-0000-000000000001",     // or null
  "name": "anthropic.chat",
  "span_type": "LLM",                                           // lmnr.span.type
  "start_time": "2026-10-28T09:00:00.010000Z",
  "end_time": "2026-10-28T09:00:01.900000Z",
  "attributes": { "gen_ai.input.messages": "[...]", ... },      // raw OTel values
  "error": { "type": "...", "message": "...", "stacktrace": "..." }  // optional, from the `exception` event
}
```

Attribute values are what the instrumentation set on the span: messages, tool
definitions and span input/output are JSON strings. The cases mirror what the
Python SDK's instrumentations record for each provider.

## Cases

| Case | Covers |
|---|---|
| `anthropic-tool-loop` | Anthropic content blocks, thinking, `tool_result` splitting, usage and cost, request params, tool span matching |
| `openai-chat-overrides` | OpenAI chat, `developer` role, per-call system prompt and tools overrides, malformed tool arguments, compaction (`input_messages`) and the call after it, session id |
| `responses-and-gemini` | OpenAI Responses items and `instructions`, built-in tools, Gemini parts, generated Gemini call ids, two conversations in one agent, `defs` dedup |
| `errors-dispatch-multitrace` | Failed LLM and tool calls, `is_error` tool results, injected text in a user message, a tool span no LLM call issued, two traces in one file |
| `subagents` | Agent markers, nested LLM spans, same-name markers, implicit subagents, a subagent no tool spawned |
| `root-agent-collapse` | A lone top-level agent is the root agent |

## Updating

The Python SDK regenerates `trajectory.json` with
`LMNR_ATIF_UPDATE_GOLDEN=1 uv run pytest tests/test_lmnr_atif_builder.py`.
Review the diff by hand: these files are the contract.

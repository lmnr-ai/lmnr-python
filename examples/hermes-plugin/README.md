# lmnr-hermes

Laminar tracing plugin for [Hermes Agent](https://github.com/nousresearch/hermes-agent).

Emits nested OpenTelemetry spans for every Hermes conversation turn, tool call, and
subagent delegation — and lets Laminar's raw-provider instrumentors (OpenAI,
Anthropic, Bedrock) nest their own GenAI spans under each turn for free.

## What gets traced

Per turn (one Hermes `run_conversation` call):

- `hermes.turn` (root span) — input: user message + model/platform; output:
  assistant response. Session and user IDs are attached.
- `tool.<name>` (child, `span_type=TOOL`) — one per tool call, with args,
  result, and `duration_ms`.
- `subagent.<role>` (child) — one per finished subagent delegation.
- OpenAI / Anthropic / Bedrock spans — nested under the turn automatically,
  with token usage and messages.

Span attributes include `hermes.model`, `hermes.provider`, `hermes.api_mode`,
`hermes.finish_reason`, `hermes.usage.*`, and `hermes.tool_duration_ms`.

## Install (local dev, no pip publish)

```bash
# 1. Editable install — exposes the hermes_agent.plugins entry point
pip install -e path/to/lmnr-python/examples/hermes-plugin

# 2. Enable in Hermes
hermes plugins enable lmnr-hermes

# 3. Set your Laminar project key and run
export LMNR_PROJECT_API_KEY=...
hermes run
```

Alternative: drop the `src/lmnr_hermes/` directory as `~/.hermes/plugins/lmnr-hermes/`
and Hermes will discover it on startup (the `plugin.yaml` manifest ships inside
that directory). Then `hermes plugins enable lmnr-hermes`.

## Configuration

The plugin reads standard Laminar environment variables:

| Variable               | Purpose                                           |
|------------------------|---------------------------------------------------|
| `LMNR_PROJECT_API_KEY` | Your project key (required unless OTel env is set) |
| `LMNR_BASE_URL`        | Override the Laminar endpoint (default: api.lmnr.ai) |

If the key is missing, the plugin no-ops silently — Hermes keeps working
without tracing.

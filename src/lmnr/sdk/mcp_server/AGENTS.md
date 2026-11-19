# sdk/mcp_server: `wrap_mcp_server` (MCP observability, LAM-2563)

`wrap_mcp_server(server, *, context=True, record_payloads=True, redact_keys=None, turn_gap_seconds=20)` makes every `tools/call` / `prompts/get` / `resources/read` a server handles a root SERVER span. The span contract (names, attributes, `error.type` strings, `context` field, redaction vectors, turns) is shared with the TS SDK (`wrapMcpServer`) and the app-server's own MCP server; the MCP dashboard queries those exact names, so change them in all three or not at all.

## Layout: settings here, spans in `McpInstrumentor`

There is ONE server-span code path, the instrumentor's (`instrumentation/mcp/server.py`, driven by its `_handle_request` hook on 1.x and `OpenTelemetryMiddleware.__call__` hook on 2.x). `wrap_mcp_server` only attaches an `McpServerSettings` (`instrumentation/mcp/settings.py`) to the server, and the hooks read it; servers without one are recorded with `UNWRAPPED`. Never add a second span path here (an earlier version had its own middleware / handler dict, plus markers so the instrumentor stood aside; it duplicated every attribute and error rule).

- `wrap.py`: finds the mcp low-level `Server` behind the object (`_lowlevel_server` on `MCPServer`, `_mcp_server` on mcp 1.x `FastMCP` and standalone FastMCP 3/4, else the object itself) and sets `SETTINGS_ATTR` on it. Never raises; an unsupported object is returned unchanged with a warning. Wrapping twice is a no-op.
  - mcp 2.x: the middleware `ctx` has no reference back to the server, so the settings also go on a fresh `OpenTelemetryMiddleware` put FIRST in `Server.middleware` (other instances removed: each would open a span). First = before params validation (where `context` must already be gone) and around FastMCP 4's own dispatch middleware (whose spans it suppresses). FastMCP 4 removes the SDK middleware, so this also puts one back.
  - mcp 1.x: the hook's `instance` is the low-level `Server`.
  - Builds the framework's tool-schema lookup (`_schema_lookup`) and FastMCP's suppression switch into the settings.
- `fastmcp_compat.py`: turning standalone FastMCP's own spans off (wrap-time only).
- In `instrumentation/mcp/`: `settings.py` (settings, `McpRequest`), `server.py` (`ServerSpan`: start / activate / record / end, error mapping), `context_field.py`, `turns.py`, `redaction.py`. `lmnr.opentelemetry_lib.tracing.id_generator`: the forced-trace-id generator (below).
- Consequence: the spans need the MCP instrumentation, which `Laminar.initialize` turns on by default. With `Instruments.MCP` disabled a wrapped server gets no Laminar spans (and on 2.x the re-added `OpenTelemetryMiddleware` opens mcp's native span).

## Behaviour that must not change (wrapped servers)

Pinned by `test_wrap_mcp_server_v1.py` / `_v2.py`. Each was a decision; a cleanup must keep all of them.

- One ROOT SERVER span per `tools/call` / `prompts/get` / `resources/read`, including failures no handler saw (unknown tool, bad arguments, missing handler). None for `initialize`, `ping`, listing, notifications.
- Name `{method} {target}`; the attribute set, `error.type` values and `context` rules below; payloads redacted; `record_payloads=False` keeps no arguments, results, context, URIs or exception text.
- Trace id order: `turnId` hash → session tracker → `traceparent` (no session) → random; the `traceparent` is always a link. Gap 20 s default, clamped to 25 s.
- Handler spans nest under the request span; FastMCP native spans suppressed for wrapped servers only (subclasses included); results reach the client unchanged; context stripping works even when Laminar is not initialized.
- A wrapped server's root span ignores OTel's suppress-instrumentation flag (an in-process caller's flag is not the server's); unwrapped servers honour it.
- Degrade to a random trace id when the provider can't force one; never raise.

## Native OTel in the SDKs (checked against mcp 2.3.0, fastmcp 4.2.2 / 3.4.9)

- **mcp 2.x**: `OpenTelemetryMiddleware` (default in every `Server.middleware`) opens a SERVER span per inbound request, `initialize` / `tools/list` / `server/discover` included: `{method} {target}`, `mcp.method.name`, `mcp.protocol.version`, `jsonrpc.request.id`, `gen_ai.operation.name`, `gen_ai.tool.name` / `gen_ai.prompt.name`, `error.type` + `rpc.response.status_code`. Parented on `_meta.traceparent` (not a root). No session id, client info, payloads, span type or `exception` event; unknown tool / bad arguments come out as `tool_error` (MCPServer folds them into `is_error`).
- **FastMCP 4** (on mcp 2.x): drops the SDK middleware and opens its own "seam" SERVER span per request (plus `mcp.session.id`, `fastmcp.*`); `error.type` is `ValidationError` / `NotFoundError` / `tool_error`. Ships `fastmcp.telemetry.suppress_fastmcp_telemetry()`.
- **FastMCP 3** (on mcp 1.x): SERVER spans per high-level op (tools/list too), no protocol version / request id, no switch to turn them off.
- **mcp 1.x**: nothing native.
- What the wrapper adds over all of them: root span + turn grouping (trace-id forcing), links instead of parents, Laminar span type, redacted payloads, `context`, client name/version, session for Laminar, exception events, `-32602` classification of folded errors, no spans for listing / handshake.

## Wrapped vs unwrapped servers (same span path)

- Same for both: span names, attributes (session, client, protocol, request id), payload keys, redaction of secret-looking keys, `error.type` values. So the MCP dashboard counts unwrapped servers too.
- Only wrapped: root spans + turn grouping + links, `traces()` limited to the three methods, the `context` field, `record_payloads` / `redact_keys`, FastMCP suppression.
- Only unwrapped: the span is parented on the client's `_meta.traceparent` (else the current context), and every request is traced (2.x: `initialize` and listing too). claude_agent relies on the parenting to nest SDK MCP tool calls under its agent span (`claude_agent/sdk_mcp.py`, which also limits those servers to `tools/call`). Unwrapped 2.x servers that removed `OpenTelemetryMiddleware` (FastMCP 4 included) get no span from us, and without `Laminar.initialize` mcp's native span is left in place.
- CLIENT spans (`{tool}.tool` / `{method}.mcp`) are unchanged and separate (`utils.py`).
- FastMCP: every request of a wrapped FastMCP server runs with FastMCP's spans suppressed (4: its own switch; 3: `fastmcp.server.telemetry.get_tracer` is swapped, once, for one that returns a pass-through tracer while our contextvar is set and the original otherwise). Other FastMCP servers in the process keep their native spans. "Is it FastMCP?" walks the class's MRO for a `fastmcp` module: a user's `FastMCP` subclass lives in their own package, and checking only the class kept its native spans on (two SERVER spans per request).
- **`Laminar.use_span` attaches Laminar's isolated context as the global OTel context, dropping values set on the previous global context.** FastMCP 4's suppression key is such a value, so `ServerSpan.activate` re-enters the suppression inside `use_span`; without that, the handler's part of the request got a FastMCP span again.

## Trace ids, roots and the id generator

- Every span is started with an empty `Context()`: always a root, also when the server runs in the caller's context (`Client(server)` / `DirectDispatcher`). A client `traceparent` is always a span link; its trace id is used only to start a session's turn or when there is no session (below). Never fake a remote parent: the dangling parent id hides the span in the trace view.
- Trace id order: `turnId` hash (`sha256("{session or ''}\n{turnId}")[:16]`) → `TurnTracker` (when the request has a session; a NEW turn starts with the call's `traceparent` trace id, random without one, and later calls within the gap join it) → `traceparent` trace id (no session) → random. The tracker is updated on every call with a session, including turn-id calls (the turn-id trace becomes the session's current turn).
- **Why the tracker beats `traceparent`** (contract change after a live run): an OTel-instrumented mcp 2.x client sends a fresh `traceparent` per call (its own `MCP send ...` / CLIENT span, a new trace each time), so with `traceparent` first every call of a turn became its own trace. Now it only seeds a new turn (so the first call shares the client's trace); each span still links to its own `traceparent`.
- Forcing: `LaminarIdGenerator` (installed in `init_tracing` on Laminar's own `TracerProvider`) returns the id set by `forced_trace_id(...)` (a contextvar set only around `start_span`). OTel calls the generator only for spans without a valid parent, so it can't leak into children.
- **Degrade path**: `can_force_trace_id(provider)` is false when Laminar's provider was not built by `init_tracing` (the provider is created once per process; a `LaminarSpanProcessor` added to a user-owned provider without `Laminar.initialize` produces no wrapper spans at all, like every Laminar instrumentation). Then the span just gets a random trace id (one trace per call) and a debug log is written once; links still work.
- With the instrumentor on, a client in this process sends its CLIENT span as `traceparent`: without a session (in-memory transports, stateless HTTP) each call takes its client span's trace id; with one, the turn starts in the first call's client trace and the tracker keeps later calls in it. Tests that need no `traceparent` at all suppress OTel instrumentation around a `Client(server)` call (the JSON-RPC dispatcher would still inject mcp's own client span).
- `turn_gap_seconds` is clamped to 25 (Signals wait for 30 s of quiet). No session (stdio, 2026-07-28 stateless HTTP without `openai/session`) means no tracker.

## Requests and sessions

- Session id: 2.x `ctx.session._connection.session_id`, else the `mcp-session-id` header; 1.x the header of the HTTP request in `message.message_metadata.request_context`; then `_meta["openai/session"]`. Set as `mcp.session.id` and `lmnr.association.properties.session_id` (so child spans inherit it via `use_span`).
- Client name / version: `session.client_params.client_info` (2.x; 2026-07-28 requests synthesize it from the envelope) / `.clientInfo` (1.x). `initialize` itself is never traced.
- `network.transport` / `network.protocol.name` only when an HTTP request is visible (`tcp` / `http`); stdio vs in-memory can't be told apart, so it is omitted there.
- Protocol version: 2.x `ctx.protocol_version`; 1.x the `mcp-protocol-version` header, else what `ServerSession` negotiated from the client's `initialize`.

## The `context` field

- Injected into `tools/list` results (copies; dicts on 2.x, models / `ServerResult` on 1.x), never into `required`. Off with `context=False` or when payloads are off (including `LMNR_TRACE_CONTENT=false`).
- Injected on 2.x into the `tools/list` result `call_next` returns, on 1.x into the response passed to `message.respond` (the `_Responder` proxy), so the low-level server's own tool cache keeps the real schemas.
- Stripped from `tools/call` arguments before the SDK validates them: 2.x by `dataclasses.replace(ctx, params=...)` in the middleware (runs before params validation); 1.x by `model_copy` of `req` before `_handle_request` hands it to the handler (where `validate_input` runs). FastMCP 3/4 reject unexpected arguments outright, so this is load-bearing.
- "Does the tool declare its own `context`?" is answered by the framework's registry when there is one (another server copy may have served `tools/list`): 2.x `Server.get_tool_input_schema` (set by `MCPServer` and FastMCP 4), mcp 1.x `FastMCP._tool_manager`, FastMCP 3 `await get_tool(name)`, 1.x low-level `_get_cached_tool_definition`. Otherwise what this process saw in `tools/list`; otherwise strip.
- The recorded `gen_ai.tool.call.arguments` are the client's arguments, `context` included.

## Errors

- `isError` results: `-32602` when the first text block reads like an unknown tool / validation failure (`_INVALID_PARAMS_TEXT`, matched against what mcp 1.x/2.x and FastMCP 3/4 actually return), else `tool_error`; no `rpc.response.status_code` (it was not a JSON-RPC error).
- Raised `McpError` / `MCPError`, or an `ErrorData` response (1.x): the code, plus `rpc.response.status_code`. 2.x `ValidationError` (params validation inside the chain): `-32602`. Anything else raised (2.x): the class name. On 1.x the hook only sees the response: `_handle_request` answers any other handler exception with code `0`, so that is the `error.type` (no extra hook just to recover the class name). 1.x's low-level `call_tool` decorator folds every tool exception into `isError` anyway (`tool_error`).
- A traced method with no handler is still a span with `-32601`: 2.x raises `MCPError(-32601)` inside the chain; 1.x `_handle_request` responds with that `ErrorData` itself.
- Without payloads, the `exception` event never carries the raised exception or its text (fixed messages: `Tool returned an error`, `JSON-RPC error {code}`, the class name).

## Tests

- `tests/test_instrumentations/test_mcp/test_wrap_mcp_server_v2.py` (mcp 2.x, FastMCP 4, streamable HTTP via `httpx2.ASGITransport`; the URL's host must include a port to pass the DNS-rebinding check, and `json_response=True` because ASGITransport buffers), `test_wrap_mcp_server_v1.py` (mcp 1.x overlay, as CI runs it), `test_wrap_mcp_server_units.py` (both majors). The unwrapped contract is in `test_mcp_v1.py` / `test_mcp_v2.py`. The FastMCP 3 test is skipped unless fastmcp 3 is installed: `uv run --with mcp==1.30.0 --with fastmcp==3.4.9 pytest tests/test_instrumentations/test_mcp -k wrap`.
- The dev dependency `fastmcp` is pinned to a 4.x (it needs mcp 2.x, matching the `mcp` pin). Under the 1.x overlay the 4.x package is still installed but unimportable; nothing imports it unless the server is a FastMCP one.

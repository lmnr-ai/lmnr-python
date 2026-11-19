# sdk/mcp_server: `wrap_mcp_server` (MCP observability, LAM-2563)

`wrap_mcp_server(server, *, context=True, record_payloads=True, redact_keys=None, turn_gap_seconds=20)` makes every `tools/call` / `prompts/get` / `resources/read` a server handles a root SERVER span. The span contract (names, attributes, `error.type` strings, `context` field, redaction vectors, turns) is shared with the TS SDK (`wrapMcpServer`) and the app-server's own MCP server; the MCP dashboard queries those exact names, so change them in all three or not at all.

## Layout

- `wrap.py`: entry point. Finds the mcp low-level `Server` behind the object (`_lowlevel_server` on `MCPServer`, `_mcp_server` on mcp 1.x `FastMCP` and standalone FastMCP 3/4, else the object itself), marks it (`OBSERVED_SERVER_ATTR`, from the leaf `instrumentation/mcp/observed.py`) and installs `v2` (it has a `middleware` list) or `v1` (it has a `request_handlers` dict). Never raises; an unsupported object is returned unchanged with a warning. Wrapping twice is a no-op (marker on the low-level server).
- `observe.py`: everything version-independent. `McpRequest` (what a request carries), `McpCall` (start → `activate` → `record_result` / `record_exception` → `end`), the `context` field (`tools_with_context`, `split_context`), error mapping.
- `v2.py` (mcp 2.x): `ObservedServerMiddleware`, put FIRST in `Server.middleware`; `OpenTelemetryMiddleware` instances are removed (matched by module + class name, so no private import).
- `v1.py` (mcp 1.x): `ObservedHandlers`, a `dict` subclass replacing `Server.request_handlers` whose `get` / `__getitem__` hand out traced handlers.
- `fastmcp_compat.py`: turning standalone FastMCP's own spans off. `turns.py`: trace-id choice. `redaction.py`. `config.py`: per-server state.
- `lmnr.opentelemetry_lib.tracing.id_generator`: the forced-trace-id generator on Laminar's provider (below).

## Native OTel in the SDKs (checked against mcp 2.3.0, fastmcp 4.2.2 / 3.4.9)

- **mcp 2.x**: `OpenTelemetryMiddleware` (default in every `Server.middleware`) opens a SERVER span per inbound request, `initialize` / `tools/list` / `server/discover` included: `{method} {target}`, `mcp.method.name`, `mcp.protocol.version`, `jsonrpc.request.id`, `gen_ai.operation.name`, `gen_ai.tool.name` / `gen_ai.prompt.name`, `error.type` + `rpc.response.status_code`. Parented on `_meta.traceparent` (not a root). No session id, client info, payloads, span type or `exception` event; unknown tool / bad arguments come out as `tool_error` (MCPServer folds them into `is_error`).
- **FastMCP 4** (on mcp 2.x): drops the SDK middleware and opens its own "seam" SERVER span per request (plus `mcp.session.id`, `fastmcp.*`); `error.type` is `ValidationError` / `NotFoundError` / `tool_error`. Ships `fastmcp.telemetry.suppress_fastmcp_telemetry()`.
- **FastMCP 3** (on mcp 1.x): SERVER spans per high-level op (tools/list too), no protocol version / request id, no switch to turn them off.
- **mcp 1.x**: nothing native.
- What the wrapper adds over all of them: root span + turn grouping (trace-id forcing), links instead of parents, Laminar span type, redacted payloads, `context`, client name/version, session for Laminar, exception events, `-32602` classification of folded errors, no spans for listing / handshake.

## Layering with `McpInstrumentor` (one span per request)

- The auto-instrumentor keeps doing what it did for unwrapped servers, and both its SERVER wrappers stand aside for wrapped ones; its CLIENT spans (including server-to-client requests made inside a handler, which nest under our span) are untouched.
  - 2.x: removing `OpenTelemetryMiddleware` from the wrapped server removes both the native span and the instrumentor's replacement of it. If one is added back later, `OBSERVED_REQUEST` (the request's `ServerSession`, set by our middleware) makes the instrumentor's wrapper pass through. It holds the session, not a flag, so an unwrapped in-process server called from a wrapped server's tool still gets its span.
  - 1.x: the instrumentor's `Server._handle_request` wrapper checks `is_observed_server(instance)`.
- FastMCP: every request of a wrapped FastMCP server runs with FastMCP's spans suppressed (4: its own switch; 3: `fastmcp.server.telemetry.get_tracer` is swapped, once, for one that returns a pass-through tracer while our contextvar is set and the original otherwise). Other FastMCP servers in the process keep their native spans.
- **`Laminar.use_span` attaches Laminar's isolated context as the global OTel context, dropping values set on the previous global context.** FastMCP 4's suppression key is such a value, so `McpCall.activate` re-enters the suppression inside `use_span`; without that, the handler's part of the request got a FastMCP span again.

## Trace ids, roots and the id generator

- Every span is started with an empty `Context()`: always a root, also when the server runs in the caller's context (`Client(server)` / `DirectDispatcher`). A client `traceparent` is always a span link; its trace id is used only to start a session's turn or when there is no session (below). Never fake a remote parent: the dangling parent id hides the span in the trace view.
- Trace id order: `turnId` hash (`sha256("{session or ''}\n{turnId}")[:16]`) → `TurnTracker` (when the request has a session; a NEW turn starts with the call's `traceparent` trace id, random without one, and later calls within the gap join it) → `traceparent` trace id (no session) → random. The tracker is updated on every call with a session, including turn-id calls (the turn-id trace becomes the session's current turn).
- **Why the tracker beats `traceparent`** (contract change after a live run): an OTel-instrumented mcp 2.x client sends a fresh `traceparent` per call (its own `MCP send ...` / CLIENT span, a new trace each time), so with `traceparent` first every call of a turn became its own trace. Now it only seeds a new turn (so the first call shares the client's trace); each span still links to its own `traceparent`.
- Forcing: `LaminarIdGenerator` (installed in `init_tracing` on Laminar's own `TracerProvider`) returns the id set by `forced_trace_id(...)` (a contextvar set only around `start_span`). OTel calls the generator only for spans without a valid parent, so it can't leak into children.
- **Degrade path**: `can_force_trace_id(provider)` is false when Laminar's provider was not built by `init_tracing` (the provider is created once per process; a `LaminarSpanProcessor` added to a user-owned provider without `Laminar.initialize` produces no wrapper spans at all, like every Laminar instrumentation). Then the span just gets a random trace id (one trace per call) and a debug log is written once; links still work.
- With the instrumentor on, a client in this process sends its CLIENT span as `traceparent`: without a session (in-memory transports, stateless HTTP) each call takes its client span's trace id; with one, the turn starts in the first call's client trace and the tracker keeps later calls in it. Tests that need no `traceparent` at all suppress OTel instrumentation around a `Client(server)` call (the JSON-RPC dispatcher would still inject mcp's own client span).
- `turn_gap_seconds` is clamped to 25 (Signals wait for 30 s of quiet). No session (stdio, 2026-07-28 stateless HTTP without `openai/session`) means no tracker.

## Requests and sessions

- Session id: 2.x `ctx.session._connection.session_id`, else the `mcp-session-id` header; 1.x the header of the HTTP request in `server.request_context.request`; then `_meta["openai/session"]`. Set as `mcp.session.id` and `lmnr.association.properties.session_id` (so child spans inherit it via `use_span`).
- Client name / version: `session.client_params.client_info` (2.x; 2026-07-28 requests synthesize it from the envelope) / `.clientInfo` (1.x). `initialize` itself is never traced.
- `network.transport` / `network.protocol.name` only when an HTTP request is visible (`tcp` / `http`); stdio vs in-memory can't be told apart, so it is omitted there.
- Protocol version: 2.x `ctx.protocol_version`; 1.x the `mcp-protocol-version` header, else what `ServerSession` negotiated from the client's `initialize`.

## The `context` field

- Injected into `tools/list` results (copies; dicts on 2.x, models / `ServerResult` on 1.x), never into `required`. Off with `context=False` or when payloads are off (including `LMNR_TRACE_CONTENT=false`).
- Stripped from `tools/call` arguments before the SDK validates them: 2.x by `dataclasses.replace(ctx, params=...)` in the middleware (runs before params validation); 1.x by `model_copy` of the request before the handler (where `validate_input` runs). FastMCP 3/4 reject unexpected arguments outright, so this is load-bearing.
- "Does the tool declare its own `context`?" is answered by the framework's registry when there is one (another server copy may have served `tools/list`): 2.x `Server.get_tool_input_schema` (set by `MCPServer` and FastMCP 4), mcp 1.x `FastMCP._tool_manager`, FastMCP 3 `await get_tool(name)`, 1.x low-level `_get_cached_tool_definition`. Otherwise what this process saw in `tools/list`; otherwise strip.
- The recorded `gen_ai.tool.call.arguments` are the client's arguments, `context` included.

## Errors

- `isError` results: `-32602` when the first text block reads like an unknown tool / validation failure (`_INVALID_PARAMS_TEXT`, matched against what mcp 1.x/2.x and FastMCP 3/4 actually return), else `tool_error`; no `rpc.response.status_code` (it was not a JSON-RPC error).
- Raised `McpError` / `MCPError`: the code, plus `rpc.response.status_code`. 2.x `ValidationError` (params validation inside the chain): `-32602`. Anything else: the class name. 1.x's low-level `call_tool` decorator folds every handler exception into `isError`, so there it's `tool_error`.
- A traced method with no handler is still a span: 2.x raises `MCPError(-32601)` inside the chain; on 1.x `ObservedHandlers.get` hands out a stub raising the same `McpError` `_handle_request` would have answered with.
- Without payloads, the `exception` event never carries the raised exception or its text (fixed messages: `Tool returned an error`, `JSON-RPC error {code}`, the class name).

## Tests

- `tests/test_instrumentations/test_mcp/test_wrap_mcp_server_v2.py` (mcp 2.x, FastMCP 4, streamable HTTP via `httpx2.ASGITransport`; the URL's host must include a port to pass the DNS-rebinding check, and `json_response=True` because ASGITransport buffers), `test_wrap_mcp_server_v1.py` (mcp 1.x overlay, as CI runs it), `test_wrap_mcp_server_units.py` (both majors). The FastMCP 3 test is skipped unless fastmcp 3 is installed: `uv run --with mcp==1.30.0 --with fastmcp==3.4.9 pytest tests/test_instrumentations/test_mcp -k wrap`.
- The dev dependency `fastmcp` is pinned to a 4.x (it needs mcp 2.x, matching the `mcp` pin). Under the 1.x overlay the 4.x package is still installed but unimportable; nothing imports it unless the server is a FastMCP one.

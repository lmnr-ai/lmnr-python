# lmnr-mock: the mock server protocol

- **Protocol version:** 1 (in every URL: `/_lmnr/v1/`)
- **Reference server:** `lmnr.evals.mock_server` (lmnr-python)
- **Conformance cases:** [`cases/`](./cases)

An `eval.py` runner (`lmnr.evals.EvalRunner`) declares mocks as Python
handlers. In every trial one mock server runs inside the agent container and
answers two kinds of callers:

- **Function mocks:** a Laminar SDK in the app under test asks the server
  before running an observed function whose span name is a mocked key, and
  returns or raises what the server says instead of running it.
- **HTTP mocks:** the app's own HTTP client, pointed at the server through a
  base URL environment variable.

This document is the contract between the server (lmnr-python) and the SDKs
(lmnr-python and lmnr-ts). It is copied byte-identically into lmnr-ts, like
`spec/lmnr-atif`; change both together.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## 1. Activation

The runner sets `LMNR_MOCK_URL` in the environment of every command it runs
in the container:

```
LMNR_MOCK_URL=http://127.0.0.1:<port>/_lmnr/v1/<token>
```

`<token>` is 32 random URL-safe characters (`[A-Za-z0-9_-]`), new for every
trial. Every protocol endpoint below is under that URL, so only a process that
got the variable can reach them.

An SDK MUST do nothing mock-related when `LMNR_MOCK_URL` is unset or empty.
An SDK MUST NOT log or record the variable's value (it holds the token).

## 2. Endpoints

All bodies are JSON (`Content-Type: application/json`) unless stated otherwise.
`{url}` is the value of `LMNR_MOCK_URL`.

### 2.1 `GET {url}/health`

`200 {"ok": true}` once the server serves.

### 2.2 `GET {url}/keys`

`200 {"keys": [<string>, ...]}`: every function mock key, sorted. HTTP mocks
are not listed.

An SDK SHOULD fetch the keys once, at initialization, and only send calls of
observed functions whose key is in the list. A function's key is its span
name (`observe(name=...)`, else the function's name), unless the SDK offers an
explicit mock key.

### 2.3 `POST {url}/fn/{key}`

`{key}` is the function key, percent-encoded as one path segment (`/` is
`%2F`, a space is `%20`).

Request body:

| Field | Type | |
|---|---|---|
| `args` | object or array | REQUIRED. The call's arguments, serialized as the SDK serializes span input: an object by parameter name (Python), an array of positional arguments (TypeScript). |
| `lang` | `"python"` or `"typescript"` | REQUIRED |
| `trace_id` | string or null | The current trace id (32 hex digits), if any |
| `span_id` | string or null | The id of the span the SDK records for this call (16 hex digits), if any |

Responses:

| Status | Body | The SDK MUST |
|---|---|---|
| 200 | `{"action": "respond", "value": <json>}` | return `value` instead of calling the function (`null` included) |
| 200 | `{"action": "raise", "error": {"type": <string>, "message": <string>}}` | raise: `MockedError` (Python) or an `Error` whose `name` is `type` (TypeScript), with `message` |
| 200 | `{"action": "passthrough"}` | call the real function |
| 404 | `{"error": {"type": "UnknownMockKey", "message": <string>}}` | fail (raise) |
| 500 | `{"action": "handler_error", "error": {"type": <string>, "message": <string>}}` | fail (raise), with the handler's error type and message |
| 400 | `{"error": {"type": "BadRequest", "message": <string>}}` | fail (raise) |

Any other status, a body that isn't one of the above, and a transport error
(connection refused, timeout) are failures too: the SDK raises. It MUST NOT
fall back to calling the real function on failure: a mocked call must never
reach the real dependency by accident.

A failure SHOULD say which key failed and why (status and error message).

### 2.4 Control endpoints (the runner)

- `GET {url}/calls` → `200`, `application/x-ndjson`: the call log, one JSON
  object per line (§4).
- `GET {url}/log` → `200`, `text/plain`: the server's own log (startup,
  one line per call, handler tracebacks).

There is no endpoint that lists handlers, fixtures or configuration.

### 2.5 Anything else

A request whose token doesn't match, an unknown path or a method an endpoint
doesn't take gets `404` with the body `Not Found` (`text/plain`), like a
plain web server.

## 3. HTTP mocks

An HTTP mock is declared with an API's base URL as key
(`"https://api.zendesk.com"`, optionally with a path:
`"https://api.example.com/v2"`). The runner sets the mock's `base_url_env`
variable to:

```
http://127.0.0.1:<port>/h/<host>[:<port>]<path of the key>
```

e.g. `http://127.0.0.1:41234/h/api.zendesk.com`. The app uses it as its API
client's base URL. These routes take any method and carry **no token**.

A request to `/h/<host>[:<port>]/<path>?<query>` goes to the handler of that
host (case-insensitive). The handler sees `path` = `/<path>` (everything after
the host, so including the key's own path) and the parsed query.

| Handler does | Response |
|---|---|
| returns `mock.Response(status, json=/text=/body=, headers=)` | sent as given (`Content-Type` defaults to the body kind) |
| returns any other JSON value | `200`, that value as JSON |
| returns `None`, returns `mock.PASSTHROUGH`, or raises `mock.Unmatched` | `501 {"error": {"type": "Unmatched", "message": ...}}`, logged as `unmatched` |
| raises `mock.Error(type, message)` | `500 {"error": {"type", "message"}}`, logged as `raise` |
| raises anything else | `500 {"error": {"type": <exception class>, "message"}}`, logged as `handler_error` |

A host with no HTTP mock gets `404 Not Found` and is not in the call log.

## 4. Call log

One object per call, in the order the server handled them:

```json
{"seq": 3, "ts": "2026-11-26T10:00:00.123Z", "type": "function", "key": "payments.refund", "index": 1,
 "request": {"args": {"charge_id": "ch_1"}, "lang": "python"},
 "outcome": "raise", "response": {"error": {"type": "DuplicateRefund", "message": "charge already refunded"}},
 "duration_ms": 0.4, "trace_id": "...", "span_id": "..."}
```

| Field | |
|---|---|
| `seq` | 0-based, across all keys |
| `ts` | when the server received the call, UTC, milliseconds |
| `type` | `function` or `http` |
| `key` | the mock's key as declared (`"https://api.zendesk.com"` for HTTP mocks) |
| `index` | 0-based, per key (`call.index` / `req.index` the handler saw); null for an unknown function key |
| `request` | function: `{"args", "lang"}`; HTTP: `{"method", "path", "query", "headers", "body_text" or "body_b64"}` (headers lower-cased) |
| `outcome` | `respond`, `raise`, `passthrough`, `unmatched` or `handler_error` |
| `response` | function: `{"value"}` (respond), `{"error"}` (raise, handler_error, unmatched), `{}` (passthrough); HTTP: `{"status", "headers", "body_text" or "body_b64"}` |
| `duration_ms` | time in the server, handler included |
| `trace_id`, `span_id` | function: from the request; HTTP: from a W3C `traceparent` header; else null |

Bodies are `body_text` when they are valid UTF-8, `body_b64` (standard
base64) otherwise. An unknown function key is logged with outcome `unmatched`.

Handlers run one at a time (one lock), so `index`, earlier `calls` and the
per-trial `state` are deterministic for a given order of arrival.

## 5. Conformance cases

[`cases/`](./cases) holds one JSON file per scenario, run against a fresh
server whose mocks are `cases/mocks.json` (key → handler name) with the
handlers in `cases/lmnr_mock_reference.py`:

```json
{"description": "...",
 "calls": [{"method": "POST", "path": "/fn/crm.get_customer",
            "body": {"args": {"email": "ana@example.com"}, "lang": "python"},
            "status": 200, "response": {"action": "respond", "value": {...}},
            "sdk": {"returns": {...}}}]}
```

`path` is relative to `LMNR_MOCK_URL`. Each call's `sdk` says what an SDK must
do with that response:

- `{"returns": <json>}`: return the value;
- `{"raises": {"type", "message"}}`: raise the mocked error;
- `{"calls_real_function": true}`: run the function itself;
- `{"fails": true}`: raise a failure (not the mocked error, not the real call).

The server's own test (`tests/test_evals_mocks.py` in lmnr-python) checks
`status` and `response`; an SDK's test serves the recorded responses (or runs
the reference server) and checks `sdk`.

## Changelog

- **1** (2026-11-27): first version.

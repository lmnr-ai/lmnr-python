# LAM-1715 — Debugger Cache v2: Python SDK Component Plan

> **Status:** design / not yet implemented.
> **Read `00-shared-spec.md` first.** This file only describes how the Python SDK
> (`lmnr`) fulfils the shared contract. Anything about the wire format, the three
> outcomes, the input hash, or what app-server does lives in the shared spec — do
> not re-derive it here.
> Branch: `feat/lam-1715-debugger-cache-v2` (off
> `refactor/lam-1672-sdk-debugger-rework`).

This plan is the Python mirror of `lmnr-ts/plans/lam-1715/02-ts-sdk.md`. Where TS
and Python differ, it is called out explicitly; otherwise the two SDKs implement
the same contract and should stay line-comparable on the parity surface
(`config`, `hash`, `pointer`, test vectors).

---

## 0. Orientation: what the Python SDK does today (v1)

The v1 replay path is entirely in-process and **synchronous** (unlike TS, which
fills the cache asynchronously after `initialize()` returns):

- `src/lmnr/sdk/debug/__init__.py` — `DebugRuntime`, builds an in-memory
  `ReplayCache` **synchronously** inside `init_debug_runtime` (no async loading
  window), holds per-path occurrence counters (`_counters`), exposes
  `get_cached(span_path)`.
- `src/lmnr/sdk/debug/replay_cache.py` — the in-memory cache (`ReplayCache`,
  `payloads[:cache_until]`, `get_cached(path, occurrence)`).
- `src/lmnr/sdk/debug/source_trace.py` — two-phase ClickHouse fetch over
  `LaminarClient` (`fetch_spine_metadata`, `fetch_spine_payloads`).
- `src/lmnr/sdk/debug/spine.py` — spine detection, `has_overlap`,
  `resolve_cache_until_span_id`.
- `src/lmnr/sdk/debug/config.py` — `build_debug_config`, `_parse_cache_until`
  (count OR span id), `_load_last_run`, the truthy set.
- `src/lmnr/sdk/debug/replay.py` — generic helpers `replay_enabled`,
  `span_path_from_span`, `cached_payload_for`, `mark_span_cached`.
- `src/lmnr/sdk/debug/pointer.py` — run pointer (console line +
  `.lmnr/last-run.json`).
- The replay **consumers** are the four per-provider rollout wrappers:
  `src/lmnr/opentelemetry_lib/opentelemetry/instrumentation/{anthropic,openai,
  google_genai,litellm}/rollout.py`. Each calls `span_path_from_span(span)` →
  `cached_payload_for(span_path)` → `cached_response_to_<provider>(cached)` →
  `mark_span_cached(span)` and synthesizes a provider-native (streaming) response.

v2 removes the spine/occurrence/in-memory-cache machinery and replaces the
per-call lookup with one HTTP round-trip to app-server keyed by an **input hash**.

---

## 1. Removed / kept / added (file-level)

### Removed
- `src/lmnr/sdk/debug/replay_cache.py` — delete. No in-process cache anymore.
- `src/lmnr/sdk/debug/source_trace.py` — delete. The SDK no longer fetches the
  source trace; app-server warms the cache from ClickHouse (shared spec §6.3).
- `src/lmnr/sdk/debug/spine.py` — delete. No spine detection, no `has_overlap`,
  no `resolve_cache_until_span_id`.
- The spine/occurrence test vectors under `tests/data/debug/` — delete the
  spine-specific ones; keep / add the **hash parity** vector (§7).
- From `src/lmnr/sdk/debug/__init__.py`: `ReplayCache` import/use, `_build_cache`,
  the `_counters` dict, `get_cached`, and the `replay_active` plumbing that exists
  only to model the (TS) async-loading window. Because Python builds
  synchronously, `replay_configured` collapses to "replay is configured" with no
  `cache is None`-but-active state to represent. `init_debug_runtime` stops
  building a cache entirely — it only parses config, registers the session, and
  constructs the (now cache-less) `DebugRuntime`.
- From `src/lmnr/sdk/debug/config.py`: the **count form** of `cache_until`.
  `cache_until` becomes span-id-only (shared spec §3, §4). Remove
  `_parse_cache_until`'s integer branch, the `cache_until: int` field, and the
  numeric-wins precedence. `DebugConfig.replay_enabled` becomes
  `replay_trace_id is not None and cache_until_span_id is not None`.
- From `src/lmnr/sdk/debug/replay.py`: `cached_payload_for` (occurrence-counter
  advance) — replaced by the per-call endpoint lookup. Keep `mark_span_cached`
  and `span_path_from_span` (still useful for logging / span marking).

### Kept (do NOT change)
- `src/lmnr/sdk/debug/pointer.py` — run pointer mechanism unchanged (shared spec
  §3 "Preserved"). `rollout.session_id` metadata key stays.
- The session registration call (`RolloutSessions.register` on both clients) and
  the debugger-URL construction (`DebugRuntime.debugger_session_url`,
  `record_project_id`, `record_trace_id`).
- `mark_span_cached` (`src/lmnr/sdk/debug/replay.py`) — still stamps
  `lmnr.span.type=CACHED` / `lmnr.span.original_type=LLM` on a served span.
- Each provider's `cached_response_to_<provider>` reconstruction + cached
  streaming synthesis (`_create_cached_stream` etc.) — the SOURCE of the cached
  payload changes (endpoint, not in-memory cache), but the reconstruction is
  untouched. Notably the OpenAI/LiteLLM `created=int(start_time/1e9)` derivation
  from the cached span's `start_time` nanoseconds stays (see §3).

### Added
- `src/lmnr/sdk/debug/hash.py` — the canonical input hash (shared spec §5):
  `canonical_json` + `blake3` + system-message exclusion. Cross-language parity
  with the TS `hash.ts` and app-server's `debug_input_hash`; shared test
  vector (§7). **No AI SDK reshape module** — Python provider messages already
  match the stored shape (shared spec §9; the reshape is TS-only).
- A `cache` method on `RolloutSessions` for **both** the sync
  (`.../synchronous/resources/rollout_sessions.py`) and async
  (`.../asynchronous/resources/rollout_sessions.py`) clients — POSTs to
  `/v1/rollouts/{session_id}/cache`, returns the discriminated HIT/MISS/COLD
  outcome (§5).
- A process-wide **"run live" static flag** on `Laminar` (shared spec §7.3), set
  on first MISS, reset in `Laminar.shutdown()`.
- Rewritten per-provider `wrap_create` flow in each `rollout.py`: compute input
  hash → call the cache endpoint → HIT serves cached / MISS sets flag + live /
  COLD-degraded runs live this call (§3). The shared decision logic is hoisted
  into `replay.py` so the four providers stay in lockstep (§3).

---

## 2. Config changes (`src/lmnr/sdk/debug/config.py`)

- `DebugConfig` loses `cache_until: int`; keeps `cache_until_span_id: str | None`
  (the suffix-match needle, already normalized to hyphen-stripped lowercase hex
  by `_normalize_span_id`, which is **kept unchanged**).
- `DebugConfig.replay_enabled` ⇒ `replay_trace_id is not None and
  cache_until_span_id is not None` (shared spec §4: both env vars must resolve
  non-empty).
- `LMNR_DEBUG_CACHE_UNTIL` is parsed **only** as a span-id needle — `_parse_cache_until`
  drops the `int(value)` branch and the numeric-wins precedence, returning just
  the needle (or None + warn). The four needle forms (full UUID / last-two-groups
  / raw 16-hex / short suffix) are still accepted via `_normalize_span_id`; they
  are sent verbatim to app-server, which does the suffix match against the source
  trace's span ids (shared spec §6.2). The SDK no longer resolves the needle
  locally (no spine to resolve against).
- `_load_last_run` / `LMNR_DEBUG_FROM_LAST_RUN` unchanged **except** the pointer
  no longer persists a resolved integer `cache_until` — it persists the span-id
  needle as-is (there is no resolution step). Keep the `cache_until` field name
  in `last-run.json` for back-compat; its value is now the span-id string.

> Parity: keep `config.py` line-comparable with TS `config.ts`. The truthy set
> (`{"true","1","yes","on"}`), `_HEX_RE`, the needle forms, pointer field order,
> and console prefix stay byte-identical across the two SDKs.

---

## 3. The replay consumers: the four `rollout.py` wrappers

Today each provider's `wrap_create` independently calls
`span_path_from_span` → `cached_payload_for` → reconstruct. In v2 the **decision
logic is identical across all four providers** (compute hash, hit the endpoint,
branch on outcome), so hoist it into a shared helper in `replay.py` and have each
provider supply only its provider-specific message extraction + reconstruction.

New shared helper (in `src/lmnr/sdk/debug/replay.py`), conceptually:

```
def cache_outcome_for(span, input_messages) -> CacheOutcome | None:
    if not replay_enabled():            return None        # unchanged gate
    if Laminar.debug_run_live:          return LIVE        # MISS flag latched
    input_hash = debug_input_hash(input_messages)          # §4, debug/hash.py
    runtime = get_runtime()
    outcome = runtime.client.rollout_sessions.cache(        # §5
        session_id=runtime.session_id,
        replay_trace_id=runtime.replay_trace_id,
        cache_until=runtime.cache_until_span_id,
        input_hash=input_hash,
    )
    if outcome.kind == "miss":  Laminar.debug_run_live = True
    return outcome
```

Each provider's `wrap_create` then becomes:

```
input_messages = self._extract_input_messages(kwargs)      # provider-specific
outcome = cache_outcome_for(span, input_messages)
if outcome is not None and outcome.kind == "hit":
    response = self.cached_response_to_<provider>(outcome.cached)
    if response is not None:
        mark_span_cached(span)
        return <stream/coroutine/plain wrap as today>
return wrapped(*args, **kwargs)                            # MISS / LIVE / no-hit
```

Key points:
- `replay_enabled()` keeps gating on `get_runtime()?.replay_configured` — but
  `replay_configured` is now just "replay is configured" (trace id + cache_until
  needle present). Python already builds synchronously, so there was never an
  async cache-load window to wait on; the getter simply stops modelling one.
- `Laminar.debug_run_live` (process-wide static / class attr) short-circuits
  BEFORE the network call once any call in this process has seen MISS. One
  redundant MISS call per distributed worker is the accepted v1 limitation
  (shared spec §7.2, §11).
- COLD is invisible to the SDK as a distinct branch in the happy path: app-server
  **blocks and warms**, then returns HIT or MISS. The only COLD the SDK ever sees
  is `"live"` — the warmup-timeout degrade (shared spec §7.2, §7 CRITICAL). On
  `"live"` we run live for THIS call only and do **not** set the static flag
  (next call retries the endpoint; the cache may be warm by then).
- The provider reconstruction (`cached_response_to_anthropic` /
  `_to_openai` / `_to_google` / `_to_litellm` and the cached-stream synthesizers)
  is **kept as-is** — only the source of the cached dict changes. In particular,
  the OpenAI/LiteLLM `created=int(cached_span.get("start_time", 0) / 1e9)`
  derivation (`openai/rollout.py:119`) still reads `start_time` from the cached
  payload, so the HIT body must carry `start_time` (nanoseconds) — see §5.
- The async providers (async OpenAI/Anthropic/Gemini paths) need the cache call
  awaitable: the async client's `RolloutSessions.cache` is the async variant;
  the shared helper has a sync and an async form (`cache_outcome_for` /
  `acache_outcome_for`) mirroring how the providers already split sync/async.
- **Disabled-tracing guard (Python-specific):** `span_path_from_span` reads
  `lmnr.span.path` off the live span's attributes. During replay the span path
  must remain resolvable even when user-level tracing is otherwise suppressed —
  preserve the existing behavior that the processor still stamps the span path on
  rollout spans so replay isn't masked. Do not let the v2 rewrite drop this.
- `mark_span_cached` marks the **live** span being created for this call
  (`span` passed into `wrap_create`), exactly as today.

---

## 4. Input hashing (§5) — `src/lmnr/sdk/debug/hash.py`

One new module on the cross-language parity surface (Python needs **no** AI SDK
reshape — shared spec §9):

`debug_input_hash(messages) -> str` reproduces app-server's `canonical_json` +
`blake3` + `extract_system_message`:
- `canonical_json(value)`: objects → keys sorted lexicographically (recursive);
  arrays → order preserved; scalars → JSON-equivalent encoding. Hash the whole
  non-system message array as ONE blob (shared spec §5.1). Match Python's
  `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
  to the exact byte sequence app-server's `serde_json` canonical form produces —
  pin with the shared vector (§7); do not assume the two encoders agree on
  non-ASCII / escaping without the vector proving it.
- System exclusion: strip the first `role == "system"` entry before hashing
  (handle string / list / parts content shapes), mirroring
  `extract_system_message` (`app-server/src/traces/prompt_hash.rs`)
  (shared spec §5.2).
- `blake3` over the canonical-json UTF-8 bytes → hex. Use the `blake3` PyPI
  package (add as a dependency if not already present; pin the version).
- **No number canonicalization** (shared spec §5.1 deferred). Document the
  limitation in a one-line comment, identically worded to `hash.ts`.

> The messages fed to `debug_input_hash` are each provider's already-stored input
> shape (what the instrumentation writes to `gen_ai.input.messages` /
> `gen_ai.prompt`). Python providers do NOT reshape — unlike the TS AI SDK
> wrapper, which must port `input_chat_messages_from_json` (shared spec §9).
> Action item: confirm per-provider that the SDK-side message list the wrapper
> sees matches what app-server stored before hashing; pin each with the vector.

---

## 5. Client resource: `RolloutSessions.cache` (sync + async)

Add a `cache` method to BOTH
`src/lmnr/sdk/client/synchronous/resources/rollout_sessions.py` and
`src/lmnr/sdk/client/asynchronous/resources/rollout_sessions.py`, alongside the
existing `register` / `delete`:

```
def cache(self, session_id, replay_trace_id, cache_until, input_hash) -> CacheOutcome
```

- `POST /v1/rollouts/{session_id}/cache`, body `{"replayTraceId":…,
  "cacheUntil":…, "inputHash":…}`, `self._headers()` (ProjectApiKey) — same
  auth/shape as the sibling `register` method (shared spec §7.1).
- Parse app-server's discriminated response (the app-server plan §1 defines
  `Hit{response}/Miss{}/Live{}`) into a small `CacheOutcome` (a dataclass or
  tagged dict: `kind ∈ {"hit","miss","live"}`, plus `cached` on hit). Shape the
  `Hit` body so the provider `cached_response_to_*` reconstruction reads it
  unchanged — it must carry `attributes` (incl. `lmnr.sdk.raw.response` /
  `gen_ai.output.messages` / `gen_ai.response.finish_reason`) AND the
  `start_time` nanoseconds that OpenAI/LiteLLM derive `created` from (§3).
- Error posture: on a non-OK response, log + degrade to `kind="live"` for this
  call (best-effort, never crash the user's program). Do NOT latch the static
  flag on a transport error — only a real MISS latches it. Keep this parallel to
  how `register` currently surfaces errors, but for `cache` swallow-and-degrade
  rather than raise (a replay miss must never take down the user's run).
- The async resource's `cache` is `async def` and `await`s the httpx async
  client; the sync resource's is blocking. Both return the same `CacheOutcome`.

---

## 6. Static "run live" flag on `Laminar`

- Add a process-wide flag on `Laminar` (`src/lmnr/sdk/laminar.py`) — a private
  class attribute `__debug_run_live = False` with a classmethod getter/setter (or
  a module-level flag in `debug/replay.py` if threading `Laminar` into the
  providers is awkward; prefer keeping it on `Laminar` to mirror TS
  `Laminar.debugRunLive`).
- Set `True` on the first MISS (in the shared `cache_outcome_for` helper, §3).
- Reset to `False` in `Laminar.shutdown()` right next to the existing
  `reset_debug_runtime()` call (`laminar.py:1377`) so a later `initialize()`
  starts clean (shared spec §7.3). The pointer-emit + exit-hook unregister logic
  there is unchanged.

---

## 7. Tests & parity vectors

- **Hash parity vector** (`tests/data/debug/input_hash_cases.json`): a set of
  `{messages, expected_hash}` rows, **byte-identical** to the TS copy
  (`lmnr-ts/.../test/data/debug/`) and asserted against app-server's
  `debug_input_hash` (the app-server plan ships the same vector). This is the
  single most important test — it proves SDK ⟷ server hash agreement across all
  three languages (shared spec §10 first checklist item).
- Unit tests for the three outcomes through the shared `cache_outcome_for` helper
  with a faked `rollout_sessions.cache` (HIT serves cached + marks span; MISS
  sets the static flag and the NEXT call skips the endpoint; LIVE runs live
  without setting the flag). Exercise at least one sync and one async provider so
  the sync/async split is covered.
- Per-provider smoke test that a HIT body round-trips through
  `cached_response_to_<provider>` and still yields the right `created`
  (OpenAI/LiteLLM `start_time` → seconds) and zeroed usage tokens.
- Delete the spine / occurrence / overlap tests.
- Run `pytest`, `ruff`/lint, and the existing instrumentation tests before
  commit (see repo CLAUDE.md). Keep `config.py` / `hash.py` line-comparable with
  the TS counterparts.

---

## 8. Open questions / action items
1. Confirm, per provider, that the message list the wrapper hashes equals what
   app-server stored before hashing (no hidden reshape on the Python side).
   Pin each with the shared hash vector.
2. Confirm `blake3` is acceptable as a runtime dependency; pin the version and
   verify wheels exist for the SDK's supported Python versions / platforms.
3. Confirm the HIT body carries everything the four `cached_response_to_*`
   reconstructions read — especially `attributes` and the OpenAI/LiteLLM
   `start_time` nanoseconds.
4. Decide where `Laminar.debug_run_live` lives (class attr vs module flag) and
   make sure both sync and async provider paths can read/set it without a
   circular import (`replay.py` ↔ `laminar.py`).
5. Confirm the async client's `cache` is reachable from the async provider
   wrappers the same way the sync one is from the sync wrappers.

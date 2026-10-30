# lmnr-atif: Laminar's extension to ATIF

- **Spec version:** 0.1.0
- **Base format:** [ATIF](https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md) v1.7 or later. Writers emit `ATIF-v1.8`.
- **Machine-readable schema:** [`lmnr-atif.schema.json`](./lmnr-atif.schema.json)
- **Examples / conformance fixtures:** [`fixtures/`](./fixtures)

ATIF (Agent Trajectory Interchange Format) is Harbor's format for recording an
agent run as a list of steps. It is designed for comparing and evaluating
trajectories, not for observability, so it can't express some of what a Laminar
trace knows: per-call system prompts and tool sets, exact call timings, tool
durations, errors, and links back to the trace.

This document defines the fields Laminar writes into ATIF's own `extra` objects
to fill those gaps. A file that follows it is still plain, valid ATIF: a reader
that knows nothing about Laminar sees an ordinary trajectory, and a Laminar-aware
reader can rebuild each LLM call's input and the spans behind it.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## 1. Namespace

Every field defined here lives under an `lmnr` key inside an ATIF `extra`
object, e.g. `step.extra.lmnr.start_time`. Laminar never writes ATIF core
fields with non-standard meanings, and never writes keys outside `extra.lmnr`.

Field paths in this document start from the ATIF object that holds them:

| Prefix | Object |
|---|---|
| `trajectory` | A trajectory: the root document, or an entry of `trajectory.subagent_trajectories` |
| `step` | An entry of `trajectory.steps` |
| `tool_call` | An entry of `step.tool_calls` |
| `observation_result` | An entry of `step.observation.results` |
| `metrics` | `step.metrics` |
| `final_metrics` | `trajectory.final_metrics` |
| `subagent_trajectory_ref` | An entry of `observation_result.subagent_trajectory_ref` |

`steps[k]` is `trajectory.steps[k]`. An "`agent` step" (or `user`, `system`
step) is a step whose `step.source` has that value.

ATIF has an `extra` object on: `trajectory`, `trajectory.agent`, `step`,
`tool_call`, `observation_result`, `metrics`, `final_metrics` and
`subagent_trajectory_ref`. `step.observation` itself has no `extra` (it forbids unknown keys),
which is why tool outcomes live on each observation result.

## 2. Versioning

`trajectory.extra.lmnr.spec_version` is REQUIRED on every trajectory that
carries Laminar fields, including each embedded subagent trajectory, since ATIF
defines those as complete documents. A reader treats an embedded subagent
without `trajectory.extra.lmnr.spec_version` as having its parent's version.

The version is a [semver](https://semver.org) string:

- While the major version is 0, a MINOR bump is a breaking change and a PATCH
  bump only adds optional fields or clarifies the text.
- From 1.0, standard semver applies: MAJOR is breaking, MINOR is additive,
  PATCH is a clarification.

Two versions are **compatible** when they share the same major version and,
while the major is 0, the same minor version. A reader:

- MUST read a file with a compatible version, ignoring keys it doesn't know.
- MUST treat a file with an incompatible, missing or unparseable version as
  plain ATIF, ignoring every `extra.lmnr` in it (a reader SHOULD log this once).

A writer MUST emit exactly the version it implements.

The spec version is independent of ATIF's `trajectory.schema_version`. Fields that live in
`tool_call.extra`, `observation_result.extra` and the `step.llm_call_count` they
depend on were added in ATIF-v1.7, so this spec requires at least that.

### Keeping implementations in sync

This spec, the JSON schema and the fixtures are copied verbatim into every
Laminar SDK that reads or writes the format (Python first, then TypeScript).
Each copy carries the same spec version, and each repository has a test that
checks its constant, its schema and its fixtures agree. Change the spec here,
bump the version, add a changelog entry, then port.

## 3. Common types

| Type | Definition |
|---|---|
| `Timestamp` | ISO 8601 string in UTC with a `Z` suffix and microsecond precision, e.g. `2026-10-28T12:00:01.900000Z` |
| `LmnrId` | A Laminar trace or span id as a lowercase UUID string, as shown in the Laminar UI and API (a span id is its 8 OTel bytes, left-padded with zeros to 16) |
| `Status` | `"ok"` or `"error"` |
| `Error` | `{"type": string, "message": string, "stacktrace"?: string}`, from the span's `exception` event |
| `Ref` | `{"$ref": string}`, standing in for an override value; see §6.6 |
| `ToolDefinition` | One tool in OpenAI's function-calling shape, as in ATIF's `trajectory.agent.tool_definitions`: `{"type": "function", "function": {"name", "description"?, "parameters"?}}`. Non-function provider tools keep their own shape. |
| `ContentPart` | ATIF's content part (`text`, `image`, `audio`) |
| `LmnrMessage` | One chat message, see below |

`LmnrMessage` is only used by `step.extra.lmnr.input_messages` (§6.4):

```jsonc
{
  "role": "system" | "developer" | "user" | "assistant" | "tool",
  "content": "string" | [ContentPart, ...],   // REQUIRED, may be ""
  "reasoning_content": "string",              // assistant only
  "tool_calls": [ToolCall, ...],              // assistant only, ATIF ToolCall shape
  "tool_call_id": "string"                    // tool only
}
```

## 4. Field reference

All fields are OPTIONAL unless marked REQUIRED.

### 4.1 `trajectory.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `spec_version` | string | REQUIRED. This spec's version, e.g. `"0.1.0"` |
| `producer` | `{name: string, version: string}` | REQUIRED on the root trajectory. The writer, e.g. `{"name": "lmnr-python", "version": "0.8.4"}` |
| `trace_ids` | `LmnrId[]` | Every Laminar trace that contributed steps. Usually exactly one |
| `root_span_id` | `LmnrId` | The root span, when there is exactly one trace |
| `project_id` | string | The Laminar project, when the writer knows it |
| `exported` | boolean | `true` when the spans behind this file were also exported to Laminar, so a reader that turns the file back into spans can avoid recording them twice |
| `complete` | boolean | `false` when the file is a snapshot written while spans were still open (e.g. the agent was killed on a timeout). Absent means complete |
| `agent_span_id` | `LmnrId` | Subagent trajectories only: the span that defined the subagent |
| `implicit_subagent` | boolean | Subagent trajectories only: `true` when the subagent was inferred from LLM calls inside a tool rather than from an agent marker (§6.7) |
| `defs` | object | Deduplicated override values (§6.6): `{"system_prompts"?: {key: string \| ContentPart[]}, "tool_definitions"?: {key: ToolDefinition[]}, "input_messages"?: {key: LmnrMessage[]}}` |

### 4.2 `step.extra.lmnr` on an agent step with an LLM call (`source: "agent"`, `llm_call_count: 1`)

| Key | Type | Meaning |
|---|---|---|
| `span_id` | `LmnrId` | The LLM span |
| `trace_id` | `LmnrId` | The LLM span's trace. Only written when `trajectory.extra.lmnr.trace_ids` has more than one entry |
| `start_time` | `Timestamp` | When the call started |
| `end_time` | `Timestamp` | When the call ended. Equal to `step.timestamp` (§6.2) |
| `provider` | string | The LLM provider, e.g. `"anthropic"`, `"openai"` |
| `request` | object | Request parameters other than messages and tools, keyed by the suffix of their OTel `gen_ai.request.*` attribute: `{"temperature": 0, "max_tokens": 4096, "thinking_type": "enabled", ...}`. Reasoning effort goes in ATIF's own `step.reasoning_effort` instead |
| `finish_reason` | string | Why the model stopped, as the provider reported it |
| `response_id` | string | The provider's response id |
| `status` | `Status` | Outcome of the call. Absent means `"ok"` |
| `error` | `Error` | Set when `step.extra.lmnr.status` is `"error"` |
| `system_prompt` | string \| `ContentPart[]` \| `Ref` | Override: the exact system prompt of this call (§6.3). `""` means this call had no system prompt |
| `tool_definitions` | `ToolDefinition[]` \| `Ref` | Override: the complete set of tools offered to this call (§6.5). `[]` means no tools |
| `input_messages` | `LmnrMessage[]` \| `Ref` | Override: the exact conversation this call received, excluding the system prompt (§6.4) |

### 4.3 `step.extra.lmnr` on an agent step with no LLM call (`source: "agent"`, `llm_call_count: 0`)

| Key | Type | Meaning |
|---|---|---|
| `synthetic` | `true` | The step was made up by the writer to carry tool calls (or a subagent) that no LLM call issued (§6.5, §6.7) |
| `span_id` | `LmnrId` | The span the step stands for (the tool's or the subagent's) |
| `trace_id` | `LmnrId` | As in §4.2 |
| `start_time`, `end_time` | `Timestamp` | The span's bounds. `step.timestamp` is `step.extra.lmnr.end_time` |

### 4.4 `step.extra.lmnr` on a `system` or `user` step

| Key | Type | Meaning |
|---|---|---|
| `source_span_id` | `LmnrId` | The first LLM span whose input contained this message |
| `role` | string | The message's original role, when ATIF can't express it. E.g. OpenAI's `"developer"` role is written as a `system` step with `step.extra.lmnr.role: "developer"` |

### 4.5 `tool_call.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `span_id` | `LmnrId` | The span that executed the tool |
| `start_time`, `end_time` | `Timestamp` | When the tool ran |
| `raw_arguments` | string | The arguments exactly as the model produced them, when they were not a JSON object. `tool_call.arguments` is then `{}` (ATIF requires an object) |
| `synthetic_id` | `true` | The writer generated `tool_call.tool_call_id` because the source had none |

### 4.6 `observation_result.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `status` | `Status` | Outcome of the tool call. Absent means unknown, see §7 |
| `error` | `Error` | Set when `observation_result.extra.lmnr.status` is `"error"` |

### 4.7 `metrics.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `reasoning_tokens` | integer | Subset of `metrics.completion_tokens` spent on reasoning |
| `cache_creation_tokens` | integer | Subset of `metrics.prompt_tokens` written to the provider's prompt cache |
| `input_cost_usd` | number | Cost of the input tokens. `metrics.cost_usd` stays the total |
| `output_cost_usd` | number | Cost of the output tokens |

ATIF's `metrics.prompt_tokens` is the full input including cached tokens, so writers
MUST add cache reads and cache writes into it when a provider reports them
separately (Anthropic does).

### 4.8 `final_metrics.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `duration_ms` | number | From the first step's start to the last step's end |
| `llm_calls` | integer | Number of LLM calls, including subagents |
| `tool_calls` | integer | Number of tool calls, including subagents |
| `errors` | integer | Number of failed LLM and tool calls, including subagents |

### 4.9 `subagent_trajectory_ref.extra.lmnr`

| Key | Type | Meaning |
|---|---|---|
| `span_id` | `LmnrId` | The root span of the subagent |

## 5. Identity fields

Writers fill these ATIF core fields as follows:

- `trajectory.session_id`: the Laminar session id if the trace has one, otherwise the trace id.
- `trajectory.trajectory_id`: on embedded subagents, `"lmnr-" + <subagent span id>`. On the root, the trace id when there is one trace.
- `trajectory.agent.name`: the root span's name. On subagents, the subagent's name.
- `trajectory.agent.version`: the release (git SHA or deploy id) the SDK detected, or `"unknown"`.
- `trajectory.agent.model_name`: the model used by the most calls, as `provider/model`.

## 6. Semantics

### 6.1 Steps

- Each LLM call becomes one `agent` step with `step.llm_call_count: 1`. Its
  output text goes in `step.message` (`""` when it only called tools), its
  reasoning in `step.reasoning_content`, its tool calls in `step.tool_calls`.
- Agent steps are ordered by their call's start time.
- A message that appears in an LLM call's input for the first time becomes a
  `user` or `system` step placed just before that call's agent step. Messages
  keep their original order.
- Tool results never become steps of their own. They go in the
  `step.observation` of the agent step whose tool call produced them, matched
  by `observation_result.source_call_id`.
- Text injected into a user message (e.g. Claude Code's `<system-reminder>`
  blocks) stays inside that `user` step's `step.message`. Writers MUST NOT split it
  into separate steps.
- `step.step_id` runs from 1 without gaps, as ATIF requires.

### 6.2 Timing

- An `agent` step's `step.timestamp` is the **end** of its LLM call (or of its
  span, for synthetic steps). This matches how Harbor's own agents record
  steps. The exact bounds are `step.extra.lmnr.start_time` and
  `step.extra.lmnr.end_time`.
- A `user` or `system` step's `step.timestamp` is the start of the LLM call that
  first received it.
- A tool call's execution bounds are `tool_call.extra.lmnr.start_time` and
  `tool_call.extra.lmnr.end_time`. ATIF has no core field for them.

### 6.3 System prompt

The system prompt is what an LLM call was configured with. A system-role
message in the middle of a conversation is something else: an ordinary message.

The **leading system steps** of a trajectory are the `system` steps before its
first `user` or `agent` step. Each is one system message; a text-only consumer
joins them with `"\n\n"`.

The system prompt of agent step *k* is:

1. `steps[k].extra.lmnr.system_prompt`, if present (after resolving a `Ref`).
   `""` means no system prompt.
2. Otherwise, the leading system steps.
3. Otherwise, none.

Overrides are not sticky: they apply only to the step that carries them.
Writers emit an override only when a call's system prompt differs from the
leading system steps.

A `system` step after the first `user` or `agent` step is an in-history
message (a compaction note, injected context, a mid-conversation system
message). It never changes the system prompt.

A provider request has no such distinction: every system message before the
first other message looks like part of the prompt. When a call's leading system
messages start with the trajectory's leading system steps, writers treat the
remaining ones as in-history messages rather than as a different system prompt.

### 6.4 Input of an LLM call

The input of agent step *k* is its system prompt (§6.3), followed by:

1. `steps[k].extra.lmnr.input_messages`, if present (after resolving a `Ref`).
2. Otherwise, if an earlier agent step *j* has
   `steps[j].extra.lmnr.input_messages`, take the latest such *j*:
   `steps[j].extra.lmnr.input_messages`, then the messages of steps *j* through
   *k − 1*. An override starts a new base, so the calls after a
   compaction need no override of their own.
3. Otherwise, the messages of every step before *k*, in order, skipping the
   leading system steps.

A step contributes its messages as follows:
- `system` → one `system` message (or the role in `step.extra.lmnr.role`, if set).
- `user` → one `user` message.
- `agent` → one `assistant` message with its `step.message`,
  `step.reasoning_content` and `step.tool_calls`, then one `tool` message per
  observation result that has an `observation_result.source_call_id`, then one
  `user` message per observation result without one.

Steps with `step.is_copied_context: true` are included: they are part of the
context, even though they are not new. Steps with `step.extra.lmnr.synthetic: true`
contribute nothing: no LLM call issued them, so the model never saw them. If
code later shows the model their result, that message is new input and becomes
its own step (§6.1).

Writers emit `step.extra.lmnr.input_messages` only when rules 2 and 3 would give the wrong answer. This
happens when the conversation was not simply extended between calls: after
compaction or history editing, when an agent branches, or when calls of one
agent run concurrently. `step.extra.lmnr.input_messages` never contains the system prompt, so a
`system` message inside it is an in-history message.

When a writer emits `step.extra.lmnr.input_messages`, it SHOULD still emit the messages that
follow the longest common prefix with the previous call's input as ordinary
steps, so that readers unaware of this spec get a sensible linear conversation.

### 6.5 Tools

The tools offered to agent step *k* are:

1. `steps[k].extra.lmnr.tool_definitions`, if present (after resolving a `Ref`).
   This is the complete set, not a diff. `[]` means no tools.
2. Otherwise, `trajectory.agent.tool_definitions`.

Overrides are not sticky. Writers set `trajectory.agent.tool_definitions` to the set used
by the most calls (the earliest on a tie) and emit an override on every call
that used a different set.

Tool results:

- `observation_result.content` is what the model was given as the
  tool's result, when the writer can see it (the next call's input). Otherwise
  it is the tool span's output.
- Execution bounds and outcome come from the tool's span (§4.5, §4.6).

Tool work that no LLM call issued (code dispatching a tool directly) becomes an
`agent` step with `step.llm_call_count: 0`, `step.extra.lmnr.synthetic: true`,
and a tool call with a writer-generated id (`tool_call.extra.lmnr.synthetic_id: true`).

### 6.6 Deduplication

Any override (`step.extra.lmnr.system_prompt`, `step.extra.lmnr.tool_definitions`,
`step.extra.lmnr.input_messages`) MAY be a `Ref`: `{"$ref": "<key>"}`. It
resolves against the containing trajectory's
`trajectory.extra.lmnr.defs.system_prompts`,
`trajectory.extra.lmnr.defs.tool_definitions` or
`trajectory.extra.lmnr.defs.input_messages` respectively. If the key isn't there, readers look in the
enclosing trajectories of an embedded subagent, innermost first.

Keys are opaque strings chosen by the writer. Readers MUST NOT recompute or
interpret them. Writers SHOULD prefix them with the hash algorithm, e.g.
`sha256:<hex of the value's canonical JSON>`.

A `Ref` that doesn't resolve is treated as if the override were absent.

### 6.7 Subagents

A subagent is a separate agent instance with its own conversation, not a
separate LLM call. It becomes an entry in `trajectory.subagent_trajectories`
with `trajectory.trajectory_id = "lmnr-" + <span id>` and its own
`trajectory.extra.lmnr`.

- The subagent is referenced (`observation_result.subagent_trajectory_ref`)
  from the observation result of the tool call that spawned it.
- If no tool call spawned it, the writer adds a synthetic `step.llm_call_count: 0`
  step whose tool call is named after the subagent and whose observation result
  holds the reference.
- LLM calls nested inside a tool with no agent marker of their own become a
  subagent named after the tool, with `trajectory.extra.lmnr.implicit_subagent: true`, so they stay
  out of the parent's conversation.

### 6.8 What stays in Laminar

ATIF has no place for spans that are neither LLM calls nor tool calls
(retrieval, HTTP requests, custom functions). They are not written.
`trajectory.extra.lmnr.trace_ids` (plus `trajectory.extra.lmnr.project_id`
when present) and each `step.extra.lmnr.span_id` link back to the full trace,
where those spans still are.

## 7. Reader rules

- A missing or incompatible `trajectory.extra.lmnr.spec_version` means plain ATIF (§2).
- Unknown keys are ignored.
- A `Ref` that doesn't resolve means the override is absent (§6.6).
- A tool call failed when its observation result has
  `observation_result.extra.lmnr.status == "error"`. For files written by
  other producers, readers also accept Harbor's
  `observation_result.extra.tool_result_is_error == true` and
  `observation_result.extra.is_error == true`.
- Usage fields are read from `metrics.extra.lmnr`. For files written by other
  producers, readers also accept `metrics.extra.reasoning_tokens` and
  `metrics.extra.cache_creation_input_tokens`.

## 8. Fixtures

[`fixtures/`](./fixtures) holds example trajectories that follow this spec.
Every fixture MUST validate against both [`lmnr-atif.schema.json`](./lmnr-atif.schema.json)
and Harbor's ATIF validator:

```sh
uv run --with harbor==0.25.0 python -m harbor.utils.trajectory_validator spec/lmnr-atif/fixtures/<file>.json
```

[`writer-golden/`](./writer-golden) is the conformance suite for writers: the
spans of a run, and the exact trajectory a writer must produce from them. Its
README defines the span input format.

## Changelog

### 0.1.0

- Initial version.

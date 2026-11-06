"""Build an lmnr-atif trajectory (`spec/lmnr-atif/lmnr-atif.md`) from spans.

`build_trajectory` is a pure function from finished spans (`SpanRecord`s) to
the trajectory dict. It doesn't touch the tracing stack: the span processor
that collects records and the file writer live elsewhere, so this can be
tested from JSON fixtures, and the same fixtures can test the TypeScript port.

Overview (spec §6):

1. Spans form a tree per trace; all traces go into one trajectory.
2. The tree is cut into agent scopes. A span is a scope boundary if it marks a
   new agent (`gen_ai.agent.name` or `gen_ai.operation.name ==
   "invoke_agent"`, with a different name from the enclosing agent), or if it
   is a TOOL span that runs LLM calls itself (an implicit subagent). A lone
   agent at the top with nothing else beside it is the root agent, not a
   subagent. Each boundary becomes an embedded subagent trajectory.
3. Within a scope, each LLM span is one agent step. Messages new in a call's
   input become user/system steps before it; tool results in the next call's
   input become the observation of the call that made them. When a call's
   input isn't the previous call's input + output + new messages, the step
   gets an `input_messages` override.
4. TOOL spans are matched to the tool calls the LLM made (by call id, then
   name and arguments, then name). Unmatched ones become synthetic
   `llm_call_count: 0` steps.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from lmnr.sdk.trajectory.normalize import (
    MediaSink,
    Message,
    content_text,
    normalize_call,
    normalize_tools,
    tool_call,
)
from lmnr.sdk.trajectory.reader import as_dict, dicts
from lmnr.sdk.trajectory.spec import ATIF_SCHEMA_VERSION, SPEC_VERSION
from lmnr.version import __version__

# `gen_ai.request.*` attributes that describe the transport, or that have their
# own place in ATIF, rather than request parameters.
_NOT_REQUEST_PARAMS = {"model", "base_url", "api_type", "api_version", "reasoning_effort"}
_SYSTEM_ROLES = ("system", "developer")


@dataclass
class SpanRecord:
    """A finished span, as much of it as the builder needs.

    The JSON form (`to_json` / `from_json`) is language-neutral: ids are
    Laminar UUID strings, times are ISO 8601 strings with microseconds, and
    attributes are the span's raw OTel attribute values.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    span_type: str
    start_time: str
    end_time: str
    attributes: dict[str, Any] = field(default_factory=dict)
    # `{type, message[, stacktrace]}` from the span's `exception` event.
    error: dict[str, str] | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> SpanRecord:
        return cls(
            trace_id=str(value["trace_id"]),
            span_id=str(value["span_id"]),
            parent_span_id=value.get("parent_span_id"),
            name=str(value.get("name") or ""),
            span_type=str(value.get("span_type") or "DEFAULT"),
            start_time=str(value["start_time"]),
            end_time=str(value["end_time"]),
            attributes=dict(value.get("attributes") or {}),
            error=value.get("error"),
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attributes": self.attributes,
        }
        if self.error is not None:
            result["error"] = self.error
        return result


@dataclass
class BuildOptions:
    producer: dict[str, str] = field(
        default_factory=lambda: {"name": "lmnr-python", "version": __version__}
    )
    # `agent.version`; falls back to the spans' `lmnr.span.release`.
    release: str | None = None
    project_id: str | None = None
    exported: bool = False
    # False while spans of the run are still open (a snapshot).
    complete: bool = True
    # Stores inline media; without it, images become text placeholders.
    media: MediaSink | None = None


def parse_time_ns(value: str) -> int:
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    delta = dt - datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    return (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def format_time(ns: int) -> str:
    seconds, rest = divmod(ns, 10**9)
    dt = datetime.datetime.fromtimestamp(seconds, datetime.UTC).replace(microsecond=rest // 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def def_key(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _json_attribute(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _message_key(message: Message) -> str:
    """Identity of a message across calls: what the model sees, minus reasoning
    (often not sent back) and writer metadata."""
    calls = [
        [c.get("tool_call_id"), c.get("function_name"), c.get("arguments")]
        for c in dicts(message.get("tool_calls"))
    ]
    return canonical_json(
        [
            message.get("role"),
            message.get("content") if message.get("content") is not None else "",
            calls,
            message.get("tool_call_id"),
        ]
    )


def _public(message: Message) -> Message:
    """A normalized message without the normalizer's internal `_` keys."""
    return {k: v for k, v in message.items() if not k.startswith("_")}


class _Span:
    def __init__(self, record: SpanRecord):
        self.record: SpanRecord = record
        self.id: str = record.span_id
        self.attrs: dict[str, Any] = record.attributes
        self.start: int = parse_time_ns(record.start_time)
        self.end: int = max(parse_time_ns(record.end_time), self.start)
        self.parent: _Span | None = None
        self.children: list[_Span] = []
        self.is_llm: bool = record.span_type == "LLM"
        self.is_tool: bool = record.span_type == "TOOL"

    @property
    def agent_marker(self) -> str | None:
        name = self.attrs.get("gen_ai.agent.name")
        if isinstance(name, str) and name:
            return name
        if self.attrs.get("gen_ai.operation.name") == "invoke_agent":
            return self.record.name or "agent"
        return None

    def descendants(self) -> Iterable[_Span]:
        for child in self.children:
            yield child
            yield from child.descendants()

    def hosts_llm(self) -> bool:
        """An LLM call runs below this span with no agent marker in between."""
        stack = list(self.children)
        while stack:
            span = stack.pop()
            if span.agent_marker is not None:
                continue
            if span.is_llm:
                return True
            stack.extend(span.children)
        return False

    def json_attr(self, key: str) -> Any:
        return _json_attribute(self.attrs.get(key))


@dataclass
class _Call:
    span: _Span
    system: list[Message]
    conversation: list[Message]
    output: Message | None
    finish_reason: str | None
    tools: list[dict[str, Any]]


@dataclass
class _Totals:
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    cost: float | None = None
    start: int | None = None
    end: int | None = None

    def add_cost(self, value: float | None) -> None:
        if value is not None:
            self.cost = (self.cost or 0) + value

    def add_time(self, start: int, end: int) -> None:
        self.start = start if self.start is None else min(self.start, start)
        self.end = end if self.end is None else max(self.end, end)

    def merge(self, other: _Totals) -> None:
        self.llm_calls += other.llm_calls
        self.tool_calls += other.tool_calls
        self.errors += other.errors
        self.add_cost(other.cost)
        if other.start is not None and other.end is not None:
            self.add_time(other.start, other.end)


class _Builder:
    def __init__(self, records: list[SpanRecord], options: BuildOptions):
        self.options: BuildOptions = options
        spans: dict[str, _Span] = {}
        for record in records:
            spans[record.span_id] = _Span(record)
        self.spans: list[_Span] = sorted(spans.values(), key=lambda s: (s.start, s.end))
        self.by_id: dict[str, _Span] = spans
        for span in self.spans:
            parent_id = span.record.parent_span_id
            span.parent = spans.get(parent_id) if parent_id else None
            if span.parent is not None:
                span.parent.children.append(span)
        self.roots: list[_Span] = [s for s in self.spans if s.parent is None]
        # When frameworks nest LLM spans (a LangChain run around the provider
        # call), the innermost one has the provider's exact messages.
        for span in self.spans:
            if span.is_llm and any(d.is_llm for d in span.descendants()):
                span.is_llm = False

        self.trace_ids: list[str] = list(dict.fromkeys(s.record.trace_id for s in self.spans))
        self.multi_trace: bool = len(self.trace_ids) > 1
        self.release: str = options.release or next(
            (r for s in self.spans if isinstance(r := s.attrs.get("lmnr.span.release"), str) and r),
            "unknown",
        )
        self.session_id: str | None = next(
            (
                v
                for s in self.spans
                if isinstance(v := s.attrs.get("lmnr.association.properties.session_id"), str) and v
            ),
            None,
        )
        self._assign_scopes()

    # -- scopes -------------------------------------------------------------

    def _assign_scopes(self) -> None:
        collapsed: set[str] = set()
        while True:
            self.scope: dict[str, str | None] = {}
            self.boundaries: dict[str, str] = {}
            for root in self.roots:
                self._visit(root, None, None, collapsed)
            own = [s for s in self.spans if (s.is_llm or s.is_tool) and self.scope[s.id] is None]
            top = [b for b, scope in self._boundary_scopes() if scope is None]
            if own or len(top) != 1 or self.boundaries[top[0]] != "agent":
                return
            collapsed.add(top[0])

    def _boundary_scopes(self) -> list[tuple[str, str | None]]:
        return [(b, self.scope[b]) for b in self.boundaries]

    def _visit(
        self, span: _Span, scope: str | None, agent: str | None, collapsed: set[str]
    ) -> None:
        self.scope[span.id] = scope
        marker = span.agent_marker
        child_scope, child_agent = scope, agent
        if marker is not None and marker != agent and span.id not in collapsed:
            self.boundaries[span.id] = "agent"
            child_scope, child_agent = span.id, marker
        elif span.is_tool and span.hosts_llm():
            self.boundaries[span.id] = "implicit"
            child_scope, child_agent = span.id, _tool_name(span)
        elif marker is not None:
            child_agent = marker
        for child in span.children:
            self._visit(child, child_scope, child_agent, collapsed)

    # -- calls --------------------------------------------------------------

    def _call(self, span: _Span) -> _Call:
        input_raw = span.json_attr("gen_ai.input.messages")
        if input_raw is None:
            input_raw = span.json_attr("lmnr.span.input")
        output_raw = span.json_attr("gen_ai.output.messages")
        if output_raw is None:
            output_raw = span.json_attr("lmnr.span.output")
        messages, output, finish_reason = normalize_call(input_raw, output_raw, self.options.media)

        system: list[Message] = []
        while messages and messages[0].get("role") in _SYSTEM_ROLES:
            system.append(messages.pop(0))
        # The OpenAI Responses API records its system prompt (`instructions`)
        # only in the legacy indexed attributes.
        if not system and span.attrs.get("gen_ai.prompt.0.role") == "system":
            content = span.attrs.get("gen_ai.prompt.0.content")
            if isinstance(content, str) and content:
                system = [{"role": "system", "content": content}]

        reasons = span.attrs.get("gen_ai.response.finish_reasons")
        if isinstance(reasons, (list, tuple)) and reasons:
            finish_reason = str(next(iter(cast(Any, reasons))))
        elif isinstance(reasons, str) and reasons:
            finish_reason = reasons

        # Calls whose tool calls have no id get one, so results can point at it.
        if output is not None:
            for index, call in enumerate(dicts(output.get("tool_calls"))):
                if not call.get("tool_call_id"):
                    call["tool_call_id"] = f"lmnr-{span.id}-{index}"
                    call.setdefault("extra", {}).setdefault("lmnr", {})["synthetic_id"] = True
        tools = normalize_tools(span.json_attr("gen_ai.tool.definitions"))
        return _Call(span, system, messages, output, finish_reason, tools)

    # -- trajectories ---------------------------------------------------------

    def build(self) -> dict[str, Any] | None:
        trajectory, _totals = self._trajectory(None)
        if not trajectory["steps"]:
            return None
        return trajectory

    def _trajectory(self, scope: str | None) -> tuple[dict[str, Any], _Totals]:
        spans = [s for s in self.spans if self.scope.get(s.id) == scope]
        calls = [self._call(s) for s in spans if s.is_llm]
        tool_spans = [s for s in spans if s.is_tool]
        children = [b for b, parent in self._boundary_scopes() if parent == scope]
        totals = _Totals()

        sub_trajectories: dict[str, dict[str, Any]] = {}
        for child in children:
            sub, sub_totals = self._trajectory(child)
            sub_trajectories[child] = sub
            totals.merge(sub_totals)

        default_system = calls[0].system if calls else []
        # A system message right after the default system prompt (a compaction
        # note, injected context) is part of the conversation, not the prompt.
        # Without a default system prompt, a later call's leading system
        # messages are its own system prompt.
        default_keys = [_message_key(m) for m in default_system]
        for call in calls[1:] if default_system else []:
            extra = len(call.system) - len(default_system)
            if (
                extra > 0
                and [_message_key(m) for m in call.system[: len(default_system)]] == default_keys
            ):
                call.conversation = call.system[len(default_system) :] + call.conversation
                call.system = call.system[: len(default_system)]
        tool_keys = [canonical_json(c.tools) for c in calls]
        default_tools: list[dict[str, Any]] = []
        if tool_keys:
            counts = Counter(tool_keys)
            best = max(counts.values())
            default_tools = next(c.tools for c in calls if counts[canonical_json(c.tools)] == best)

        # Overrides are collected first so values used twice can go in defs.
        overrides: list[dict[str, Any]] = [{} for _ in calls]
        # (time, call index, group, position, step): a call's new messages,
        # then the call, then steps without an LLM call.
        entries: list[tuple[int, int, int, int, dict[str, Any]]] = []
        matched = self._match_tools(calls, tool_spans)
        # Tool results seen in the next call's input, by (index of the call
        # that made the tool call, tool call id). Ids alone can repeat across
        # conversations (Gemini's generated `<name>_<k>`).
        results_from_input: dict[tuple[int, str], Message] = {}
        base: list[Message] = []
        prev_call_ids: list[str] = []
        for k, call in enumerate(calls):
            conversation = call.conversation
            keys = [_message_key(m) for m in conversation]
            base_keys = [_message_key(m) for m in base]
            common = 0
            while common < min(len(keys), len(base_keys)) and keys[common] == base_keys[common]:
                common += 1
            new = conversation[common:]
            results = [
                m for m in new if m.get("role") == "tool" and m.get("tool_call_id") in prev_call_ids
            ]
            others = [m for m in new if not any(m is r for r in results)]
            result_ids = [str(m.get("tool_call_id")) for m in results]
            # A reader rebuilds this input as: the previous input and output,
            # that call's tool results in tool-call order (from the input, or
            # else from the tool spans), then the new messages as steps. Any
            # other shape needs the exact input written out.
            rebuilt = sorted(results, key=lambda m: prev_call_ids.index(str(m.get("tool_call_id"))))
            from_spans = [i for i in prev_call_ids if i not in result_ids and (k - 1, i) in matched]
            extends = (
                common == len(base)
                and len(set(result_ids)) == len(result_ids)
                and not from_spans
                and [_message_key(m) for m in rebuilt + others] == keys[common:]
                and all(m.get("role") not in ("assistant", "tool") for m in others)
            )
            if not extends:
                overrides[k]["input_messages"] = [_public(m) for m in conversation]
            for j, message in enumerate(new):
                role = message.get("role")
                if any(message is r for r in results):
                    key = (k - 1, str(message["tool_call_id"]))
                    _ = results_from_input.setdefault(key, message)
                    continue
                if role == "assistant":
                    continue
                step: dict[str, Any] = {
                    "timestamp": format_time(call.span.start),
                    "source": "system" if role in _SYSTEM_ROLES else "user",
                    "message": (
                        message.get("content")
                        if role != "tool"
                        else content_text(message.get("content"))
                    )
                    or "",
                    "extra": {"lmnr": {"source_span_id": call.span.id}},
                }
                if role == "developer":
                    step["extra"]["lmnr"]["role"] = "developer"
                entries.append((call.span.start, k, 0, j, step))

            if [_message_key(m) for m in call.system] != [_message_key(m) for m in default_system]:
                if not call.system:
                    overrides[k]["system_prompt"] = ""
                elif len(call.system) == 1:
                    overrides[k]["system_prompt"] = call.system[0].get("content") or ""
                else:
                    overrides[k]["system_prompt"] = "\n\n".join(
                        content_text(m.get("content")) for m in call.system
                    )
            if canonical_json(call.tools) != canonical_json(default_tools):
                overrides[k]["tool_definitions"] = call.tools

            # A call without output (a failed one) is still a step, and a reader
            # adds its empty assistant message to the next input (§6.4).
            base = conversation + [
                call.output if call.output is not None else {"role": "assistant", "content": ""}
            ]
            prev_call_ids = [
                str(c.get("tool_call_id")) for c in dicts((call.output or {}).get("tool_calls"))
            ]

        # Defs: override values used by more than one step.
        defs: dict[str, dict[str, Any]] = {}
        kinds = {
            "system_prompt": "system_prompts",
            "tool_definitions": "tool_definitions",
            "input_messages": "input_messages",
        }
        usage = Counter(
            (name, canonical_json(value))
            for override in overrides
            for name, value in override.items()
            if value not in ("", [])
        )
        for override in overrides:
            for name, value in list(override.items()):
                if value in ("", []) or usage[(name, canonical_json(value))] < 2:
                    continue
                key = def_key(value)
                defs.setdefault(kinds[name], {})[key] = value
                override[name] = {"$ref": key}

        matched_spans = {span.id for span, _ in matched.values()}

        # Which observation result (call index, tool call id) or synthetic step
        # (span id) references each subagent.
        refs_by_call: dict[tuple[int, str], list[str]] = {}
        refs_by_span: dict[str, list[str]] = {}
        unspawned: list[str] = []
        span_to_call = {span.id: key for key, (span, _) in matched.items()}
        for child in children:
            spawner = self._spawner(child, scope)
            if spawner is None:
                unspawned.append(child)
            elif spawner.id in span_to_call:
                refs_by_call.setdefault(span_to_call[spawner.id], []).append(child)
            else:
                refs_by_span.setdefault(spawner.id, []).append(child)

        for k, call in enumerate(calls):
            step, step_totals = self._agent_step(
                call, k, overrides[k], matched, results_from_input, refs_by_call, sub_trajectories
            )
            totals.merge(step_totals)
            entries.append((call.span.start, k, 1, 0, step))

        # Steps without an LLM call sort after any call starting at the same time.
        after_calls = len(calls)
        for span in tool_spans:
            if span.id in matched_spans:
                continue
            step, step_totals = self._synthetic_tool_step(
                span, refs_by_span.get(span.id, []), sub_trajectories
            )
            totals.merge(step_totals)
            entries.append((span.start, after_calls, 2, 0, step))
        for child in unspawned:
            span = self.by_id[child]
            step = self._synthetic_agent_step(span, sub_trajectories[child])
            totals.tool_calls += 1
            if span.record.error:
                totals.errors += 1
            entries.append((span.start, after_calls, 2, 0, step))

        entries.sort(key=lambda e: e[:4])
        steps = [self._system_step(m, calls[0]) for m in default_system] + [e[4] for e in entries]
        steps = [{"step_id": i + 1, **step} for i, step in enumerate(steps)]
        document = self._document(
            scope, steps, calls, default_tools, defs, totals, sub_trajectories
        )
        return document, totals

    def _spawner(self, child: str, scope: str | None) -> _Span | None:
        """The TOOL span in `scope` that started subagent `child`, if any."""
        span = self.by_id[child]
        if self.boundaries[child] == "implicit":
            return span
        node = span.parent
        while node is not None and node.id != scope:
            if node.is_tool:
                return node
            node = node.parent
        return None

    def _system_step(self, message: Message, first_call: _Call) -> dict[str, Any]:
        lmnr: dict[str, Any] = {"source_span_id": first_call.span.id}
        if message.get("role") == "developer":
            lmnr["role"] = "developer"
        return {
            "timestamp": format_time(first_call.span.start),
            "source": "system",
            "message": message.get("content") or "",
            "extra": {"lmnr": lmnr},
        }

    def _match_tools(
        self, calls: list[_Call], tool_spans: list[_Span]
    ) -> dict[tuple[int, str], tuple[_Span, dict[str, Any]]]:
        """TOOL spans that executed each LLM call's tool calls, by (call, id)."""
        matched: dict[tuple[int, str], tuple[_Span, dict[str, Any]]] = {}
        used: set[str] = set()
        windows: list[list[_Span]] = []
        for k, call in enumerate(calls):
            # A call's tools start after it does, and before the next call that
            # started after it ended (and so could have seen their results).
            # Calls running at the same time (fan-out) don't cut the window.
            window_end = next(
                (c.span.start for c in calls[k + 1 :] if c.span.start >= call.span.end), None
            )
            windows.append(
                [
                    s
                    for s in tool_spans
                    if s.start >= call.span.start and (window_end is None or s.start < window_end)
                ]
            )
        # By call id (exact, so not limited to the window), then by name and
        # arguments, then by name alone. Each strategy runs over every call
        # before the next, weaker one, so a name-only match can't take a span
        # that a concurrent call matches more precisely.
        for strategy in range(3):
            for k, call in enumerate(calls):
                if strategy == 0:
                    window = [s for s in tool_spans if s.start >= call.span.start]
                else:
                    window = windows[k]
                for tool in dicts((call.output or {}).get("tool_calls")):
                    key = (k, str(tool.get("tool_call_id")))
                    if key in matched:
                        continue
                    span = next(
                        (
                            s
                            for s in window
                            if s.id not in used and _tool_matches(strategy, s, tool)
                        ),
                        None,
                    )
                    if span is not None:
                        matched[key] = (span, tool)
                        used.add(span.id)
        return matched

    def _span_extra(self, span: _Span) -> dict[str, Any]:
        extra: dict[str, Any] = {"span_id": span.id}
        if self.multi_trace:
            extra["trace_id"] = span.record.trace_id
        return extra

    def _tool_output(self, span: _Span) -> str:
        value = span.json_attr("lmnr.span.output")
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _result_extra(self, span: _Span | None, from_input: Message | None) -> dict[str, Any]:
        if span is not None:
            if span.record.error:
                return {"status": "error", "error": _error(span.record.error)}
            return {"status": "ok"}
        if from_input is not None and from_input.get("_is_error"):
            return {"status": "error", "error": {"type": "ToolError", "message": ""}}
        return {}

    def _refs(
        self, children: list[str], sub_trajectories: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "trajectory_id": sub_trajectories[child]["trajectory_id"],
                "extra": {"lmnr": {"span_id": child}},
            }
            for child in children
        ]

    def _agent_step(
        self,
        call: _Call,
        k: int,
        overrides: dict[str, Any],
        matched: dict[tuple[int, str], tuple[_Span, dict[str, Any]]],
        results_from_input: dict[tuple[int, str], Message],
        refs_by_call: dict[tuple[int, str], list[str]],
        sub_trajectories: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], _Totals]:
        span, attrs = call.span, call.span.attrs
        totals = _Totals(llm_calls=1)
        totals.add_time(span.start, span.end)
        output = call.output or {}
        step: dict[str, Any] = {"timestamp": format_time(span.end), "source": "agent"}
        model = attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model")
        if isinstance(model, str) and model:
            step["model_name"] = model
        effort = attrs.get("gen_ai.request.reasoning_effort")
        if isinstance(effort, (str, int, float)) and not isinstance(effort, bool):
            step["reasoning_effort"] = effort
        step["message"] = output.get("content") or ""
        if output.get("reasoning_content"):
            step["reasoning_content"] = output["reasoning_content"]

        tool_calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for tool in dicts(output.get("tool_calls")):
            call_id = str(tool.get("tool_call_id"))
            entry = dict(tool)
            lmnr = dict(as_dict(as_dict(tool.get("extra")).get("lmnr")))
            tool_span = matched.get((k, call_id), (None, None))[0]
            if tool_span is not None:
                lmnr["span_id"] = tool_span.id
                lmnr["start_time"] = format_time(tool_span.start)
                lmnr["end_time"] = format_time(tool_span.end)
            if lmnr:
                entry["extra"] = {"lmnr": lmnr}
            tool_calls.append(entry)
            totals.tool_calls += 1

            from_input = results_from_input.get((k, call_id))
            refs = refs_by_call.get((k, call_id), [])
            if from_input is None and tool_span is None and not refs:
                continue
            result: dict[str, Any] = {"source_call_id": call_id}
            if from_input is not None:
                result["content"] = from_input.get("content") or ""
            elif tool_span is not None:
                result["content"] = self._tool_output(tool_span)
            else:
                result["content"] = _final_message(sub_trajectories[refs[0]])
            if refs:
                result["subagent_trajectory_ref"] = self._refs(refs, sub_trajectories)
            result_extra = self._result_extra(tool_span, from_input)
            if result_extra:
                result["extra"] = {"lmnr": result_extra}
                if result_extra["status"] == "error":
                    totals.errors += 1
            results.append(result)
        if tool_calls:
            step["tool_calls"] = tool_calls
        if results:
            step["observation"] = {"results": results}

        metrics: dict[str, Any] = {}
        for key, attribute in (
            ("prompt_tokens", "gen_ai.usage.input_tokens"),
            ("completion_tokens", "gen_ai.usage.output_tokens"),
            ("cached_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ):
            if (value := _number(attrs.get(attribute))) is not None:
                metrics[key] = int(value)
        input_cost = _number(attrs.get("gen_ai.usage.input_cost"))
        output_cost = _number(attrs.get("gen_ai.usage.output_cost"))
        cost = _number(attrs.get("gen_ai.usage.cost"))
        if cost is None and input_cost is not None and output_cost is not None:
            cost = input_cost + output_cost
        if cost is not None:
            metrics["cost_usd"] = cost
            totals.add_cost(cost)
        metrics_extra: dict[str, Any] = {}
        for key, attribute in (
            ("reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
            ("cache_creation_tokens", "gen_ai.usage.cache_creation_input_tokens"),
        ):
            if (value := _number(attrs.get(attribute))) is not None and value > 0:
                metrics_extra[key] = int(value)
        if input_cost is not None:
            metrics_extra["input_cost_usd"] = input_cost
        if output_cost is not None:
            metrics_extra["output_cost_usd"] = output_cost
        if metrics_extra:
            metrics["extra"] = {"lmnr": metrics_extra}
        if metrics:
            step["metrics"] = metrics
        step["llm_call_count"] = 1

        lmnr_step = self._span_extra(span) | {
            "start_time": format_time(span.start),
            "end_time": format_time(span.end),
        }
        provider = attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system")
        if isinstance(provider, str) and provider:
            lmnr_step["provider"] = provider
        request = {
            key[len("gen_ai.request.") :]: _json_attribute(value)
            if key.endswith("_schema")
            else value
            for key, value in attrs.items()
            if key.startswith("gen_ai.request.")
            and key[len("gen_ai.request.") :] not in _NOT_REQUEST_PARAMS
            and value not in (None, "", [], ())
        }
        if request:
            lmnr_step["request"] = dict(sorted(request.items()))
        if call.finish_reason:
            lmnr_step["finish_reason"] = call.finish_reason
        if isinstance(attrs.get("gen_ai.response.id"), str):
            lmnr_step["response_id"] = attrs["gen_ai.response.id"]
        if span.record.error:
            lmnr_step["status"] = "error"
            lmnr_step["error"] = _error(span.record.error)
            totals.errors += 1
        else:
            lmnr_step["status"] = "ok"
        lmnr_step.update(overrides)
        step["extra"] = {"lmnr": lmnr_step}
        return step, totals

    def _synthetic_tool_step(
        self,
        span: _Span,
        children: list[str],
        sub_trajectories: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], _Totals]:
        totals = _Totals(tool_calls=1)
        totals.add_time(span.start, span.end)
        call_id = f"lmnr-{span.id}"
        arguments = span.json_attr("lmnr.span.input")
        call = tool_call(call_id, _tool_name(span) or "tool", arguments, synthetic_id=True)
        call["extra"]["lmnr"].update(
            {
                "span_id": span.id,
                "start_time": format_time(span.start),
                "end_time": format_time(span.end),
            }
        )
        result: dict[str, Any] = {"source_call_id": call_id, "content": self._tool_output(span)}
        if children:
            result["subagent_trajectory_ref"] = self._refs(children, sub_trajectories)
        result["extra"] = {"lmnr": self._result_extra(span, None)}
        if span.record.error:
            totals.errors += 1
        return self._synthetic_step(span, call, result), totals

    def _synthetic_agent_step(self, span: _Span, sub: dict[str, Any]) -> dict[str, Any]:
        call_id = f"lmnr-{span.id}"
        arguments = span.json_attr("lmnr.span.input")
        name = span.agent_marker or span.record.name or "subagent"
        call = tool_call(
            call_id, name, arguments if isinstance(arguments, dict) else {}, synthetic_id=True
        )
        call["extra"]["lmnr"].update(
            {
                "span_id": span.id,
                "start_time": format_time(span.start),
                "end_time": format_time(span.end),
            }
        )
        result = {
            "source_call_id": call_id,
            "content": _final_message(sub),
            "subagent_trajectory_ref": self._refs([span.id], {span.id: sub}),
            "extra": {"lmnr": {"status": "error", "error": _error(span.record.error)}}
            if span.record.error
            else {"lmnr": {"status": "ok"}},
        }
        return self._synthetic_step(span, call, result)

    def _synthetic_step(
        self, span: _Span, call: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "timestamp": format_time(span.end),
            "source": "agent",
            "message": "",
            "tool_calls": [call],
            "observation": {"results": [result]},
            "llm_call_count": 0,
            "extra": {
                "lmnr": {"synthetic": True}
                | self._span_extra(span)
                | {"start_time": format_time(span.start), "end_time": format_time(span.end)}
            },
        }

    def _document(
        self,
        scope: str | None,
        steps: list[dict[str, Any]],
        calls: list[_Call],
        default_tools: list[dict[str, Any]],
        defs: dict[str, dict[str, Any]],
        totals: _Totals,
        sub_trajectories: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        boundary = next((s for s in self.spans if s.id == scope), None)
        if boundary is None:
            name = self.roots[0].record.name if self.roots else "agent"
        elif self.boundaries[boundary.id] == "agent":
            name = boundary.agent_marker or boundary.record.name
        else:
            name = _tool_name(boundary)
        agent: dict[str, Any] = {"name": name or "agent", "version": self.release}
        models = Counter(
            "/".join(p for p in (_provider(c.span), _model(c.span)) if p)
            for c in calls
            if _model(c.span)
        )
        if models:
            best = max(models.values())
            agent["model_name"] = next(m for m in models if models[m] == best)
        if default_tools:
            agent["tool_definitions"] = default_tools

        document: dict[str, Any] = {"schema_version": ATIF_SCHEMA_VERSION}
        session = self.session_id or (self.trace_ids[0] if self.trace_ids else None)
        if session:
            document["session_id"] = session
        if boundary is not None:
            document["trajectory_id"] = f"lmnr-{boundary.id}"
        elif not self.multi_trace and self.trace_ids:
            document["trajectory_id"] = self.trace_ids[0]
        document["agent"] = agent
        document["steps"] = steps

        own_steps = [s for s in steps if s.get("source") == "agent"]
        final: dict[str, Any] = {}
        for key, metric in (
            ("total_prompt_tokens", "prompt_tokens"),
            ("total_completion_tokens", "completion_tokens"),
            ("total_cached_tokens", "cached_tokens"),
        ):
            values = [s["metrics"][metric] for s in own_steps if metric in s.get("metrics", {})]
            if values:
                final[key] = sum(values)
        # ATIF defines token totals over this trajectory's steps, but
        # `total_cost_usd` as including subagents, so only the cost is merged.
        if totals.cost is not None:
            final["total_cost_usd"] = round(totals.cost, 10)
        final["total_steps"] = len(steps)
        final_extra: dict[str, Any] = {}
        if totals.start is not None and totals.end is not None:
            final_extra["duration_ms"] = round((totals.end - totals.start) / 1e6, 3)
        final_extra |= {
            "llm_calls": totals.llm_calls,
            "tool_calls": totals.tool_calls,
            "errors": totals.errors,
        }
        final["extra"] = {"lmnr": final_extra}
        document["final_metrics"] = final

        lmnr: dict[str, Any] = {"spec_version": SPEC_VERSION}
        if boundary is None:
            lmnr["producer"] = dict(self.options.producer)
            lmnr["trace_ids"] = list(self.trace_ids)
            if not self.multi_trace and self.roots:
                lmnr["root_span_id"] = self.roots[0].id
            if self.options.project_id:
                lmnr["project_id"] = self.options.project_id
            lmnr["exported"] = self.options.exported
            lmnr["complete"] = self.options.complete
        else:
            lmnr["agent_span_id"] = boundary.id
            if self.boundaries[boundary.id] == "implicit":
                lmnr["implicit_subagent"] = True
        if defs:
            lmnr["defs"] = {
                kind: dict(sorted(table.items())) for kind, table in sorted(defs.items())
            }
        document["extra"] = {"lmnr": lmnr}
        if sub_trajectories:
            document["subagent_trajectories"] = list(sub_trajectories.values())
        return document


def _tool_matches(strategy: int, span: _Span, tool: dict[str, Any]) -> bool:
    if strategy == 0:
        return _tool_call_id(span) == tool.get("tool_call_id")
    if _tool_name(span) != tool.get("function_name"):
        return False
    return strategy == 2 or span.json_attr("lmnr.span.input") == tool.get("arguments")


def _tool_call_id(span: _Span) -> Any:
    """The id of the tool call a TOOL span executed (`ai.toolCall.id`: the AI SDK)."""
    for key in ("gen_ai.tool.call.id", "ai.toolCall.id"):
        if key in span.attrs:
            return span.attrs[key]
    return None


def _tool_name(span: _Span) -> str:
    """The name of the tool a TOOL span ran. Span names often decorate it:
    `execute_tool <name>` (OpenTelemetry GenAI conventions), `ai.tool <name>`
    (AI SDK), `<name>.tool` (MCP)."""
    for key in ("gen_ai.tool.name", "ai.toolCall.name"):
        value = span.attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return span.record.name


def _provider(span: _Span) -> str | None:
    value = span.attrs.get("gen_ai.provider.name") or span.attrs.get("gen_ai.system")
    return value if isinstance(value, str) and value else None


def _model(span: _Span) -> str | None:
    value = span.attrs.get("gen_ai.response.model") or span.attrs.get("gen_ai.request.model")
    return value if isinstance(value, str) and value else None


def _error(value: Any) -> dict[str, str]:
    error = as_dict(value)
    result = {
        "type": str(error.get("type") or "Error"),
        "message": str(error.get("message") or ""),
    }
    if error.get("stacktrace"):
        result["stacktrace"] = str(error["stacktrace"])
    return result


def _final_message(trajectory: dict[str, Any]) -> str:
    for step in reversed(dicts(trajectory.get("steps"))):
        if step.get("source") == "agent" and step.get("llm_call_count") != 0:
            text = content_text(step.get("message"))
            if text:
                return text
    return ""


def build_trajectory(
    records: Iterable[SpanRecord], options: BuildOptions | None = None
) -> dict[str, Any] | None:
    """Build the trajectory for a set of finished spans, or None if they hold
    no LLM or tool calls."""
    return _Builder(list(records), options or BuildOptions()).build()

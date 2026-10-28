"""Convert Harbor ATIF trajectories into Laminar spans.

Harbor agents record what they did as an ATIF trajectory (`agent/trajectory.json`
inside the trial directory). Agents usually run inside a sandbox where we can't
instrument their LLM calls, so the trajectory is the only record of the run we
get. This module replays it after the fact as LLM and TOOL spans under the
trial's agent span, using the timestamps recorded in the trajectory.

Trajectories written by a Laminar SDK carry lmnr-atif fields (`extra.lmnr`,
see `spec/lmnr-atif/lmnr-atif.md`): exact call and tool timings, per-call
system prompts and tools, errors, and links back to the original spans. They
are read through `lmnr.sdk.trajectory.reader`, which falls back to plain ATIF
semantics for every other trajectory.

Works on plain dicts (the JSON file) so it doesn't depend on `harbor` itself.
"""

import datetime
import json
from pathlib import Path
from typing import Any, cast

from opentelemetry.trace import Status, StatusCode

from lmnr import LaminarSpan
from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.trajectory import reader
from lmnr.sdk.trajectory.reader import TrajectoryExtension
from lmnr.sdk.types import LaminarSpanContext
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

# Guard against reference cycles between file-ref subagent trajectories.
MAX_SUBAGENT_DEPTH = 8

# ATIF gives a step's LLM call, its tool calls and the next LLM call the same
# timestamp, and the trace view orders siblings by start time at millisecond
# precision. Spans that would tie are nudged apart by this much so they show up
# in the order they happened.
ORDER_GAP_NS = 1_000_000

# Links a replayed span to the span it was recorded from (lmnr-atif `span_id`).
SOURCE_SPAN_ID = "lmnr.trajectory.source_span_id"


def _not_before(value: int | None, floor: int | None) -> int | None:
    if value is None or floor is None:
        return value
    return max(value, floor)


def _shift(value: int | None, delta: int) -> int | None:
    return None if value is None else value + delta


def parse_timestamp_ns(value: Any) -> int | None:
    """Parse an ISO 8601 string or a datetime into nanoseconds since the epoch."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        try:
            # Python 3.10 `fromisoformat` doesn't accept the `Z` suffix.
            dt = datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return int(dt.timestamp() * 1e9)


def _bounds(lmnr: dict[str, Any]) -> tuple[int, int] | None:
    """Exact `(start, end)` from lmnr-atif `start_time` / `end_time`, if both parse."""
    start = parse_timestamp_ns(lmnr.get("start_time"))
    end = parse_timestamp_ns(lmnr.get("end_time"))
    if start is None or end is None:
        return None
    return start, max(start, end)


def split_model_name(model_name: str | None) -> tuple[str | None, str | None]:
    """Split a LiteLLM-style `provider/model` name into (provider, model)."""
    if not model_name:
        return None, None
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
        return provider, model
    return None, model_name


def content_to_text(content: Any) -> str:
    """Flatten an ATIF message / observation content into plain text.

    Images and audio are referenced by path rather than embedded.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            if part.get("type") == "text":
                parts.append(part.get("text") or "")
            else:
                source = part.get("source") or {}
                parts.append(f"[{part.get('type', 'attachment')}: {source.get('path')}]")
        return "\n".join(parts)
    return str(content)


def _tool_call_arguments(tool_call: dict[str, Any], ext: TrajectoryExtension | None) -> Any:
    """The arguments as the model produced them (lmnr-atif `raw_arguments` wins)."""
    raw = reader.fields(tool_call, ext).get("raw_arguments")
    return raw if isinstance(raw, str) else tool_call.get("arguments")


def _tool_call_to_openai(
    tool_call: dict[str, Any], ext: TrajectoryExtension | None = None
) -> dict[str, Any]:
    arguments = _tool_call_arguments(tool_call, ext)
    return {
        "id": tool_call.get("tool_call_id"),
        "type": "function",
        "function": {
            "name": tool_call.get("function_name"),
            "arguments": (arguments if isinstance(arguments, str) else json.dumps(arguments)),
        },
    }


def openai_message(
    message: dict[str, Any], ext: TrajectoryExtension | None = None
) -> dict[str, Any]:
    """An lmnr-atif `LmnrMessage` in the OpenAI chat format spans are shown in."""
    result: dict[str, Any] = {
        "role": message.get("role"),
        "content": content_to_text(message.get("content")),
    }
    if message.get("reasoning_content"):
        result["reasoning_content"] = message["reasoning_content"]
    tool_calls = reader.dicts(message.get("tool_calls"))
    if tool_calls:
        result["tool_calls"] = [_tool_call_to_openai(tc, ext) for tc in tool_calls]
    if message.get("tool_call_id"):
        result["tool_call_id"] = message["tool_call_id"]
    return result


def assistant_message(
    step: dict[str, Any], ext: TrajectoryExtension | None = None
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content_to_text(step.get("message")),
    }
    if step.get("reasoning_content"):
        message["reasoning_content"] = step["reasoning_content"]
    if step.get("tool_calls"):
        message["tool_calls"] = [
            _tool_call_to_openai(tc, ext) for tc in reader.dicts(step["tool_calls"])
        ]
    return message


def _record_error(span: Any, error: dict[str, str], timestamp_ns: int | None) -> None:
    """Mark a span failed from a recorded `{type, message[, stacktrace]}`.

    app-server only treats a span as failed when it has an `exception` event,
    and there is no exception object to pass to `record_exception` here.
    """
    span.add_event(
        "exception",
        attributes={
            "exception.type": error["type"],
            "exception.message": error["message"],
            "exception.stacktrace": error.get("stacktrace", ""),
            "exception.escaped": True,
        },
        timestamp=timestamp_ns,
    )
    span.set_status(Status(StatusCode.ERROR, error["message"]))


def _request_attribute(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    return json_dumps(value)


class TrajectoryConverter:
    """Emit spans for an ATIF trajectory under a parent span.

    Each agent step that made an LLM call becomes an LLM span whose input is the
    conversation so far and whose output is the assistant message. Its duration
    runs from the previous step's timestamp to this step's, since that is when
    the agent was waiting on the model. Each tool call becomes a TOOL span at
    the step's timestamp. ATIF doesn't record how long a tool ran, so tool
    spans have no duration, unless the call spawned a subagent: embedded and
    file-referenced subagent trajectories become nested spans under the tool
    call, covering the time range of the subagent's own steps.

    Since these timings are partly synthetic, sibling spans are nudged by
    `ORDER_GAP_NS` where needed so that start times follow the trajectory's
    order: an LLM call, then its tool calls one by one, then the next LLM call
    after the last tool span has ended.

    lmnr-atif trajectories record exact call and tool bounds; spans that have
    them use them as-is, with no nudging. Their input is the call's exact input
    (per-call system prompt, or an `input_messages` override after compaction),
    and their tool definitions, request parameters and errors are recorded too.
    One with `exported: true` emits no spans: they are already in Laminar.
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir
        self.span_count = 0
        # Attributes for the span the trajectory was emitted under (the agent
        # span), from the root trajectory's lmnr-atif fields.
        self.parent_attributes: dict[str, Any] = {}
        self._last_start: int | None = None
        self._tools_end: int | None = None

    def emit(
        self,
        trajectory: dict[str, Any],
        parent: LaminarSpanContext,
        start_ns: int | None = None,
        depth: int = 0,
        parent_ext: TrajectoryExtension | None = None,
    ) -> None:
        ext = TrajectoryExtension.read(trajectory, parent=parent_ext)
        if depth == 0 and ext is not None:
            self.parent_attributes = _trajectory_attributes(trajectory, ext)
            # The SDK already exported the spans behind this file; rebuilding
            # them would record every call twice. `lmnr.trajectory.trace_ids`
            # on the agent span links to them instead.
            if ext.fields.get("exported") is True:
                return
        steps = reader.steps_of(trajectory)
        agent = trajectory.get("agent") or {}
        default_model = agent.get("model_name")
        subagents = cast(
            dict[str, Any],
            {
                sub.get("trajectory_id"): sub
                for sub in reader.dicts(trajectory.get("subagent_trajectories"))
                if sub.get("trajectory_id")
            },
        )

        # Latest span start and latest tool span end so far at this level, used
        # to keep sibling start times in trajectory order. Saved and restored
        # because subagents recurse into `emit` with their own level.
        outer_order = self._last_start, self._tools_end
        self._last_start, self._tools_end = None, None
        try:
            self._emit_steps(
                trajectory, steps, parent, start_ns, depth, default_model, subagents, ext
            )
        finally:
            self._last_start, self._tools_end = outer_order

    def _emit_steps(
        self,
        trajectory: dict[str, Any],
        steps: list[dict[str, Any]],
        parent: LaminarSpanContext,
        start_ns: int | None,
        depth: int,
        default_model: str | None,
        subagents: dict[str, dict[str, Any]],
        ext: TrajectoryExtension | None,
    ) -> None:
        prev_ns = start_ns
        for index, step in enumerate(steps):
            step_ns = parse_timestamp_ns(step.get("timestamp"))
            is_agent_step = step.get("source") == "agent" and not step.get("is_copied_context")
            if is_agent_step and step.get("llm_call_count") != 0:
                bounds = _bounds(reader.fields(step, ext))
                if bounds is not None:
                    llm_start, llm_end = bounds
                else:
                    llm_start = prev_ns if prev_ns is not None else step_ns
                    llm_start = _not_before(llm_start, _shift(self._last_start, ORDER_GAP_NS))
                    llm_start = _not_before(llm_start, _shift(self._tools_end, ORDER_GAP_NS))
                    llm_end = _not_before(step_ns, llm_start)
                self._emit_llm_span(
                    trajectory,
                    steps,
                    index,
                    parent,
                    start_ns=llm_start,
                    end_ns=llm_end,
                    default_model=default_model,
                    ext=ext,
                )
            if is_agent_step:
                self._emit_tool_spans(
                    step,
                    parent,
                    step_ns=step_ns if step_ns is not None else prev_ns,
                    subagents=subagents,
                    depth=depth,
                    ext=ext,
                )
            if step_ns is not None:
                prev_ns = step_ns

    def _emit_llm_span(
        self,
        trajectory: dict[str, Any],
        steps: list[dict[str, Any]],
        index: int,
        parent: LaminarSpanContext,
        start_ns: int | None,
        end_ns: int | None,
        default_model: str | None,
        ext: TrajectoryExtension | None,
    ) -> None:
        step = steps[index]
        lmnr = reader.fields(step, ext)
        model_name = step.get("model_name") or default_model
        provider, model = split_model_name(model_name)
        if isinstance(lmnr.get("provider"), str) and lmnr["provider"]:
            provider = lmnr["provider"]
        if provider is None:
            # Agents like terminus-2 record the bare model the API returned
            # ("claude-sonnet-4-5-20250929"); the agent's model has the provider.
            provider = split_model_name(default_model)[0]
        attributes: dict[str, Any] = {}
        if provider:
            attributes[Attributes.PROVIDER.value] = provider
        if model:
            attributes[Attributes.REQUEST_MODEL.value] = model
            attributes[Attributes.RESPONSE_MODEL.value] = model
        for key, value in reader.as_dict(lmnr.get("request")).items():
            if value is not None:
                attributes[f"gen_ai.request.{key}"] = _request_attribute(value)
        if isinstance(step.get("reasoning_effort"), (str, int, float)):
            attributes["gen_ai.request.reasoning_effort"] = step["reasoning_effort"]
        if isinstance(lmnr.get("finish_reason"), str):
            attributes["gen_ai.response.finish_reasons"] = [lmnr["finish_reason"]]
        if isinstance(lmnr.get("response_id"), str):
            attributes[Attributes.RESPONSE_ID.value] = lmnr["response_id"]
        tools = reader.tool_definitions(trajectory, step, ext)
        if tools:
            attributes["gen_ai.tool.definitions"] = json_dumps(tools)

        metrics = cast(dict[str, int | float], step.get("metrics") or {})
        input_tokens = cast(int | None, metrics.get("prompt_tokens"))
        output_tokens = cast(int | None, metrics.get("completion_tokens"))
        if input_tokens is not None:
            attributes[Attributes.INPUT_TOKEN_COUNT.value] = input_tokens
        if output_tokens is not None:
            attributes[Attributes.OUTPUT_TOKEN_COUNT.value] = output_tokens
        if input_tokens is not None or output_tokens is not None:
            attributes[Attributes.TOTAL_TOKEN_COUNT.value] = (input_tokens or 0) + (
                output_tokens or 0
            )
        if metrics.get("cached_tokens") is not None:
            attributes["gen_ai.usage.cache_read_input_tokens"] = metrics["cached_tokens"]
        if metrics.get("cost_usd") is not None:
            attributes[Attributes.TOTAL_COST.value] = metrics["cost_usd"]
        usage = reader.usage_extension(step.get("metrics"), ext)
        for key, attribute in (
            ("reasoning_tokens", "gen_ai.usage.reasoning_tokens"),
            ("cache_creation_tokens", "gen_ai.usage.cache_creation_input_tokens"),
            ("input_cost_usd", Attributes.INPUT_COST.value),
            ("output_cost_usd", Attributes.OUTPUT_COST.value),
        ):
            if key in usage:
                attributes[attribute] = usage[key]
        if step.get("step_id") is not None:
            attributes["harbor.step_id"] = step["step_id"]
        if isinstance(lmnr.get("span_id"), str):
            attributes[SOURCE_SPAN_ID] = lmnr["span_id"]

        history = [openai_message(m, ext) for m in reader.input_messages(steps, index, ext)]
        span = cast(
            LaminarSpan,
            Laminar.start_span(
                model or "llm",
                input=history,
                span_type="LLM",
                parent_span_context=parent,
                attributes=attributes,
                start_time=start_ns,
            ),
        )
        span.set_output([assistant_message(step, ext)])
        error = reader.step_error(step, ext)
        if error is not None:
            _record_error(span, error, end_ns)
        span.end(end_time=end_ns)
        self.span_count += 1
        self._record_start(start_ns)

    def _record_start(self, start_ns: int | None) -> None:
        if start_ns is not None:
            self._last_start = _not_before(start_ns, self._last_start)

    def _emit_tool_spans(
        self,
        step: dict[str, Any],
        parent: LaminarSpanContext,
        step_ns: int | None,
        subagents: dict[str, dict[str, Any]],
        depth: int,
        ext: TrajectoryExtension | None,
    ) -> None:
        results_by_call: dict[str, list[dict[str, Any]]] = {}
        unattributed = []
        for result in reader.observation_results(step):
            if result.get("source_call_id"):
                results_by_call.setdefault(result["source_call_id"], []).append(result)
            else:
                unattributed.append(result)

        tool_calls = reader.dicts(step.get("tool_calls"))
        for i, tool_call in enumerate(tool_calls):
            lmnr = reader.fields(tool_call, ext)
            results = results_by_call.get(tool_call.get("tool_call_id") or "", [])
            if i == len(tool_calls) - 1:
                # Agents like terminus-2 record one terminal observation per step
                # without a source_call_id; it is the output left after the
                # step's last call.
                results = results + unattributed
            spawned = []
            for result in results:
                for ref in result.get("subagent_trajectory_ref") or []:
                    sub = self._resolve_subagent(ref, subagents)
                    if sub is None:
                        continue
                    if depth >= MAX_SUBAGENT_DEPTH:
                        logger.debug("Skipping subagent trajectory: too deeply nested")
                        continue
                    sub_start, sub_end = _time_range(sub, ext)
                    spawned.append(
                        (
                            sub,
                            sub_start if sub_start is not None else step_ns,
                            sub_end if sub_end is not None else step_ns,
                        )
                    )

            bounds = _bounds(lmnr)
            if bounds is not None:
                start_ns, end_ns = bounds
            else:
                starts = [t for _, t, _ in spawned if t is not None]
                ends = [t for _, _, t in spawned if t is not None]
                if step_ns is not None:
                    starts.append(step_ns)
                    ends.append(step_ns)
                start_ns = _not_before(
                    min(starts) if starts else None,
                    _shift(self._last_start, ORDER_GAP_NS),
                )
                end_ns = _not_before(max(ends) if ends else None, start_ns)
            attributes: dict[str, Any] = {}
            if isinstance(lmnr.get("span_id"), str):
                attributes[SOURCE_SPAN_ID] = lmnr["span_id"]
            span = cast(
                LaminarSpan,
                Laminar.start_span(
                    tool_call.get("function_name") or "tool",
                    input=_tool_call_arguments(tool_call, ext),
                    span_type="TOOL",
                    parent_span_context=parent,
                    attributes=attributes,
                    start_time=start_ns,
                ),
            )
            output = "\n".join(content_to_text(r.get("content")) for r in results)
            span.set_output(output if results else None)
            for sub, sub_start, sub_end in spawned:
                sub_start = _not_before(sub_start, start_ns)
                self._emit_subagent(
                    sub,
                    span.get_laminar_span_context(),
                    start_ns=sub_start,
                    end_ns=_not_before(sub_end, sub_start),
                    depth=depth + 1,
                    parent_ext=ext,
                )
            error = next(
                (e for r in results if (e := reader.result_error(r, ext)) is not None), None
            )
            if error is not None:
                _record_error(span, error, end_ns)
            span.end(end_time=end_ns)
            self.span_count += 1
            self._record_start(start_ns)
            if end_ns is not None:
                self._tools_end = _not_before(end_ns, self._tools_end)

    def _emit_subagent(
        self,
        trajectory: dict[str, Any],
        parent: LaminarSpanContext,
        start_ns: int | None,
        end_ns: int | None,
        depth: int,
        parent_ext: TrajectoryExtension | None,
    ) -> None:
        agent = trajectory.get("agent") or {}
        sub_ext = TrajectoryExtension.read(trajectory, parent=parent_ext)
        agent_span_id = reader.fields(trajectory, sub_ext).get("agent_span_id")
        span = cast(
            LaminarSpan,
            Laminar.start_span(
                agent.get("name") or "subagent",
                input=trajectory.get("trajectory_id") or trajectory.get("session_id"),
                parent_span_context=parent,
                attributes=(
                    {SOURCE_SPAN_ID: agent_span_id} if isinstance(agent_span_id, str) else None
                ),
                start_time=start_ns,
            ),
        )
        self.emit(
            trajectory,
            span.get_laminar_span_context(),
            start_ns=start_ns,
            depth=depth,
            parent_ext=parent_ext,
        )
        span.set_output(trajectory.get("final_metrics"))
        span.end(end_time=end_ns)
        self.span_count += 1

    def _resolve_subagent(
        self, ref: Any, subagents: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not isinstance(ref, dict):
            return None
        if ref.get("trajectory_path"):
            path = Path(ref["trajectory_path"])
            if not path.is_absolute() and self.base_dir is not None:
                path = self.base_dir / path
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.debug(f"Could not read subagent trajectory {path}: {e}")
                return None
        return subagents.get(ref.get("trajectory_id") or "")


def _time_range(
    trajectory: dict[str, Any], parent_ext: TrajectoryExtension | None = None
) -> tuple[int | None, int | None]:
    """Earliest and latest times of a trajectory's steps (and, with lmnr-atif
    fields, of their exact call and tool bounds)."""
    ext = TrajectoryExtension.read(trajectory, parent=parent_ext)
    times: list[int] = []
    for step in reader.steps_of(trajectory):
        candidates = [step.get("timestamp")]
        for obj in [step, *reader.dicts(step.get("tool_calls"))]:
            lmnr = reader.fields(obj, ext)
            candidates += [lmnr.get("start_time"), lmnr.get("end_time")]
        times += [t for value in candidates if (t := parse_timestamp_ns(value)) is not None]
    if not times:
        return None, None
    return min(times), max(times)


def _trajectory_attributes(trajectory: dict[str, Any], ext: TrajectoryExtension) -> dict[str, Any]:
    """Attributes for the agent span from the root trajectory's lmnr-atif fields."""
    lmnr = ext.fields
    attributes: dict[str, Any] = {"lmnr.trajectory.spec_version": ext.spec_version}
    trace_ids = [t for t in lmnr.get("trace_ids") or [] if isinstance(t, str)]
    if trace_ids:
        attributes["lmnr.trajectory.trace_ids"] = trace_ids
    for key in ("exported", "complete"):
        if isinstance(lmnr.get(key), bool):
            attributes[f"lmnr.trajectory.{key}"] = lmnr[key]
    totals = reader.fields(trajectory.get("final_metrics"), ext)
    for key in ("duration_ms", "llm_calls", "tool_calls", "errors"):
        value = totals.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            attributes[f"lmnr.trajectory.{key}"] = value
    return attributes


def emit_trajectory_spans(
    trajectory_path: Path,
    parent: LaminarSpanContext,
    start_ns: int | None = None,
) -> TrajectoryConverter:
    """Read an ATIF trajectory file and emit its spans.

    Returns the converter: `span_count` and `parent_attributes` (to set on the
    span the trajectory was emitted under).
    """
    trajectory = json.loads(trajectory_path.read_text())
    converter = TrajectoryConverter(base_dir=trajectory_path.parent)
    converter.emit(trajectory, parent, start_ns=start_ns)
    return converter


def final_agent_message(trajectory_path: Path) -> str | None:
    """Text of the last agent message in a trajectory, if any."""
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except Exception:
        return None
    for step in reversed(trajectory.get("steps") or []):
        if isinstance(step, dict) and step.get("source") == "agent":
            text = content_to_text(step.get("message"))
            if text:
                return text
    return None

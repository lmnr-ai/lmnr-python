"""Convert Harbor ATIF trajectories into Laminar spans.

Harbor agents record what they did as an ATIF trajectory (`agent/trajectory.json`
inside the trial directory). Agents usually run inside a sandbox where we can't
instrument their LLM calls, so the trajectory is the only record of the run we
get. This module replays it after the fact as LLM and TOOL spans under the
trial's agent span, using the timestamps recorded in the trajectory.

Works on plain dicts (the JSON file) so it doesn't depend on `harbor` itself.
"""

import datetime
import json
from pathlib import Path
from typing import Any

from lmnr.opentelemetry_lib.tracing.attributes import Attributes
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanContext

logger = get_default_logger(__name__)

# Guard against reference cycles between file-ref subagent trajectories.
MAX_SUBAGENT_DEPTH = 8


def parse_timestamp_ns(value: Any) -> int | None:
    """Parse an ISO 8601 string or a datetime into nanoseconds since the epoch."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, str):
        try:
            # Python 3.10 `fromisoformat` doesn't accept the `Z` suffix.
            dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1e9)


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
                parts.append(
                    f"[{part.get('type', 'attachment')}: {source.get('path')}]"
                )
        return "\n".join(parts)
    return str(content)


def _tool_call_to_openai(tool_call: dict[str, Any]) -> dict[str, Any]:
    arguments = tool_call.get("arguments")
    return {
        "id": tool_call.get("tool_call_id"),
        "type": "function",
        "function": {
            "name": tool_call.get("function_name"),
            "arguments": (
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            ),
        },
    }


def assistant_message(step: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content_to_text(step.get("message")),
    }
    if step.get("reasoning_content"):
        message["reasoning_content"] = step["reasoning_content"]
    if step.get("tool_calls"):
        message["tool_calls"] = [_tool_call_to_openai(tc) for tc in step["tool_calls"]]
    return message


def _observation_results(step: dict[str, Any]) -> list[dict[str, Any]]:
    observation = step.get("observation") or {}
    return [r for r in (observation.get("results") or []) if isinstance(r, dict)]


def observation_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Messages the agent saw after this step: tool results or plain feedback."""
    messages = []
    for result in _observation_results(step):
        content = content_to_text(result.get("content"))
        if result.get("source_call_id"):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["source_call_id"],
                    "content": content,
                }
            )
        else:
            messages.append({"role": "user", "content": content})
    return messages


def step_messages(step: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenAI-format messages that a step contributes to the conversation."""
    source = step.get("source")
    if source == "agent":
        return [assistant_message(step), *observation_messages(step)]
    role = "system" if source == "system" else "user"
    return [{"role": role, "content": content_to_text(step.get("message"))}]


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
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir
        self.span_count = 0

    def emit(
        self,
        trajectory: dict[str, Any],
        parent: LaminarSpanContext,
        start_ns: int | None = None,
        depth: int = 0,
    ) -> None:
        steps = [s for s in (trajectory.get("steps") or []) if isinstance(s, dict)]
        agent = trajectory.get("agent") or {}
        default_model = agent.get("model_name")
        subagents = {
            sub.get("trajectory_id"): sub
            for sub in (trajectory.get("subagent_trajectories") or [])
            if isinstance(sub, dict) and sub.get("trajectory_id")
        }

        history: list[dict[str, Any]] = []
        prev_ns = start_ns
        for step in steps:
            step_ns = parse_timestamp_ns(step.get("timestamp"))
            is_llm_step = (
                step.get("source") == "agent"
                and not step.get("is_copied_context")
                and step.get("llm_call_count") != 0
            )
            if is_llm_step:
                self._emit_llm_span(
                    step,
                    list(history),
                    parent,
                    start_ns=prev_ns if prev_ns is not None else step_ns,
                    end_ns=step_ns,
                    default_model=default_model,
                )
            if step.get("source") == "agent" and not step.get("is_copied_context"):
                self._emit_tool_spans(
                    step,
                    parent,
                    step_ns=step_ns if step_ns is not None else prev_ns,
                    subagents=subagents,
                    depth=depth,
                )
            history.extend(step_messages(step))
            if step_ns is not None:
                prev_ns = step_ns

    def _emit_llm_span(
        self,
        step: dict[str, Any],
        history: list[dict[str, Any]],
        parent: LaminarSpanContext,
        start_ns: int | None,
        end_ns: int | None,
        default_model: str | None,
    ) -> None:
        model_name = step.get("model_name") or default_model
        provider, model = split_model_name(model_name)
        attributes: dict[str, Any] = {}
        if provider:
            attributes[Attributes.PROVIDER.value] = provider
        if model:
            attributes[Attributes.REQUEST_MODEL.value] = model
            attributes[Attributes.RESPONSE_MODEL.value] = model

        metrics = step.get("metrics") or {}
        input_tokens = metrics.get("prompt_tokens")
        output_tokens = metrics.get("completion_tokens")
        if input_tokens is not None:
            attributes[Attributes.INPUT_TOKEN_COUNT.value] = input_tokens
        if output_tokens is not None:
            attributes[Attributes.OUTPUT_TOKEN_COUNT.value] = output_tokens
        if input_tokens is not None or output_tokens is not None:
            attributes[Attributes.TOTAL_TOKEN_COUNT.value] = (input_tokens or 0) + (
                output_tokens or 0
            )
        if metrics.get("cached_tokens") is not None:
            attributes["gen_ai.usage.cache_read_input_tokens"] = metrics[
                "cached_tokens"
            ]
        if metrics.get("cost_usd") is not None:
            attributes[Attributes.TOTAL_COST.value] = metrics["cost_usd"]
        if step.get("step_id") is not None:
            attributes["harbor.step_id"] = step["step_id"]

        span = Laminar.start_span(
            model or "llm",
            input=history,
            span_type="LLM",
            parent_span_context=parent,
            attributes=attributes,
            start_time=start_ns,
        )
        span.set_output([assistant_message(step)])
        span.end(end_time=end_ns)
        self.span_count += 1

    def _emit_tool_spans(
        self,
        step: dict[str, Any],
        parent: LaminarSpanContext,
        step_ns: int | None,
        subagents: dict[str, dict[str, Any]],
        depth: int,
    ) -> None:
        results_by_call: dict[str, list[dict[str, Any]]] = {}
        for result in _observation_results(step):
            if result.get("source_call_id"):
                results_by_call.setdefault(result["source_call_id"], []).append(result)

        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            results = results_by_call.get(tool_call.get("tool_call_id"), [])
            spawned = []
            for result in results:
                for ref in result.get("subagent_trajectory_ref") or []:
                    sub = self._resolve_subagent(ref, subagents)
                    if sub is None:
                        continue
                    if depth >= MAX_SUBAGENT_DEPTH:
                        logger.debug("Skipping subagent trajectory: too deeply nested")
                        continue
                    sub_start, sub_end = _time_range(sub)
                    spawned.append(
                        (
                            sub,
                            sub_start if sub_start is not None else step_ns,
                            sub_end if sub_end is not None else step_ns,
                        )
                    )

            starts = [t for _, t, _ in spawned if t is not None]
            ends = [t for _, _, t in spawned if t is not None]
            if step_ns is not None:
                starts.append(step_ns)
                ends.append(step_ns)
            start_ns = min(starts) if starts else None
            span = Laminar.start_span(
                tool_call.get("function_name") or "tool",
                input=tool_call.get("arguments"),
                span_type="TOOL",
                parent_span_context=parent,
                start_time=start_ns,
            )
            output = "\n".join(content_to_text(r.get("content")) for r in results)
            span.set_output(output if results else None)
            for sub, sub_start, sub_end in spawned:
                self._emit_subagent(
                    sub,
                    span.get_laminar_span_context(),
                    start_ns=sub_start,
                    end_ns=sub_end,
                    depth=depth + 1,
                )
            span.end(end_time=max(ends) if ends else None)
            self.span_count += 1

    def _emit_subagent(
        self,
        trajectory: dict[str, Any],
        parent: LaminarSpanContext,
        start_ns: int | None,
        end_ns: int | None,
        depth: int,
    ) -> None:
        agent = trajectory.get("agent") or {}
        span = Laminar.start_span(
            agent.get("name") or "subagent",
            input=trajectory.get("trajectory_id") or trajectory.get("session_id"),
            parent_span_context=parent,
            start_time=start_ns,
        )
        self.emit(
            trajectory,
            span.get_laminar_span_context(),
            start_ns=start_ns,
            depth=depth,
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
        return subagents.get(ref.get("trajectory_id"))


def _time_range(trajectory: dict[str, Any]) -> tuple[int | None, int | None]:
    """Earliest and latest step timestamps of a trajectory."""
    times = [
        t
        for step in trajectory.get("steps") or []
        if isinstance(step, dict)
        and (t := parse_timestamp_ns(step.get("timestamp"))) is not None
    ]
    if not times:
        return None, None
    return min(times), max(times)


def emit_trajectory_spans(
    trajectory_path: Path,
    parent: LaminarSpanContext,
    start_ns: int | None = None,
) -> int:
    """Read an ATIF trajectory file and emit its spans. Returns the span count."""
    trajectory = json.loads(trajectory_path.read_text())
    converter = TrajectoryConverter(base_dir=trajectory_path.parent)
    converter.emit(trajectory, parent, start_ns=start_ns)
    return converter.span_count


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

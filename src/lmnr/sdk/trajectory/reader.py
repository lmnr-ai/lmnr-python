"""Read lmnr-atif fields (`spec/lmnr-atif/lmnr-atif.md`) from an ATIF trajectory.

Works on plain dicts (the parsed JSON file), so it depends on neither `harbor`
nor the tracing stack. Every helper takes the trajectory's
`TrajectoryExtension`, or None for a plain ATIF file, and falls back to what
plain ATIF says when there is no compatible extension. That makes this module
the single place that decides whether `extra.lmnr` can be trusted.

Messages returned here are `LmnrMessage` dicts (spec §3): `role`, `content`
(a string or ATIF content parts), plus `reasoning_content` / `tool_calls` on
assistant messages and `tool_call_id` on tool messages. Tool calls keep ATIF's
shape.
"""

from __future__ import annotations

import re
from typing import Any, cast

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.trajectory.spec import MIN_ATIF_VERSION, NAMESPACE, SPEC_VERSION, DefKind

logger = get_default_logger(__name__)

Message = dict[str, Any]

_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
_ATIF_VERSION = re.compile(r"^ATIF-v(\d+)\.(\d+)$")

# Spec versions already reported as unreadable, so each is logged once.
_reported_versions: set[str] = set()


def as_dict(value: Any) -> dict[str, Any]:
    """`value` if it is a JSON object, else {}."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def dicts(value: Any) -> list[dict[str, Any]]:
    """The JSON objects in `value` if it is a list, else []."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], v) for v in cast(list[Any], value) if isinstance(v, dict)]


def _lmnr_extra(obj: Any) -> dict[str, Any]:
    """`obj["extra"]["lmnr"]` as a dict, or {} when absent or malformed."""
    return as_dict(as_dict(as_dict(obj).get("extra")).get(NAMESPACE))


def _parse_spec_version(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = _SEMVER.match(value)
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def is_compatible(version: Any, reference: str = SPEC_VERSION) -> bool:
    """Whether a reader of `reference` can read a file of `version` (spec §2).

    Same major version and, while the major is 0, the same minor version.
    """
    ours = _parse_spec_version(reference)
    theirs = _parse_spec_version(version)
    if ours is None or theirs is None or ours[0] != theirs[0]:
        return False
    return ours[0] != 0 or ours[1] == theirs[1]


def _atif_version_ok(trajectory: dict[str, Any]) -> bool:
    value = trajectory.get("schema_version")
    if value is None:
        # Harbor's model defaults a missing schema_version to the latest.
        return True
    match = _ATIF_VERSION.match(value) if isinstance(value, str) else None
    if match is None:
        return False
    return (int(match[1]), int(match[2])) >= MIN_ATIF_VERSION


class TrajectoryExtension:
    """The lmnr-atif fields of one trajectory document.

    Build it with `read`, which returns None for plain ATIF. Embedded or
    file-referenced subagents pass their parent, which they inherit a missing
    `spec_version` from and resolve `$ref`s against when their own `defs`
    don't have the key (spec §6.6).
    """

    def __init__(
        self,
        fields: dict[str, Any],
        spec_version: str,
        parent: TrajectoryExtension | None = None,
    ):
        self.fields: dict[str, Any] = fields
        self.spec_version: str = spec_version
        self.parent: TrajectoryExtension | None = parent

    @classmethod
    def read(
        cls, trajectory: Any, parent: TrajectoryExtension | None = None
    ) -> TrajectoryExtension | None:
        if not isinstance(trajectory, dict):
            return None
        trajectory = cast(dict[str, Any], trajectory)
        fields = _lmnr_extra(trajectory)
        if not fields and parent is None:
            return None
        version = fields.get("spec_version")
        if version is None and parent is not None:
            version = parent.spec_version
        if not is_compatible(version):
            if isinstance(version, str) and version not in _reported_versions:
                _reported_versions.add(version)
                logger.info(
                    f"Ignoring Laminar fields of an lmnr-atif {version} trajectory: "
                    + f"this SDK reads lmnr-atif {SPEC_VERSION}"
                )
            return None
        if not _atif_version_ok(trajectory):
            return None
        return cls(fields, str(version), parent)

    def resolve(self, value: Any, kind: DefKind) -> Any:
        """Resolve an override value, following a `{"$ref": key}` into `defs`.

        Returns None when the override is absent or its `$ref` doesn't
        resolve, which readers treat the same way.
        """
        ref = as_dict(value).get("$ref")
        if ref is None:
            return value
        ext: TrajectoryExtension | None = self
        while ext is not None:
            table = as_dict(as_dict(ext.fields.get("defs")).get(kind))
            if isinstance(ref, str) and ref in table:
                return table[ref]
            ext = ext.parent
        logger.debug(f"Unresolved {kind} $ref {ref!r} in trajectory")
        return None

    def override(self, step: dict[str, Any], key: str, kind: DefKind) -> Any:
        """A step's resolved `extra.lmnr.<key>` override, or None if absent."""
        return self.resolve(_lmnr_extra(step).get(key), kind)


def fields(obj: Any, ext: TrajectoryExtension | None) -> dict[str, Any]:
    """`obj.extra.lmnr`, but only when the trajectory's extension is readable."""
    return _lmnr_extra(obj) if ext is not None else {}


def steps_of(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return dicts(trajectory.get("steps"))


def leading_system_count(steps: list[dict[str, Any]]) -> int:
    """Number of `system` steps before the first `user` or `agent` step."""
    count = 0
    for step in steps:
        if step.get("source") != "system":
            break
        count += 1
    return count


def _message_content(value: Any) -> Any:
    return "" if value is None else value


def _system_message(step: dict[str, Any], ext: TrajectoryExtension | None) -> Message:
    role = fields(step, ext).get("role")
    return {
        "role": role if isinstance(role, str) and role else "system",
        "content": _message_content(step.get("message")),
    }


def system_prompt(
    steps: list[dict[str, Any]], index: int, ext: TrajectoryExtension | None
) -> list[Message]:
    """System messages agent step `index` was configured with (spec §6.3)."""
    override = ext.override(steps[index], "system_prompt", "system_prompts") if ext else None
    if override is not None:
        if override == "" or override == []:
            return []
        return [{"role": "system", "content": override}]
    return [_system_message(s, ext) for s in steps[: leading_system_count(steps)]]


def tool_definitions(
    trajectory: dict[str, Any], step: dict[str, Any], ext: TrajectoryExtension | None
) -> list[dict[str, Any]] | None:
    """Tools offered to an agent step (spec §6.5); None when unknown."""
    override = ext.override(step, "tool_definitions", "tool_definitions") if ext else None
    if isinstance(override, list):
        return dicts(override)
    default = as_dict(trajectory.get("agent")).get("tool_definitions")
    return dicts(default) if isinstance(default, list) else None


def is_synthetic(step: dict[str, Any], ext: TrajectoryExtension | None) -> bool:
    return fields(step, ext).get("synthetic") is True


def observation_results(step: dict[str, Any]) -> list[dict[str, Any]]:
    return dicts(as_dict(step.get("observation")).get("results"))


def step_messages(step: dict[str, Any], ext: TrajectoryExtension | None) -> list[Message]:
    """Messages a step adds to the conversation (spec §6.4)."""
    source = step.get("source")
    if source == "system":
        return [_system_message(step, ext)]
    if source != "agent":
        return [{"role": "user", "content": _message_content(step.get("message"))}]
    if is_synthetic(step, ext):
        return []

    assistant: Message = {"role": "assistant", "content": _message_content(step.get("message"))}
    if step.get("reasoning_content"):
        assistant["reasoning_content"] = step["reasoning_content"]
    tool_calls = dicts(step.get("tool_calls"))
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    messages = [assistant]
    for result in observation_results(step):
        content = _message_content(result.get("content"))
        if result.get("source_call_id"):
            messages.append(
                {"role": "tool", "tool_call_id": result["source_call_id"], "content": content}
            )
        else:
            messages.append({"role": "user", "content": content})
    return messages


def input_messages(
    steps: list[dict[str, Any]], index: int, ext: TrajectoryExtension | None
) -> list[Message]:
    """The full input of agent step `index`: system prompt, then conversation.

    The conversation is the step's own `input_messages` override, or the latest
    earlier override plus every step since, or the whole history (spec §6.4).
    """
    prompt = system_prompt(steps, index, ext)
    start = leading_system_count(steps)
    history: list[Message] = []
    if ext is not None:
        for j in range(index, start - 1, -1):
            if steps[j].get("source") != "agent":
                continue
            override = ext.override(steps[j], "input_messages", "input_messages")
            if isinstance(override, list):
                if j == index:
                    return prompt + [dict(m) for m in dicts(override)]
                history = [dict(m) for m in dicts(override)]
                start = j
                break
    history += [message for step in steps[start:index] for message in step_messages(step, ext)]
    return prompt + history


def _error(value: Any) -> dict[str, str]:
    error = as_dict(value)
    result = {
        "type": str(error.get("type") or "Error"),
        "message": str(error.get("message") or ""),
    }
    if error.get("stacktrace"):
        result["stacktrace"] = str(error["stacktrace"])
    return result


def step_error(step: dict[str, Any], ext: TrajectoryExtension | None) -> dict[str, str] | None:
    """`{type, message[, stacktrace]}` if the step's LLM call failed, else None."""
    lmnr = fields(step, ext)
    if lmnr.get("status") != "error":
        return None
    return _error(lmnr.get("error"))


def result_error(result: dict[str, Any], ext: TrajectoryExtension | None) -> dict[str, str] | None:
    """`{type, message[, stacktrace]}` if a tool call failed, else None (spec §7).

    Also understands the flags Harbor's own agents write, so it works on
    plain ATIF from claude-code (`tool_result_is_error`) and kimi (`is_error`).
    """
    lmnr = fields(result, ext)
    status = lmnr.get("status")
    if status == "error":
        return _error(lmnr.get("error"))
    if status == "ok":
        return None
    extra = as_dict(result.get("extra"))
    if extra.get("tool_result_is_error") is True or extra.get("is_error") is True:
        return {"type": "ToolError", "message": ""}
    return None


def usage_extension(metrics: Any, ext: TrajectoryExtension | None) -> dict[str, int | float]:
    """Usage fields without an ATIF core field (spec §4.7), with fallbacks.

    Keys: `reasoning_tokens`, `cache_creation_tokens`, `input_cost_usd`,
    `output_cost_usd`. Only present ones are returned.
    """
    bare = as_dict(as_dict(metrics).get("extra"))
    lmnr = fields(metrics, ext)
    candidates: dict[str, tuple[Any, ...]] = {
        "reasoning_tokens": (lmnr.get("reasoning_tokens"), bare.get("reasoning_tokens")),
        "cache_creation_tokens": (
            lmnr.get("cache_creation_tokens"),
            bare.get("cache_creation_input_tokens"),
        ),
        "input_cost_usd": (lmnr.get("input_cost_usd"),),
        "output_cost_usd": (lmnr.get("output_cost_usd"),),
    }
    usage: dict[str, int | float] = {}
    for key, values in candidates.items():
        for value in values:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = value
                break
    return usage

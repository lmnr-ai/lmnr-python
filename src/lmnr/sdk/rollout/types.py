from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired


class LanguageModelTextBlock(TypedDict):
    type: Literal["text"]
    text: str


class LanguageModelToolDefinitionOverride(TypedDict):
    name: str
    description: str | None
    parameters: dict[str, Any]


class SingleRolloutPathOverride(TypedDict):
    system: str | list[LanguageModelTextBlock]
    tools: list[LanguageModelToolDefinitionOverride]


class RolloutPathOverride(TypedDict):
    system: str | list[LanguageModelTextBlock]
    tools: list[LanguageModelToolDefinitionOverride]


class RolloutRunEventData(TypedDict):
    trace_id: str | None
    path_to_count: dict[str, int]
    args: dict[str, Any] | list[Any]
    overrides: dict[str, RolloutPathOverride]


class RolloutRunEvent(TypedDict):
    event_type: Literal["run"]
    data: RolloutRunEventData


class RolloutParam(TypedDict):
    name: str
    type: NotRequired[str]
    required: NotRequired[bool]
    nested: NotRequired[list["RolloutParam"]]
    default: NotRequired[str]


class FunctionMetadata(TypedDict):
    name: str
    params: list[RolloutParam]


class CachedSpan(TypedDict):
    name: str
    input: str
    output: str
    attributes: dict[str, Any]


class CacheMetadata(TypedDict):
    pathToCount: dict[str, int]
    overrides: NotRequired[dict[str, SingleRolloutPathOverride]]


class CacheServerResponse(TypedDict):
    pathToCount: dict[str, int]
    overrides: NotRequired[dict[str, SingleRolloutPathOverride]]
    span: NotRequired[CachedSpan]

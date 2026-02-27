from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired


class AnthropicUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: NotRequired[int]
    cache_read_input_tokens: NotRequired[int]


class AnthropicContentBlock(TypedDict, total=False):
    type: Literal["text", "tool_use", "thinking"]
    text: str
    id: str
    name: str
    input: str | dict[str, Any]
    thinking: str


class AnthropicResponseMessage(TypedDict, total=False):
    id: str
    model: str
    role: Literal["assistant"]
    content: list[AnthropicContentBlock]
    type: Literal["message"]
    usage: AnthropicUsage
    stop_reason: NotRequired[str | None]
    stop_sequence: NotRequired[str | None]


class _FunctionToolCall(TypedDict):
    function_name: str
    arguments: dict[str, Any] | None


class ToolCall(TypedDict):
    """Represents a tool call in the AI model."""

    id: str
    function: _FunctionToolCall
    type: Literal["function"]


class CompletionMessage(TypedDict):
    """Represents a message in the AI model."""

    content: Any
    role: Literal["assistant"]


@dataclass
class MessageEvent:
    """Represents an input event for the AI model."""

    content: Any
    role: str = "user"
    tool_calls: NotRequired[list[ToolCall] | None] = None


@dataclass
class ChoiceEvent:
    """Represents a completion event for the AI model."""

    index: int
    message: CompletionMessage
    finish_reason: str = "unknown"
    tool_calls: NotRequired[list[ToolCall] | None] = None

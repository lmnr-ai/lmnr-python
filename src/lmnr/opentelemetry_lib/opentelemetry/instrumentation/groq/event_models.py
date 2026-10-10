from dataclasses import dataclass
from typing import Any, Literal, TypedDict


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
    tool_calls: list[ToolCall] | None = None


@dataclass
class ChoiceEvent:
    """Represents a completion event for the AI model."""

    index: int
    message: CompletionMessage
    finish_reason: str = "unknown"
    tool_calls: list[ToolCall] | None = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

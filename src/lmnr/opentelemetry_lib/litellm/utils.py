import re
from pydantic import BaseModel
from opentelemetry.sdk.trace import Span
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypedDict


class ToolDefinition(TypedDict):
    name: str | None
    description: str | None
    parameters: dict | None


def model_as_dict(model: BaseModel | dict) -> dict:
    if isinstance(model, BaseModel) and hasattr(model, "model_dump"):
        return model.model_dump()
    elif isinstance(model, dict):
        return model
    else:
        return dict(model)


def set_span_attribute(span: Span, key: str, value: AttributeValue | None):
    if value is None or value == "":
        return
    span.set_attribute(key, value)


def get_tool_definition(tool: dict) -> ToolDefinition:
    parameters = None
    description = None
    name = (tool.get("function") or {}).get("name") or tool.get("name")
    if tool.get("type") == "function":
        function = tool.get("function") or {}
        parameters = function.get("parameters") or tool.get("parameters")
        description = function.get("description") or tool.get("description")
    elif isinstance(tool.get("type"), str) and tool.get("type").startswith("computer"):
        # Anthropic beta computer tools
        # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool

        # OpenAI computer use API
        # https://platform.openai.com/docs/guides/tools-computer-use
        if not name:
            name = tool.get("type")

        parameters = {}
        tool_parameters = (tool.get("function") or {}).get("parameters") or {}
        # Anthropic
        display_width_px = tool_parameters.get("display_width_px") or tool.get(
            "display_width_px"
        )
        display_height_px = tool_parameters.get("display_height_px") or tool.get(
            "display_height_px"
        )
        display_number = tool_parameters.get("display_number") or tool.get(
            "display_number"
        )
        if display_width_px:
            parameters["display_width_px"] = display_width_px
        if display_height_px:
            parameters["display_height_px"] = display_height_px
        if display_number:
            parameters["display_number"] = display_number
        # OpenAI
        display_width = tool_parameters.get("display_width") or tool.get(
            "display_width"
        )
        display_height = tool_parameters.get("display_height") or tool.get(
            "display_height"
        )
        environment = tool_parameters.get("environment") or tool.get("environment")
        if display_width:
            parameters["display_width"] = display_width
        if display_height:
            parameters["display_height"] = tool.get("display_height")
        if environment:  # Literal['browser', 'mac', 'windows', 'ubuntu']
            parameters["environment"] = environment

    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
    )


def is_validator_iterator(content):
    """
    Some OpenAI objects contain fields typed as Iterable, which pydantic
    internally converts to a ValidatorIterator, and they cannot be trivially
    serialized without consuming the iterator to, for example, a list.

    See: https://github.com/pydantic/pydantic/issues/9541#issuecomment-2189045051
    """
    return re.search(r"pydantic.*ValidatorIterator'>$", str(type(content)))

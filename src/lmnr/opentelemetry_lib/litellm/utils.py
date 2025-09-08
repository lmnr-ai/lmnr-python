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
    if tool.get("type") == "function":
        parameters = tool.get("parameters")
    elif isinstance(tool.get("type"), str) and tool.get("type").startswith("computer"):
        # Anthropic beta computer tools
        # https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/computer-use-tool

        # OpenAI computer use API
        # https://platform.openai.com/docs/guides/tools-computer-use

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
        name=tool.get("name") or tool.get("function", {}).get("name"),
        description=tool.get("description"),
        parameters=parameters,
    )

import base64
from collections import defaultdict
import logging
import traceback
from typing_extensions import TypedDict

from .config import (
    Config,
)
from google.genai import types
from google.genai._common import BaseModel
import pydantic
from opentelemetry.trace import Span
from typing import Any, Literal


class ToolCall(pydantic.BaseModel):
    name: str | None = pydantic.Field(default=None)
    id: str | None = pydantic.Field(default=None)
    arguments: dict[str, Any] = pydantic.Field(default={})


class ImageUrlInner(pydantic.BaseModel):
    url: str = pydantic.Field(default="")


class ImageUrl(pydantic.BaseModel):
    type: Literal["image_url"] = pydantic.Field(default="image_url")
    image_url: ImageUrlInner = pydantic.Field(default=ImageUrlInner())


class ProcessedContentPart(pydantic.BaseModel):
    content: str | None = pydantic.Field(default=None)
    function_call: ToolCall | None = pydantic.Field(default=None)
    image_url: ImageUrl | None = pydantic.Field(default=None)


class ProcessChunkResult(TypedDict):
    role: str
    model_version: str | None


def merge_text_parts(
    parts: list[types.PartDict | types.File | types.Part | str],
) -> list[types.Part]:
    if not parts:
        return []

    merged_parts: list[types.Part] = []
    accumulated_text = ""

    for part in parts:
        # Handle string input - treat as text
        if isinstance(part, str):
            accumulated_text += part
        # Handle File objects - they are not text, so don't merge
        elif isinstance(part, types.File):
            # Flush any accumulated text first
            if accumulated_text:
                merged_parts.append(types.Part(text=accumulated_text))
                accumulated_text = ""
            # Add the File as-is (wrapped in a Part if needed)
            # Note: File objects should be passed through as-is in the original part
            merged_parts.append(part)
        # Handle Part and PartDict (dicts)
        else:
            part_dict = to_dict(part)

            # Check if this is a text part
            if part_dict.get("text") is not None:
                accumulated_text += part_dict.get("text")
            else:
                # Non-text part (inline_data, function_call, etc.)
                # Flush any accumulated text first
                if accumulated_text:
                    merged_parts.append(types.Part(text=accumulated_text))
                    accumulated_text = ""

                # Add the non-text part as-is
                if isinstance(part, types.Part):
                    merged_parts.append(part)
                elif isinstance(part, dict):
                    # Convert dict to Part object
                    merged_parts.append(types.Part(**part_dict))

    # Don't forget to add any remaining accumulated text
    if accumulated_text:
        merged_parts.append(types.Part(text=accumulated_text))

    return merged_parts


def set_span_attribute(span: Span, name: str, value: Any):
    if value is not None and value != "":
        span.set_attribute(name, value)
    return


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "Laminar failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def to_dict(obj: BaseModel | pydantic.BaseModel | dict) -> dict[str, Any]:
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, pydantic.BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return obj
        else:
            return dict(obj)
    except Exception as e:
        logging.error(f"Error converting to dict: {obj}, error: {e}")
        return dict(obj)


def get_content(
    content: (
        ProcessedContentPart | dict | list[ProcessedContentPart | dict] | str | None
    ),
) -> dict | list[dict] | None:
    if isinstance(content, dict):
        return content.get("content") or content.get("image_url")
    if isinstance(content, ProcessedContentPart):
        if content.content and isinstance(content.content, str):
            return {
                "type": "text",
                "text": content.content,
            }
        elif content.image_url:
            return content.image_url.model_dump()
        else:
            return None
    elif isinstance(content, list):
        contents_list = [get_content(item) for item in content]
        return [item for item in contents_list if item is not None]
    elif isinstance(content, str):
        return {
            "type": "text",
            "text": content,
        }
    else:
        return None


def process_content_union(
    content: types.ContentUnion | types.ContentUnionDict,
) -> ProcessedContentPart | dict | list[ProcessedContentPart | dict] | None:
    if isinstance(content, types.Content):
        parts = to_dict(content).get("parts", [])
        return [_process_part(part) for part in parts]
    elif isinstance(content, list):
        return [_process_part_union(item) for item in content]
    elif isinstance(content, (types.Part, types.File, str)):
        return _process_part_union(content)
    elif isinstance(content, dict):
        if "parts" in content:
            return [_process_part_union(item) for item in content.get("parts", [])]
        else:
            # Assume it's PartDict
            return _process_part_union(content)
    else:
        return None


def _process_part_union(
    content: types.PartDict | types.File | types.Part | str,
) -> ProcessedContentPart | dict | None:
    if isinstance(content, str):
        return ProcessedContentPart(content=content)
    elif isinstance(content, types.File):
        content_dict = to_dict(content)
        name = (
            content_dict.get("name")
            or content_dict.get("display_name")
            or content_dict.get("uri")
        )
        return ProcessedContentPart(content=f"files/{name}")
    elif isinstance(content, (types.Part, dict)):
        return _process_part(content)
    else:
        return None


def _process_part(
    content: types.Part,
) -> ProcessedContentPart | dict | None:
    part_dict = to_dict(content)
    if part_dict.get("inline_data"):
        blob = to_dict(part_dict.get("inline_data"))
        if blob.get("mime_type", "").startswith("image/"):
            return _process_image_item(blob)
        else:
            # currently, only images are supported
            return ProcessedContentPart(
                content=blob.get("mime_type") or "unknown_media"
            )
    elif function_call := part_dict.get("function_call"):
        function_call_dict = to_dict(function_call)
        return ProcessedContentPart(
            function_call=ToolCall(
                name=function_call_dict.get("name"),
                id=function_call_dict.get("id"),
                arguments=function_call_dict.get("args", {}),
            )
        )
    elif part_dict.get("text") is not None:
        return ProcessedContentPart(content=part_dict.get("text"))
    else:
        return None


def role_from_content_union(
    content: types.ContentUnion | types.ContentUnionDict,
) -> str | None:
    role = None
    if isinstance(content, types.Content):
        role = to_dict(content).get("role")
    elif isinstance(content, list) and len(content) > 0:
        role = role_from_content_union(content[0])
    elif isinstance(content, dict):
        role = content.get("role")
    else:
        return None
    return role
    # return "assistant" if role == "model" else role


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _process_image_item(blob: dict[str, Any]) -> ProcessedContentPart | dict | None:
    # Convert to openai format, so backends can handle it
    data = blob.get("data")
    encoded_data = (
        base64.b64encode(data).decode("utf-8") if isinstance(data, bytes) else data
    )
    mime_type = blob.get("mime_type", "image/unknown")
    image_type = mime_type.split("/")[1] if "/" in mime_type else "unknown"

    return (
        ProcessedContentPart(
            image_url=ImageUrl(
                image_url=ImageUrlInner(
                    url=f"data:image/{image_type};base64,{encoded_data}",
                )
            )
        )
        if Config.convert_image_to_openai_format
        else blob
    )


@dont_throw
def process_stream_chunk(
    chunk: types.GenerateContentResponse,
    existing_role: str,
    existing_model_version: str | None,
    # ============================== #
    # mutable states, passed by reference
    aggregated_usage_metadata: defaultdict[str, int],
    final_parts: list[types.Part | None],
    # ============================== #
) -> ProcessChunkResult:
    role = existing_role
    model_version = existing_model_version

    if chunk.model_version:
        model_version = chunk.model_version

    # Currently gemini throws an error if you pass more than one candidate
    # with streaming
    if chunk.candidates and len(chunk.candidates) > 0 and chunk.candidates[0].content:
        final_parts += chunk.candidates[0].content.parts or []
        role = chunk.candidates[0].content.role or role
    if chunk.usage_metadata:
        usage_dict = to_dict(chunk.usage_metadata)
        # prompt token count is sent in every chunk
        # (and is less by 1 in the last chunk, so we set it once);
        # total token count in every chunk is greater by prompt token count than it should be,
        # thus this awkward logic here
        if aggregated_usage_metadata.get("prompt_token_count") is None:
            # or 0, not .get(key, 0), because sometimes the value is explicitly None
            aggregated_usage_metadata["prompt_token_count"] = (
                usage_dict.get("prompt_token_count") or 0
            )
            aggregated_usage_metadata["total_token_count"] = (
                usage_dict.get("total_token_count") or 0
            )
        aggregated_usage_metadata["candidates_token_count"] += (
            usage_dict.get("candidates_token_count") or 0
        )
        aggregated_usage_metadata["total_token_count"] += (
            usage_dict.get("candidates_token_count") or 0
        )
    return ProcessChunkResult(
        role=role,
        model_version=model_version,
    )

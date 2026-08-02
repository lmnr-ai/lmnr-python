from collections import defaultdict
import traceback
from typing_extensions import TypedDict

from .config import (
    Config,
)
from google.genai import types
from google.genai._common import BaseModel
import pydantic
from opentelemetry.trace import Span
from typing import Any

from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)


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
    func_logger = get_default_logger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            func_logger.debug(
                "Laminar failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


def to_dict(
    obj: BaseModel | pydantic.BaseModel | dict, pydantic_kwargs: dict[str, Any] = {}
) -> dict[str, Any]:
    try:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, pydantic.BaseModel):
            return obj.model_dump(**pydantic_kwargs)
        elif isinstance(obj, dict):
            return obj
        elif obj is None:
            return {}
        else:
            return dict(obj)
    except Exception as e:
        logger.debug(f"Error converting to dict: {obj}, error: {e}")
        return dict(obj)


def with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


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


def is_model_valid(obj: Any, model: BaseModel) -> bool:
    try:
        model.model_validate(obj)
        return True
    except Exception:
        return False


def strip_none_values(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        k: strip_none_values(v) if isinstance(v, dict) else v
        for k, v in obj.items()
        if v is not None
    }


def model_to_json_safe_dict(model: pydantic.BaseModel, **kwargs) -> dict[str, Any]:
    """Dump a pydantic model to a dict safe to hand to `json_dumps`.

    Deliberately `mode="python"` rather than `mode="json"`: google-genai's models
    set `ser_json_bytes="base64"`, which is pydantic's URL-SAFE alphabet, so json
    mode emits `-`/`_`. Consumers decode with `base64.b64decode`, which defaults
    to `validate=False` and silently DROPS the out-of-alphabet characters instead
    of raising — every following byte shifts and an image decodes to garbage.
    Python mode keeps `bytes` intact so `json_dumps` encodes them as standard
    base64 on the way out.
    """
    return model.model_dump(mode="python", **kwargs)


def part_to_dict(part) -> dict[str, Any]:
    """Convert a Part-like object to a serializable dict."""
    if isinstance(part, str):
        return {"text": part}
    if isinstance(part, dict):
        return strip_none_values(part)
    if hasattr(part, "model_dump"):
        return model_to_json_safe_dict(part, exclude_unset=True, exclude_none=True)
    return strip_none_values(to_dict(part))


def content_union_to_dict(
    content: types.ContentUnion | types.ContentUnionDict,
    default_role: str = "user",
) -> dict[str, Any]:
    """Convert a ContentUnion to a Gemini Content dict with 'parts' and 'role'."""
    if isinstance(content, types.Content):
        result = model_to_json_safe_dict(content, exclude_unset=True, exclude_none=True)
        if "role" not in result:
            result["role"] = default_role
        return result
    elif isinstance(content, str):
        return {"role": default_role, "parts": [{"text": content}]}
    elif isinstance(content, dict):
        if "parts" in content:
            result = dict(content)
            result["parts"] = [part_to_dict(p) for p in result["parts"]]
            if "role" not in result:
                result["role"] = default_role
            return result
        else:
            return {"role": default_role, "parts": [part_to_dict(content)]}
    elif isinstance(content, list):
        return {"role": default_role, "parts": [part_to_dict(p) for p in content]}
    else:
        return {"role": default_role, "parts": [part_to_dict(content)]}

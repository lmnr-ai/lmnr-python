from collections import defaultdict
from typing import Any, cast

import pydantic
from google.genai import types
from google.genai._common import BaseModel
from typing_extensions import TypedDict, TypeVar

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import dont_throw, to_dict
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)
T = TypeVar("T")

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
            file = cast(types.File, part)
            merged_parts.append(
                types.Part(
                    file_data=types.FileData(
                        display_name=file.display_name,
                        file_uri=file.uri,
                        mime_type=file.mime_type
                    )
                )
            )
        # Handle Part and PartDict (dicts)
        else:
            part_dict = to_dict(cast(dict[str, Any] | BaseModel, part))  # pyright: ignore[reportExplicitAny]

            # Check if this is a text part
            if part_dict.get("text") is not None:
                accumulated_text += cast(str, part_dict.get("text") or "")
            else:
                # Non-text part (inline_data, function_call, etc.)
                # Flush any accumulated text first
                if accumulated_text:
                    merged_parts.append(types.Part(text=accumulated_text))
                    accumulated_text = ""

                # Add the non-text part as-is
                if isinstance(part, types.Part):
                    merged_parts.append(part)
                elif isinstance(part, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                    # Convert dict to Part object
                    merged_parts.append(types.Part(**part_dict))  # pyright: ignore[reportAny]

    # Don't forget to add any remaining accumulated text
    if accumulated_text:
        merged_parts.append(types.Part(text=accumulated_text))

    return merged_parts


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


def is_model_valid(obj: Any, model: BaseModel) -> bool:  # pyright: ignore[reportAny, reportExplicitAny]
    try:
        _validated_model = model.model_validate(obj)
        return True
    except Exception:
        return False


def strip_none_values(obj: dict[str, Any]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    return {
        k: strip_none_values(v) if isinstance(v, dict) else v  # pyright: ignore[reportUnknownArgumentType]
        for k, v in obj.items()  # pyright: ignore[reportAny]
        if v is not None
    }


def model_to_json_safe_dict(model: pydantic.BaseModel, **kwargs: Any) -> dict[str, Any]:  # pyright: ignore[reportAny, reportExplicitAny]
    """Dump a pydantic model to a dict safe to hand to `json_dumps`.

    Deliberately `mode="python"` rather than `mode="json"`: google-genai's models
    set `ser_json_bytes="base64"`, which is pydantic's URL-SAFE alphabet, so json
    mode emits `-`/`_`. Consumers decode with `base64.b64decode`, which defaults
    to `validate=False` and silently DROPS the out-of-alphabet characters instead
    of raising — every following byte shifts and an image decodes to garbage.
    Python mode keeps `bytes` intact so `json_dumps` encodes them as standard
    base64 on the way out.

    `mode="json"` cannot be kept here, and the alternatives were checked:
    pydantic offers no standard-base64 setting (`ser_json_bytes` takes only
    `"base64"` (URL-safe) / `"hex"` / `"utf8"`, and `utf8` raises on binary), a
    per-call `context=` does not reach it, `TypeAdapter(config=...)` is rejected
    for BaseModel types, and repairing the alphabet after the fact would have to
    guess which strings came from bytes — a blind `-`/`_` swap corrupts ordinary
    text like "well-known B-tree".

    Python mode is also strictly MORE robust than json mode for the `Any`-typed
    `function_call.args` / `function_response.response` fields: json mode RAISES
    `PydanticSerializationError` on a value it doesn't recognise, and since the
    callers are `@dont_throw`, that dropped the entire `gen_ai.input.messages`
    attribute — every message in the conversation, not just the odd value.
    Python mode hands the value to `json_dumps`, which degrades just that leaf.
    """
    return model.model_dump(mode="python", **kwargs)  # pyright: ignore[reportAny]


def part_to_dict(part: Any) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny, reportAny]
    """Convert a Part-like object to a serializable dict."""
    if isinstance(part, str):
        return {"text": part}
    if isinstance(part, dict):
        return strip_none_values(part)  # pyright: ignore[reportUnknownArgumentType]
    if hasattr(part, "model_dump"):  # pyright: ignore[reportAny]
        return model_to_json_safe_dict(part, exclude_unset=True, exclude_none=True)  # pyright: ignore[reportAny]
    return strip_none_values(to_dict(part))  # pyright: ignore[reportAny]


def content_union_to_dict(
    content: types.ContentUnion | types.ContentUnionDict,  # pyright: ignore[reportUnknownMemberType, reportUnknownParameterType]
    default_role: str = "user",
) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
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
            result = dict(content)  # pyright: ignore[reportUnknownArgumentType]
            result["parts"] = [part_to_dict(p) for p in cast(list[Any], result["parts"])]  # pyright: ignore[reportExplicitAny, reportAny]
            if "role" not in result:
                result["role"] = default_role
            return result
        else:
            return {"role": default_role, "parts": [part_to_dict(content)]}
    elif isinstance(content, list):
        return {"role": default_role, "parts": [part_to_dict(p) for p in content]}  # pyright: ignore[reportUnknownVariableType]
    else:
        return {"role": default_role, "parts": [part_to_dict(content)]}

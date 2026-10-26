"""Span attribute helpers for the google-cloud-aiplatform (`vertexai`) SDK.

Messages are emitted in the native Gemini `{role, parts}` shape (same as the
`google_genai` instrumentation) so the frontend's Gemini renderer picks them up.
Everything is built from the SDK objects' own `to_dict()`; we never touch
`GenerationResponse.text`, which raises on function calls and multi-part
responses.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

import proto  # pyright: ignore[reportMissingTypeStubs]
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import Span

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.reasoning import (
    set_reasoning_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
    should_send_prompts,
    to_dict,
)
from lmnr.sdk.utils import json_dumps

_MODEL_PREFIX = "models/"
_LANGUAGE_MODEL_PARAMS = (
    "temperature",
    "max_output_tokens",
    "top_p",
    "top_k",
    "stop_sequences",
    "candidate_count",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


# proto-plus renamed `including_default_value_fields` (protobuf 5) and warns
# on the old name; older proto-plus only knows the old one.
_NO_DEFAULTS_KWARG = (
    "always_print_fields_with_no_presence"
    if "always_print_fields_with_no_presence" in inspect.signature(proto.Message.to_dict).parameters  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    else "including_default_value_fields"
)


def to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert a vertexai wrapper object, a raw proto message or a dict to a dict.

    vertexai's own `to_dict()` (and the proto conversion below, which is what it
    uses internally) yields snake_case keys and standard base64 for bytes.
    """
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)
    if isinstance(obj, proto.Message):
        return type(obj).to_dict(  # pyright: ignore[reportUnknownMemberType]
            obj,
            use_integers_for_enums=False,
            **{_NO_DEFAULTS_KWARG: False},
        )
    obj_to_dict = getattr(obj, "to_dict", None)
    if callable(obj_to_dict):
        return cast(dict[str, Any], obj_to_dict())
    return to_dict(obj)


def strip_model_prefix(model_name: str | None) -> str | None:
    """`publishers/google/models/gemini-2.0-flash` -> `gemini-2.0-flash`."""
    if not model_name:
        return None
    if _MODEL_PREFIX in model_name:
        return model_name.rsplit(_MODEL_PREFIX, 1)[-1]
    return model_name


def _is_image(obj: Any) -> bool:
    # `vertexai.generative_models.Image` has no `to_dict`; it exposes raw bytes
    return hasattr(obj, "_mime_type") and isinstance(getattr(obj, "data", None), bytes)


def _is_content(obj: Any) -> bool:
    if isinstance(obj, dict):
        return "parts" in obj
    # Content (wrapper or proto) has both `role` and `parts`; a Part has neither
    return hasattr(obj, "role") and hasattr(obj, "parts")


def part_to_dict(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"text": part}
    if _is_image(part):
        # bytes are base64-encoded (standard alphabet) by `json_dumps`
        return {"inline_data": {"mime_type": part._mime_type, "data": part.data}}
    return to_plain_dict(part)


def content_to_message(content: Any, default_role: str = "user") -> dict[str, Any]:
    """Convert a single Content-like value to a Gemini `{role, parts}` dict."""
    if isinstance(content, list):
        items = cast(list[Any], content)
        return {"role": default_role, "parts": [part_to_dict(item) for item in items]}
    if not _is_content(content):
        return {"role": default_role, "parts": [part_to_dict(content)]}
    message = dict(to_plain_dict(content))
    message["parts"] = [part_to_dict(part) for part in cast(list[Any], message.get("parts", []))]
    if not message.get("role"):
        message["role"] = default_role
    return message


def contents_to_messages(contents: Any) -> list[dict[str, Any]]:
    """Normalize vertexai's `ContentsType` into a list of Gemini messages.

    A list is a conversation if any item is a Content, and the parts of a
    single user message otherwise (that is how vertexai itself reads it).
    """
    if contents is None:
        return []
    if isinstance(contents, list):
        items = cast(list[Any], contents)
        if any(_is_content(item) for item in items):
            return [content_to_message(item) for item in items]
        return [content_to_message(items)]
    return [content_to_message(contents)]


def _system_message(system_instruction: Any) -> dict[str, Any] | None:
    if not system_instruction:
        return None
    message = content_to_message(system_instruction, default_role="system")
    message["role"] = "system"
    return message


def _get_arg(args: Sequence[Any], kwargs: dict[str, Any], name: str) -> Any:
    if args:
        return args[0]
    return kwargs.get(name)


def _set_messages(span: Span, messages: list[dict[str, Any]]):
    if messages:
        set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))


def _set_usage(
    span: Span,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
):
    set_span_attribute(span, gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    set_span_attribute(span, gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    set_span_attribute(span, "llm.usage.total_tokens", total_tokens)


# ============================================================================ #
# Generative models (Gemini): GenerativeModel.generate_content, ChatSession
# ============================================================================ #


def _generation_config_dict(generation_config: Any) -> dict[str, Any]:
    if not generation_config:
        return {}
    return to_plain_dict(generation_config)


def _function_declarations(tools: Any) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    for tool in cast(list[Any], tools or []):
        tool_dict = to_plain_dict(tool)
        declarations.extend(cast(list[dict[str, Any]], tool_dict.get("function_declarations", [])))
    return declarations


@dont_throw
def set_generative_request_attributes(
    span: Span,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    is_chat: bool,
    is_streaming: bool,
):
    # A ChatSession delegates to its GenerativeModel, where the per-model
    # defaults (config, tools, system instruction) live
    model = getattr(instance, "_model", None) if is_chat else instance

    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MODEL,
        strip_model_prefix(getattr(model, "_model_name", None)),
    )
    set_span_attribute(span, "llm.is_streaming", is_streaming)

    # vertexai replaces (not merges) the model's config with a per-call one
    config = _generation_config_dict(
        kwargs.get("generation_config") or getattr(model, "_generation_config", None)
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE, config.get("temperature")
    )
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TOP_P, config.get("top_p"))
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TOP_K, config.get("top_k"))
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS, config.get("max_output_tokens")
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY, config.get("presence_penalty")
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, config.get("frequency_penalty")
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES, config.get("stop_sequences")
    )
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_SEED, config.get("seed"))
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT, config.get("candidate_count")
    )
    set_reasoning_attributes(span, config)
    response_schema = config.get("response_schema")
    if response_schema:
        set_span_attribute(
            span, "gen_ai.request.structured_output_schema", json_dumps(response_schema)
        )

    declarations = _function_declarations(kwargs.get("tools") or getattr(model, "_tools", None))
    if declarations:
        set_span_attribute(span, "gen_ai.tool.definitions", json_dumps(declarations))

    if not should_send_prompts():
        return

    messages: list[dict[str, Any]] = []
    system_message = _system_message(getattr(model, "_system_instruction", None))
    if system_message:
        messages.append(system_message)
    if is_chat:
        messages.extend(content_to_message(content) for content in instance.history)
        messages.extend(contents_to_messages(_get_arg(args, kwargs, "content")))
    else:
        messages.extend(contents_to_messages(_get_arg(args, kwargs, "contents")))
    _set_messages(span, messages)


def _candidate_to_message(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(candidate)
    content = dict(cast(dict[str, Any], candidate.get("content") or {}))
    content.setdefault("parts", [])
    if not content.get("role"):
        content["role"] = "model"
    candidate["content"] = content
    return candidate


@dont_throw
def set_generative_response_attributes(span: Span, response: Any):
    set_generative_response_dict_attributes(span, to_plain_dict(response))


@dont_throw
def set_generative_response_dict_attributes(span: Span, response: dict[str, Any]):
    set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_ID, response.get("response_id"))
    set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_MODEL, response.get("model_version"))

    candidates = cast(list[dict[str, Any]], response.get("candidates") or [])
    finish_reasons = [c["finish_reason"] for c in candidates if c.get("finish_reason")]
    if finish_reasons:
        set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
    if should_send_prompts() and candidates:
        set_span_attribute(
            span,
            "gen_ai.output.messages",
            json_dumps([_candidate_to_message(c) for c in candidates]),
        )

    usage = cast(dict[str, Any], response.get("usage_metadata") or {})
    if not usage:
        return
    candidates_token_count = usage.get("candidates_token_count")
    # thinking tokens are not part of candidates_token_count; add them so
    # output tokens mean the same thing as in the other instrumentations
    thoughts_token_count = usage.get("thoughts_token_count")
    output_tokens = (
        (candidates_token_count or 0) + (thoughts_token_count or 0)
        if candidates_token_count is not None or thoughts_token_count is not None
        else None
    )
    _set_usage(
        span,
        usage.get("prompt_token_count"),
        output_tokens,
        usage.get("total_token_count"),
    )
    set_span_attribute(
        span, "gen_ai.usage.cache_read_input_tokens", usage.get("cached_content_token_count")
    )
    set_span_attribute(span, "gen_ai.usage.reasoning_tokens", thoughts_token_count)


def _merge_text_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate adjacent streamed text parts (keeping thoughts separate)."""
    merged: list[dict[str, Any]] = []
    for part in parts:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and set(part) <= {"text", "thought"}
            and set(previous) <= {"text", "thought"}
            and "text" in part
            and "text" in previous
            and bool(part.get("thought")) == bool(previous.get("thought"))
        ):
            merged[-1] = {**previous, "text": previous["text"] + part["text"]}
        else:
            merged.append(dict(part))
    return merged


class GenerativeStreamAccumulator:
    """Folds streamed `GenerationResponse` chunks into one response dict."""

    def __init__(self):
        self._parts: dict[int, list[dict[str, Any]]] = {}
        self._candidates: dict[int, dict[str, Any]] = {}
        self._roles: dict[int, str] = {}
        self._usage: dict[str, Any] = {}
        self._response_id: str | None = None
        self._model_version: str | None = None

    @dont_throw
    def add(self, chunk: Any):
        chunk_dict = to_plain_dict(chunk)
        self._response_id = chunk_dict.get("response_id") or self._response_id
        self._model_version = chunk_dict.get("model_version") or self._model_version
        # Vertex reports cumulative usage, so the latest value of each field wins
        for key, value in cast(dict[str, Any], chunk_dict.get("usage_metadata") or {}).items():
            if value is not None:
                self._usage[key] = value
        for position, candidate in enumerate(
            cast(list[dict[str, Any]], chunk_dict.get("candidates") or [])
        ):
            index = cast(int, candidate.get("index", position))
            content = cast(dict[str, Any], candidate.get("content") or {})
            self._parts.setdefault(index, []).extend(
                cast(list[dict[str, Any]], content.get("parts") or [])
            )
            if content.get("role"):
                self._roles[index] = content["role"]
            # keep the latest metadata (finish_reason, safety_ratings, ...)
            metadata = {k: v for k, v in candidate.items() if k != "content"}
            self._candidates[index] = {**self._candidates.get(index, {}), **metadata}

    def set_attributes(self, span: Span):
        set_generative_response_dict_attributes(span, self.response())

    def response(self) -> dict[str, Any]:
        return {
            "response_id": self._response_id,
            "model_version": self._model_version,
            "usage_metadata": self._usage,
            "candidates": [
                {
                    **self._candidates.get(index, {}),
                    "content": {
                        "role": self._roles.get(index, "model"),
                        "parts": _merge_text_parts(self._parts[index]),
                    },
                }
                for index in sorted(self._parts)
            ],
        }


# ============================================================================ #
# Legacy language models (PaLM): TextGenerationModel, language_models.ChatSession
# ============================================================================ #


@dont_throw
def set_language_model_request_attributes(
    span: Span,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    is_chat: bool,
    is_streaming: bool,
):
    model = getattr(instance, "_model", None) if is_chat else instance
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MODEL,
        strip_model_prefix(getattr(model, "_model_id", None)),
    )
    set_span_attribute(span, "llm.is_streaming", is_streaming)

    # chat sessions hold session-wide defaults as private attributes
    params = {
        name: kwargs[name] if kwargs.get(name) is not None else getattr(instance, f"_{name}", None)
        for name in _LANGUAGE_MODEL_PARAMS
    }
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE, params["temperature"])
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TOP_P, params["top_p"])
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TOP_K, params["top_k"])
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS, params["max_output_tokens"]
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY, params["presence_penalty"]
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, params["frequency_penalty"]
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES, params["stop_sequences"]
    )
    set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_SEED, params["seed"])
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT, params["candidate_count"]
    )

    if not should_send_prompts():
        return

    messages: list[dict[str, Any]] = []
    if is_chat:
        system_message = _system_message(getattr(instance, "_context", None))
        if system_message:
            messages.append(system_message)
        for chat_message in cast(list[Any], getattr(instance, "_message_history", None) or []):
            author = getattr(chat_message, "author", None)
            messages.append(
                {
                    "role": "user" if author == "user" else "model",
                    "parts": [{"text": getattr(chat_message, "content", "")}],
                }
            )
        prompt = _get_arg(args, kwargs, "message")
    else:
        prompt = _get_arg(args, kwargs, "prompt")
    if prompt is not None:
        messages.append({"role": "user", "parts": [{"text": str(prompt)}]})
    _set_messages(span, messages)


def _token_count(token_metadata: dict[str, Any], key: str) -> int | None:
    return cast(dict[str, Any], token_metadata.get(key) or {}).get("totalTokens")


@dont_throw
def set_language_model_response_attributes(span: Span, response: Any):
    candidates = cast(list[Any], getattr(response, "candidates", None) or [response])
    texts = [getattr(candidate, "text", None) or "" for candidate in candidates]
    set_language_model_output(span, texts)

    prediction_response = getattr(response, "raw_prediction_response", None)
    metadata = cast(dict[str, Any], getattr(prediction_response, "metadata", None) or {})
    token_metadata = cast(dict[str, Any], metadata.get("tokenMetadata") or {})
    if token_metadata:
        _set_usage(
            span,
            _token_count(token_metadata, "inputTokenCount"),
            _token_count(token_metadata, "outputTokenCount"),
            None,
        )


@dont_throw
def set_language_model_output(span: Span, texts: list[str]):
    if not should_send_prompts():
        return
    set_span_attribute(
        span,
        "gen_ai.output.messages",
        json_dumps([{"content": {"role": "model", "parts": [{"text": text}]}} for text in texts]),
    )


class LanguageStreamAccumulator:
    """Concatenates streamed `TextGenerationResponse` chunks."""

    def __init__(self):
        self._text: str = ""

    @dont_throw
    def add(self, chunk: Any):
        self._text += getattr(chunk, "text", None) or ""

    def set_attributes(self, span: Span):
        set_language_model_output(span, [self._text])

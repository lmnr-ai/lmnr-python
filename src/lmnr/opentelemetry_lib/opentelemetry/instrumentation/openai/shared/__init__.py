import json
import types
from typing import Any, cast

import pydantic
from httpx import URL
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_REQUEST_FREQUENCY_PENALTY,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_PRESENCE_PENALTY,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv._incubating.attributes.openai_attributes import (
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
)
from opentelemetry.trace import Span
from opentelemetry.trace.propagation import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import openai
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai.utils import (
    is_openai_v1,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    set_span_attribute,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import json_dumps

OPENAI_LLM_USAGE_TOKEN_TYPES = ["prompt_tokens", "completion_tokens"]
PROMPT_FILTER_KEY = "prompt_filter_results"
PROMPT_ERROR = "prompt_error"


logger = get_default_logger(__name__)


def set_client_attributes(
    span: Span,
    instance: Any,
):
    if not span.is_recording():
        return

    if not is_openai_v1():
        return

    client = instance._client  # pylint: disable=protected-access
    if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
        set_span_attribute(span, "gen_ai.request.base_url", str(client.base_url))
    if isinstance(client, (openai.AsyncAzureOpenAI, openai.AzureOpenAI)):
        set_span_attribute(span, "gen_ai.openai.api_version", client._api_version)


def _set_api_attributes(span: Span):
    if not span.is_recording():
        return

    if is_openai_v1():
        return

    base_url = openai.base_url if hasattr(openai, "base_url") else getattr(openai, "api_base", "https://api.openai.com")
    if isinstance(base_url, URL):
        base_url = str(base_url)

    set_span_attribute(span, "gen_ai.request.base_url", base_url)
    set_span_attribute(span, "gen_ai.request.api_type", openai.api_type)
    set_span_attribute(span, "gen_ai.request.api_version", openai.api_version)

    return


def set_tools_attributes(
    span: Span,
    tools: list[dict[str, Any]] | None,
):
    if not tools:
        return

    set_span_attribute(
        span,
        "gen_ai.tool.definitions",
        json_dumps(tools),
    )


def set_request_attributes(
    span: Span,
    kwargs: dict[str, Any],
    instance: Any | None = None,
):
    if not span.is_recording():
        return

    _set_api_attributes(span)

    base_url = _get_openai_base_url(instance) if instance else ""
    vendor = _get_vendor_from_url(base_url)
    set_span_attribute(span, GEN_AI_SYSTEM, vendor)

    model = kwargs.get("model")
    if vendor == "AWS" and model and "." in model:
        model = _cross_region_check(model)

    set_span_attribute(span, GEN_AI_REQUEST_MODEL, model)
    set_span_attribute(span, GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
    set_span_attribute(span, GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
    set_span_attribute(span, GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
    set_span_attribute(
        span, GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty")
    )
    set_span_attribute(
        span, GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty")
    )
    set_span_attribute(span, "llm.user", kwargs.get("user"))
    set_span_attribute(span, "llm.headers", str(kwargs.get("headers")))
    # The new OpenAI SDK removed the `headers` and create new field called `extra_headers`
    if kwargs.get("extra_headers") is not None:
        set_span_attribute(span, "llm.headers", str(kwargs.get("extra_headers")))
    set_span_attribute(span, "llm.is_streaming", kwargs.get("stream") or False)
    set_span_attribute(
        span,
        "gen_ai.request.reasoning_effort",
        kwargs.get("reasoning_effort"),
    )
    set_span_attribute(
        span,
        "openai.request.service_tier",
        kwargs.get("service_tier"),
    )
    if response_format := kwargs.get("response_format"):
        # backward-compatible check for
        # openai.types.shared_params.response_format_json_schema.ResponseFormatJSONSchema
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"  # pyright: ignore[reportUnknownMemberType]
            and response_format.get("json_schema")  # pyright: ignore[reportUnknownMemberType]
        ):
            schema = dict(cast(dict[str, Any], response_format.get("json_schema"))).get("schema")  # pyright: ignore[reportUnknownMemberType]
            if schema:
                set_span_attribute(
                    span,
                    "gen_ai.request.structured_output_schema",
                    json.dumps(schema),
                )
        else:
            try:
                from openai import Omit
                from openai.lib._parsing._completions import (
                    type_to_response_format_param,
                )
                from openai.types.chat.completion_create_params import ResponseFormat


                response_format_param = type_to_response_format_param(cast(ResponseFormat, response_format))
                if isinstance(response_format_param, Omit):
                    logger.debug("response_format is omitted")
                    return
                if response_format_param.get("type") == "json_schema":
                    schema = (response_format_param.get("json_schema") or {}).get("schema")
                    if schema:
                        set_span_attribute(
                            span,
                            "gen_ai.request.structured_output_schema",
                            json.dumps(schema),
                        )
            except (ImportError, TypeError, AttributeError):
                # if we fail to import from openai.lib._parsing._completions,
                # we fallback to the pydantic-based approach
                if isinstance(response_format, pydantic.BaseModel) or (
                    hasattr(response_format, "model_json_schema")  # pyright: ignore[reportUnknownArgumentType]
                    and callable(cast(pydantic.BaseModel, response_format).model_json_schema)
                ):
                    set_span_attribute(
                        span,
                        "gen_ai.request.structured_output_schema",
                        json.dumps(cast(pydantic.BaseModel, response_format).model_json_schema()),
                    )
                else:
                    schema = None
                    try:
                        schema = json.dumps(
                            pydantic.TypeAdapter(response_format).json_schema()
                        )
                    except Exception:
                        try:
                            schema = json.dumps(response_format)
                        except Exception:
                            logger.debug("Failed to stringify openai response format schema", exc_info=True)

                    if schema:
                        set_span_attribute(
                            span,
                            "gen_ai.request.structured_output_schema",
                            schema,
                        )


@dont_throw
def set_response_attributes(
    span: Span,
    response: dict[str, Any],
):
    if not span.is_recording():
        return

    if "error" in response:
        set_span_attribute(
            span,
            f"gen_ai.prompt.{PROMPT_ERROR}",
            json.dumps(response.get("error")),
        )
        return

    response_model = response.get("model")
    set_span_attribute(span, GEN_AI_RESPONSE_MODEL, response_model)
    set_span_attribute(span, GEN_AI_RESPONSE_ID, response.get("id"))

    set_span_attribute(
        span,
        OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
        response.get("system_fingerprint"),
    )
    _log_prompt_filter(span, response)
    usage = response.get("usage")
    set_span_attribute(
        span,
        "openai.response.service_tier",
        response.get("service_tier"),
    )
    if not usage:
        return

    if is_openai_v1() and not isinstance(usage, dict):
        usage = usage.__dict__

    usage = cast(dict[str, int | dict[str, int]], usage)

    set_span_attribute(span, "llm.usage.total_tokens", cast(int, usage.get("total_tokens") or 0))
    set_span_attribute(
        span,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        cast(int, usage.get("completion_tokens") or 0),
    )
    set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, cast(int, usage.get("prompt_tokens") or 0))
    prompt_tokens_details = dict(cast(dict[str, int], usage.get("prompt_tokens_details", {})))
    set_span_attribute(
        span,
        "gen_ai.usage.cache_read_input_tokens",
        prompt_tokens_details.get("cached_tokens", 0),
    )

    if completion_token_details := dict(cast(dict[str, int], usage.get("completion_tokens_details", {}))):
        reasoning_tokens = completion_token_details.get("reasoning_tokens")
        set_span_attribute(
            span,
            "gen_ai.usage.reasoning_tokens",
            reasoning_tokens or 0,
        )

    return


def _log_prompt_filter(span: Span, response_dict: dict[str, Any]):
    if response_dict.get("prompt_filter_results"):
        set_span_attribute(
            span,
            f"gen_ai.prompt.{PROMPT_FILTER_KEY}",
            json.dumps(response_dict.get("prompt_filter_results")),
        )


@dont_throw
def _set_span_stream_usage(span: Span, prompt_tokens: int | None, completion_tokens: int | None):
    if not span.is_recording():
        return

    if isinstance(completion_tokens, int) and completion_tokens >= 0:
        set_span_attribute(span, GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens)

    if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
        set_span_attribute(span, GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens)

    if (
        isinstance(prompt_tokens, int)
        and isinstance(completion_tokens, int)
        and completion_tokens + prompt_tokens >= 0
    ):
        set_span_attribute(
            span,
            "llm.usage.total_tokens",
            completion_tokens + prompt_tokens,
        )


def _get_openai_base_url(instance: Any):
    if hasattr(instance, "_client"):
        client = instance._client
        if isinstance(client, (openai.AsyncOpenAI, openai.OpenAI)):
            return str(client.base_url)

    return ""


def _get_vendor_from_url(base_url: str) -> str:
    if not base_url:
        return "openai"

    if "openai.azure.com" in base_url:
        return "Azure"
    elif "amazonaws.com" in base_url or "bedrock" in base_url:
        return "AWS"
    elif "googleapis.com" in base_url or "vertex" in base_url:
        return "Google"
    elif "openrouter.ai" in base_url:
        return "OpenRouter"

    return "openai"


def _cross_region_check(value: str) -> str:
    if not value or "." not in value:
        return value

    prefixes = ["us", "us-gov", "eu", "apac"]
    if any(value.startswith(prefix + ".") for prefix in prefixes):
        parts = value.split(".")
        if len(parts) > 2:
            return parts[2]
        else:
            return value
    else:
        _vendor, model = value.split(".", 1)
        return model


def is_streaming_response(response: Any) -> bool:
    if is_openai_v1():
        return isinstance(
            response,
            (
                openai.Stream,
                openai.AsyncStream,
                types.GeneratorType,
                types.AsyncGeneratorType,
            ),
        )

    return isinstance(response, (types.GeneratorType, types.AsyncGeneratorType))


def propagate_trace_context(
    span: Span,
    kwargs: dict[str, Any],
):
    if is_openai_v1():
        extra_headers = kwargs.get("extra_headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(extra_headers, context=ctx)
        kwargs["extra_headers"] = extra_headers
    else:
        headers = kwargs.get("headers", {})
        ctx = set_span_in_context(span)
        TraceContextTextMapPropagator().inject(headers, context=ctx)
        kwargs["headers"] = headers

"""OpenTelemetry Google Generative AI API instrumentation"""

import json
import logging
import os
from collections import defaultdict
from importlib.metadata import version
from typing import Any, AsyncGenerator, Callable, Collection, Generator, Sequence

from google.genai import types
from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, Status, StatusCode

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    safe_start_span,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    stamp_instrumentation_scope,
)
from lmnr.opentelemetry_lib.tracing.context import (
    get_event_attributes_from_context,
)
from lmnr.sdk.laminar import Laminar
from lmnr.sdk.utils import json_dumps

from .config import (
    Config,
)
from .schema_utils import SchemaJSONEncoder, process_schema
from .utils import (
    content_union_to_dict,
    dont_throw,
    merge_text_parts,
    model_to_json_safe_dict,
    process_stream_chunk,
    set_span_attribute,
    to_dict,
)

logger = logging.getLogger(__name__)

_instruments = ("google-genai >= 1.0.0",)

def should_send_prompts():
    return (
        os.getenv("LAMINAR_TRACE_CONTENT") or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


@dont_throw
def _set_request_attributes(span, args, kwargs):
    config_dict = to_dict(kwargs.get("config", {}))
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_MODEL, kwargs.get("model")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE,
        config_dict.get("temperature"),
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_TOP_P, config_dict.get("top_p")
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_TOP_K, config_dict.get("top_k")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_CHOICE_COUNT,
        config_dict.get("candidate_count"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS,
        config_dict.get("max_output_tokens"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_STOP_SEQUENCES,
        config_dict.get("stop_sequences"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
        config_dict.get("frequency_penalty"),
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
        config_dict.get("presence_penalty"),
    )
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_REQUEST_SEED, config_dict.get("seed")
    )

    if schema := config_dict.get("response_schema"):
        try:
            set_span_attribute(
                span,
                "gen_ai.request.structured_output_schema",
                json.dumps(process_schema(schema), cls=SchemaJSONEncoder),
            )
        except Exception:
            pass
    elif json_schema := config_dict.get("response_json_schema"):
        try:
            set_span_attribute(
                span,
                "gen_ai.request.structured_output_schema",
                json_dumps(json_schema),
            )
        except Exception:
            pass

    tools: list[types.FunctionDeclaration] = []
    arg_tools = config_dict.get("tools", kwargs.get("tools"))
    if arg_tools:
        for tool in arg_tools:
            if isinstance(tool, types.Tool):
                tools.extend(tool.function_declarations or [])
            elif isinstance(tool, dict) and isinstance(
                tool.get("function_declarations"), list
            ):
                tools.extend(tool.get("function_declarations", []))
            elif isinstance(tool, Callable):
                tools.append(types.FunctionDeclaration.from_callable(tool))

    if should_send_prompts():
        messages = []
        system_instruction = config_dict.get("system_instruction")
        if system_instruction:
            msg = content_union_to_dict(system_instruction, default_role="system")
            msg["role"] = "system"
            messages.append(msg)

        contents = kwargs.get("contents", [])
        if not isinstance(contents, list):
            contents = [contents]
        for content in contents:
            messages.append(content_union_to_dict(content))

        set_span_attribute(span, "gen_ai.input.messages", json_dumps(messages))
    if tools:
        span.set_attribute(
            "gen_ai.tool.definitions",
            json_dumps([to_dict(tool) for tool in tools]),
        )


@dont_throw
def _set_response_attributes(span, response: types.GenerateContentResponse):
    set_span_attribute(
        span, gen_ai_attributes.GEN_AI_RESPONSE_ID, to_dict(response).get("response_id")
    )
    set_span_attribute(
        span,
        gen_ai_attributes.GEN_AI_RESPONSE_MODEL,
        to_dict(response).get("model_version"),
    )

    if response.usage_metadata:
        usage_dict = to_dict(response.usage_metadata)
        candidates_token_count = usage_dict.get("candidates_token_count")
        # unlike OpenAI, and unlike input cached tokens, thinking tokens are
        # not counted as part of candidates token count, so we need to add them
        # separately for consistency with other instrumentations
        thoughts_token_count = usage_dict.get("thoughts_token_count")
        output_token_count = (
            (candidates_token_count or 0) + (thoughts_token_count or 0)
            if candidates_token_count is not None or thoughts_token_count is not None
            else None
        )
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS,
            usage_dict.get("prompt_token_count"),
        )
        set_span_attribute(
            span,
            gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS,
            output_token_count,
        )
        set_span_attribute(
            span,
            "llm.usage.total_tokens",
            usage_dict.get("total_token_count"),
        )
        set_span_attribute(
            span,
            "gen_ai.usage.cache_read_input_tokens",
            usage_dict.get("cached_content_token_count"),
        )
        set_span_attribute(
            span,
            "gen_ai.usage.reasoning_tokens",
            thoughts_token_count,
        )


@dont_throw
def _set_raw_response_attribute(
    span, response: types.GenerateContentResponse, record_raw_response: bool = False
):
    set_span_attribute(
        span,
        "gen_ai.output.messages",
        json_dumps(
            [
                model_to_json_safe_dict(
                    candidate, exclude_unset=True, exclude_none=True
                )
                for candidate in (response.candidates or [])
            ]
        ),
    )
    if record_raw_response:
        try:
            raw_response = json_dumps(model_to_json_safe_dict(response))
            # json_dumps swallows its own failures and returns "{}". Stamping
            # that would be worse than stamping nothing: the replay cache
            # PREFERS a non-empty raw response over reconstructing from
            # gen_ai.output.messages, so an empty object would shadow the
            # usable fallback and every cache hit would rebuild an empty
            # response.
            if raw_response and raw_response != "{}":
                set_span_attribute(span, "lmnr.sdk.raw.response", raw_response)
            else:
                logger.debug("Skipping empty lmnr.sdk.raw.response attribute")
        except Exception:
            logger.debug("Failed to set lmnr.sdk.raw.response attribute", exc_info=True)


@dont_throw
def _build_from_streaming_response(
    span: Span,
    response: Generator[types.GenerateContentResponse, None, None],
    record_raw_response: bool = False,
) -> Generator[types.GenerateContentResponse, None, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    for chunk in response:
        try:
            span.add_event("llm.content.completion.chunk")
        except Exception:
            pass
        # Important: do all processing in a separate sync function, that is
        # wrapped in @dont_throw. If we did it here, the @dont_throw on top of
        # this function would not be able to catch the errors, as they are
        # raised later, after the generator is returned, and when it is being
        # consumed.
        chunk_result = process_stream_chunk(
            chunk,
            role,
            model_version,
            aggregated_usage_metadata,
            final_parts,
        )
        # even though process_stream_chunk can't return None, the result can be
        # None, if the processing throws an error (see @dont_throw)
        if chunk_result:
            role = chunk_result["role"]
            model_version = chunk_result["model_version"]
        yield chunk

    try:
        compound_response = types.GenerateContentResponse(
            candidates=[
                {
                    "content": {
                        "parts": merge_text_parts(final_parts),
                        "role": role,
                    },
                }
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadataDict(
                **aggregated_usage_metadata
            ),
            model_version=model_version,
        )
        if span.is_recording():
            _set_raw_response_attribute(
                span, compound_response, record_raw_response=record_raw_response
            )
            _set_response_attributes(span, compound_response)
    finally:
        if span.is_recording():
            span.end()


@dont_throw
async def _abuild_from_streaming_response(
    span: Span,
    response: AsyncGenerator[types.GenerateContentResponse, None],
    record_raw_response: bool = False,
) -> AsyncGenerator[types.GenerateContentResponse, None]:
    final_parts = []
    role = "model"
    aggregated_usage_metadata = defaultdict(int)
    model_version = None
    async for chunk in response:
        try:
            span.add_event("llm.content.completion.chunk")
        except Exception:
            pass
        # Important: do all processing in a separate sync function, that is
        # wrapped in @dont_throw. If we did it here, the @dont_throw on top of
        # this function would not be able to catch the errors, as they are
        # raised later, after the generator is returned, and when it is being
        # consumed.
        chunk_result = process_stream_chunk(
            chunk,
            role,
            model_version,
            aggregated_usage_metadata,
            final_parts,
        )
        # even though process_stream_chunk can't return None, the result can be
        # None, if the processing throws an error (see @dont_throw)
        if chunk_result:
            role = chunk_result["role"]
            model_version = chunk_result["model_version"]
        yield chunk

    try:
        compound_response = types.GenerateContentResponse(
            candidates=[
                {
                    "content": {
                        "parts": merge_text_parts(final_parts),
                        "role": role,
                    },
                }
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadataDict(
                **aggregated_usage_metadata
            ),
            model_version=model_version,
        )
        if span.is_recording():
            _set_raw_response_attribute(
                span, compound_response, record_raw_response=record_raw_response
            )
            _set_response_attributes(span, compound_response)
    finally:
        if span.is_recording():
            span.end()


def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or "gemini.generate_content",
        attributes={"gen_ai.system": "gemini"},
        span_type="LLM",
    )
    if not span:
        logger.warning("Failed to start span for google genai")
        return wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _set_request_attributes(span, args, kwargs)

    try:
        # Check for debug replay mode and serve cached responses if available
        from lmnr.sdk.debug.replay import replay_enabled

        is_rollout = replay_enabled()
    except Exception:
        is_rollout = False

    try:
        if is_rollout:
            from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.rollout import (
                get_google_genai_rollout_wrapper,
            )

            rollout_wrapper = get_google_genai_rollout_wrapper()
            if rollout_wrapper:
                with Laminar.use_span(span):
                    response = rollout_wrapper.wrap_generate_content(
                        wrapped,
                        instance,
                        args,
                        kwargs,
                        is_streaming=bool(to_wrap.get("is_streaming")),
                        is_async=False,
                    )
            else:
                response = wrapped(*args, **kwargs)
        else:
            response = wrapped(*args, **kwargs)

        if to_wrap.get("is_streaming"):
            return _build_from_streaming_response(
                span, response, record_raw_response=is_rollout
            )
        if span.is_recording():
            _set_raw_response_attribute(span, response, record_raw_response=is_rollout)
            _set_response_attributes(span, response)
        span.end()
        return response
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


async def _awrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return await wrapped(*args, **kwargs)

    span = safe_start_span(
        name=to_wrap.get("span_name") or "gemini.generate_content",
        attributes={"gen_ai.system": "gemini"},
        span_type="LLM",
    )
    if not span:
        logger.warning("Failed to start span for async google genai")
        return await wrapped(*args, **kwargs)

    stamp_instrumentation_scope(span, to_wrap)
    _set_request_attributes(span, args, kwargs)

    try:
        # Check for debug replay mode and serve cached responses if available
        from lmnr.sdk.debug.replay import replay_enabled

        is_rollout = replay_enabled()
    except Exception:
        is_rollout = False

    try:
        if is_rollout:
            from lmnr.opentelemetry_lib.opentelemetry.instrumentation.google_genai.rollout import (
                get_google_genai_rollout_wrapper,
            )

            rollout_wrapper = get_google_genai_rollout_wrapper()
            if rollout_wrapper:
                # For async, we need to handle both sync and async wrapped functions
                # In rollout mode, wrapped might return cached (sync) or live (async)
                with Laminar.use_span(span):
                    result = rollout_wrapper.wrap_generate_content(
                        wrapped,
                        instance,
                        args,
                        kwargs,
                        is_streaming=bool(to_wrap.get("is_streaming")),
                        is_async=True,
                    )
                # If result is a coroutine or async generator, await/iterate it
                import inspect

                if inspect.iscoroutine(result):
                    response = await result
                elif inspect.isasyncgen(result):
                    # It's an async generator (cached streaming response)
                    response = result
                else:
                    # It's a sync response (cached non-streaming)
                    response = result
            else:
                response = await wrapped(*args, **kwargs)
        else:
            response = await wrapped(*args, **kwargs)

        if to_wrap.get("is_streaming"):
            return _abuild_from_streaming_response(
                span, response, record_raw_response=is_rollout
            )
        else:
            if span.is_recording():
                _set_raw_response_attribute(
                    span, response, record_raw_response=is_rollout
                )
                _set_response_attributes(span, response)

            span.end()
            return response
    except Exception as e:
        attributes = get_event_attributes_from_context()
        span.set_attribute(ERROR_TYPE, e.__class__.__name__)
        span.record_exception(e, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.end()
        raise


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="google.genai.models",
        object_name="Models",
        method_name="generate_content",
        span_name="gemini.generate_content",
        is_streaming=False,
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="google.genai.models",
        object_name="AsyncModels",
        method_name="generate_content",
        span_name="gemini.generate_content",
        is_streaming=False,
        is_async=True,
        wrapper_function=_awrap,
    ),
    WrappedFunctionSpec(
        package_name="google.genai.models",
        object_name="Models",
        method_name="generate_content_stream",
        span_name="gemini.generate_content_stream",
        is_streaming=True,
        is_async=False,
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="google.genai.models",
        object_name="AsyncModels",
        method_name="generate_content_stream",
        span_name="gemini.generate_content_stream",
        is_streaming=True,
        is_async=True,
        wrapper_function=_awrap,
    ),
]


class GoogleGenAiSdkInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for Google GenAI's client library."""

    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self, exception_logger=None):
        super().__init__()
        Config.exception_logger = exception_logger
        self.instrumentor_config = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                google_genai_version = version("google-genai")
            except Exception as e:
                logger.debug(f"Failed to get google-genai version {e}")
                google_genai_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="google-genai",
                version=google_genai_version,
            )
        return self._scope

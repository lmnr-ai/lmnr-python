"""LiteLLM callback logger for Laminar"""

import json
from datetime import datetime

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_PROMPT
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
from lmnr.opentelemetry_lib.decorators import json_dumps
from lmnr.opentelemetry_lib.litellm.utils import (
    get_tool_definition,
    is_validator_iterator,
    model_as_dict,
    set_span_attribute,
)
from lmnr.opentelemetry_lib.tracing import TracerWrapper

from lmnr.opentelemetry_lib.tracing.context import (
    get_current_context,
    get_event_attributes_from_context,
)
from lmnr.opentelemetry_lib.tracing.attributes import ASSOCIATION_PROPERTIES
from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

SUPPORTED_CALL_TYPES = ["completion", "acompletion", "responses", "aresponses"]

# Try to import the necessary LiteLLM components and gracefully handle ImportError
try:
    if not is_package_installed("litellm"):
        raise ImportError("LiteLLM is not installed")

    from litellm.integrations.custom_batch_logger import CustomBatchLogger

    class LaminarLiteLLMCallback(CustomBatchLogger):
        """Custom LiteLLM logger that sends logs to Laminar via OpenTelemetry spans

        Usage:
            import litellm
            from lmnr import Laminar, LaminarLiteLLMCallback

            # make sure this comes first
            Laminar.initialize()

            # Add the logger to LiteLLM callbacks
            litellm.callbacks = [LaminarLiteLLMCallback()]
        """

        logged_openai_responses: set[str]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not hasattr(TracerWrapper, "instance") or TracerWrapper.instance is None:
                raise ValueError("Laminar must be initialized before LiteLLM callback")

            self.logged_openai_responses = set()
            if is_package_installed("openai"):
                from lmnr.opentelemetry_lib.opentelemetry.instrumentation.openai import (
                    OpenAIInstrumentor,
                )

                openai_instrumentor = OpenAIInstrumentor()
                if (
                    openai_instrumentor
                    and openai_instrumentor.is_instrumented_by_opentelemetry
                ):
                    logger.info(
                        "Disabling OpenTelemetry instrumentation for OpenAI to avoid double-instrumentation of LiteLLM."
                    )
                    openai_instrumentor.uninstrument()

        def _get_tracer(self) -> Tracer:
            if not hasattr(TracerWrapper, "instance") or TracerWrapper.instance is None:
                raise ValueError("Laminar must be initialized before LiteLLM callback")
            return TracerWrapper().get_tracer()

        def log_success_event(
            self, kwargs, response_obj, start_time: datetime, end_time: datetime
        ):
            if kwargs.get("call_type") not in SUPPORTED_CALL_TYPES:
                return
            if kwargs.get("call_type") in ["responses", "aresponses"]:
                # responses API may be called multiple times with the same response_obj
                response_id = getattr(response_obj, "id", None)
                if response_id in self.logged_openai_responses:
                    return
                if response_id:
                    self.logged_openai_responses.add(response_id)
            self.logged_openai_responses.add(response_obj.id)
            try:
                self._create_span(
                    kwargs, response_obj, start_time, end_time, is_success=True
                )
            except Exception as e:
                logger.error(f"Error in log_success_event: {e}")

        def log_failure_event(
            self, kwargs, response_obj, start_time: datetime, end_time: datetime
        ):
            if kwargs.get("call_type") not in SUPPORTED_CALL_TYPES:
                return
            try:
                self._create_span(
                    kwargs, response_obj, start_time, end_time, is_success=False
                )
            except Exception as e:
                logger.error(f"Error in log_failure_event: {e}")

        async def async_log_success_event(
            self, kwargs, response_obj, start_time: datetime, end_time: datetime
        ):
            self.log_success_event(kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(
            self, kwargs, response_obj, start_time: datetime, end_time: datetime
        ):
            self.log_failure_event(kwargs, response_obj, start_time, end_time)

        def _create_span(
            self,
            kwargs,
            response_obj,
            start_time: datetime,
            end_time: datetime,
            is_success: bool,
        ):
            """Create an OpenTelemetry span for the LiteLLM call"""
            call_type = kwargs.get("call_type", "completion")
            if call_type == "aresponses":
                call_type = "responses"
            if call_type == "acompletion":
                call_type = "completion"
            span_name = f"litellm.{call_type}"
            try:
                tracer = self._get_tracer()
            except Exception as e:
                logger.error(f"Error getting tracer: {e}")
                return

            span = tracer.start_span(
                span_name,
                kind=SpanKind.CLIENT,
                start_time=int(start_time.timestamp() * 1e9),
                attributes={
                    "lmnr.internal.provider": "litellm",
                },
                context=get_current_context(),
            )
            try:
                model = kwargs.get("model", "unknown")
                if kwargs.get("custom_llm_provider"):
                    set_span_attribute(
                        span, "gen_ai.system", kwargs["custom_llm_provider"]
                    )

                messages = kwargs.get("messages", [])
                self._process_input_messages(span, messages)

                tools = kwargs.get("tools", [])
                self._process_request_tool_definitions(span, tools)

                set_span_attribute(span, "gen_ai.request.model", model)

                # Add more attributes from kwargs
                if "temperature" in kwargs:
                    set_span_attribute(
                        span, "gen_ai.request.temperature", kwargs["temperature"]
                    )
                if "max_tokens" in kwargs:
                    set_span_attribute(
                        span, "gen_ai.request.max_tokens", kwargs["max_tokens"]
                    )
                if "top_p" in kwargs:
                    set_span_attribute(span, "gen_ai.request.top_p", kwargs["top_p"])

                metadata = (
                    kwargs.get("litellm_params").get(
                        "metadata", kwargs.get("metadata", {})
                    )
                    or {}
                )
                tags = metadata.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        pass
                if (
                    tags
                    and isinstance(tags, (list, tuple, set))
                    and all(isinstance(tag, str) for tag in tags)
                ):
                    span.set_attribute(f"{ASSOCIATION_PROPERTIES}.tags", tags)

                user_id = metadata.get("user_id")
                if user_id:
                    span.set_attribute(f"{ASSOCIATION_PROPERTIES}.user_id", user_id)

                session_id = metadata.get("session_id")
                if session_id:
                    span.set_attribute(
                        f"{ASSOCIATION_PROPERTIES}.session_id", session_id
                    )

                optional_params = kwargs.get("optional_params") or {}
                if not optional_params:
                    hidden_params = metadata.get("hidden_params") or {}
                    optional_params = hidden_params.get("optional_params") or {}
                response_format = optional_params.get("response_format")
                if (
                    response_format
                    and isinstance(response_format, dict)
                    and response_format.get("type") == "json_schema"
                ):
                    schema = (response_format.get("json_schema") or {}).get("schema")
                    if schema:
                        span.set_attribute(
                            "gen_ai.request.structured_output_schema",
                            json_dumps(schema),
                        )

                if is_success:
                    span.set_status(Status(StatusCode.OK))
                    if kwargs.get("complete_streaming_response"):
                        self._process_success_response(
                            span,
                            kwargs.get("complete_streaming_response"),
                        )
                    else:
                        self._process_success_response(span, response_obj)
                else:
                    span.set_status(Status(StatusCode.ERROR))
                    if isinstance(response_obj, Exception):
                        attributes = get_event_attributes_from_context()
                        span.record_exception(response_obj, attributes=attributes)

            except Exception as e:
                attributes = get_event_attributes_from_context()
                span.record_exception(e, attributes=attributes)
                logger.error(f"Error in Laminar LiteLLM instrumentation: {e}")
            finally:
                span.end(int(end_time.timestamp() * 1e9))

        def _process_input_messages(self, span, messages):
            """Process and set message attributes on the span"""
            if not isinstance(messages, list):
                return

            prompt_index = 0
            for item in messages:
                block_dict = model_as_dict(item)
                if block_dict.get("type", "message") == "message":
                    tool_calls = block_dict.get("tool_calls", [])
                    self._process_tool_calls(
                        span, tool_calls, prompt_index, is_response=False
                    )
                    content = block_dict.get("content")
                    if is_validator_iterator(content):
                        # Have not been able to catch this in the wild, but keeping
                        # just in case, as raw OpenAI responses do that
                        content = [self._process_content_part(part) for part in content]
                    try:
                        stringified_content = (
                            content if isinstance(content, str) else json_dumps(content)
                        )
                    except Exception:
                        stringified_content = (
                            str(content) if content is not None else ""
                        )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.content",
                        stringified_content,
                    )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.role",
                        block_dict.get("role"),
                    )
                    prompt_index += 1

                elif block_dict.get("type") == "computer_call_output":
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.role",
                        "computer_call_output",
                    )
                    output_image_url = block_dict.get("output", {}).get("image_url")
                    if output_image_url:
                        set_span_attribute(
                            span,
                            f"{GEN_AI_PROMPT}.{prompt_index}.content",
                            json.dumps(
                                [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": output_image_url},
                                    }
                                ]
                            ),
                        )
                    prompt_index += 1
                elif block_dict.get("type") == "computer_call":
                    set_span_attribute(
                        span, f"{GEN_AI_PROMPT}.{prompt_index}.role", "assistant"
                    )
                    call_content = {}
                    if block_dict.get("id"):
                        call_content["id"] = block_dict.get("id")
                    if block_dict.get("action"):
                        call_content["action"] = block_dict.get("action")
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.tool_calls.0.arguments",
                        json.dumps(call_content),
                    )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.tool_calls.0.id",
                        block_dict.get("call_id"),
                    )
                    set_span_attribute(
                        span,
                        f"{GEN_AI_PROMPT}.{prompt_index}.tool_calls.0.name",
                        "computer_call",
                    )
                    prompt_index += 1
                elif block_dict.get("type") == "reasoning":
                    reasoning_summary = block_dict.get("summary")
                    if reasoning_summary and isinstance(reasoning_summary, list):
                        processed_chunks = [
                            {"type": "text", "text": chunk.get("text")}
                            for chunk in reasoning_summary
                            if isinstance(chunk, dict)
                            and chunk.get("type") == "summary_text"
                        ]
                        set_span_attribute(
                            span,
                            f"{GEN_AI_PROMPT}.{prompt_index}.reasoning",
                            json_dumps(processed_chunks),
                        )
                        set_span_attribute(
                            span,
                            f"{GEN_AI_PROMPT}.{prompt_index}.role",
                            "assistant",
                        )
                    # reasoning is followed by other content parts in the same messge,
                    # so we don't increment the prompt index
                # TODO: handle other block types

        def _process_request_tool_definitions(self, span, tools):
            """Process and set tool definitions attributes on the span"""
            if not isinstance(tools, list):
                return

            for i, tool in enumerate(tools):
                tool_dict = model_as_dict(tool)
                tool_definition = get_tool_definition(tool_dict)
                function_name = tool_definition.get("name")
                function_description = tool_definition.get("description")
                function_parameters = tool_definition.get("parameters")
                set_span_attribute(
                    span,
                    f"llm.request.functions.{i}.name",
                    function_name,
                )
                set_span_attribute(
                    span,
                    f"llm.request.functions.{i}.description",
                    function_description,
                )
                set_span_attribute(
                    span,
                    f"llm.request.functions.{i}.parameters",
                    json.dumps(function_parameters),
                )

        def _process_response_usage(self, span, usage):
            """Process and set usage attributes on the span"""
            usage_dict = model_as_dict(usage)
            if (
                not usage_dict.get("prompt_tokens")
                and not usage_dict.get("completion_tokens")
                and not usage_dict.get("total_tokens")
            ):
                return

            set_span_attribute(
                span, "gen_ai.usage.input_tokens", usage_dict.get("prompt_tokens")
            )
            set_span_attribute(
                span, "gen_ai.usage.output_tokens", usage_dict.get("completion_tokens")
            )
            set_span_attribute(
                span, "llm.usage.total_tokens", usage_dict.get("total_tokens")
            )

            if usage_dict.get("prompt_tokens_details"):
                details = usage_dict.get("prompt_tokens_details", {})
                details = model_as_dict(details)
                if details.get("cached_tokens"):
                    set_span_attribute(
                        span,
                        "gen_ai.usage.cache_read_input_tokens",
                        details.get("cached_tokens"),
                    )
                # TODO: add audio/image/text token details
            if usage_dict.get("completion_tokens_details"):
                details = usage_dict.get("completion_tokens_details", {})
                details = model_as_dict(details)
                if details.get("reasoning_tokens"):
                    set_span_attribute(
                        span,
                        "gen_ai.usage.reasoning_tokens",
                        details.get("reasoning_tokens"),
                    )

        def _process_tool_calls(self, span, tool_calls, choice_index, is_response=True):
            """Process and set tool call attributes on the span"""
            attr_prefix = "completion" if is_response else "prompt"
            if not isinstance(tool_calls, list):
                return

            for j, tool_call in enumerate(tool_calls):
                tool_call_dict = model_as_dict(tool_call)

                tool_name = tool_call_dict.get(
                    "name", tool_call_dict.get("function", {}).get("name", "")
                )
                set_span_attribute(
                    span,
                    f"gen_ai.{attr_prefix}.{choice_index}.tool_calls.{j}.name",
                    tool_name,
                )

                call_id = tool_call_dict.get("id", "")
                set_span_attribute(
                    span,
                    f"gen_ai.{attr_prefix}.{choice_index}.tool_calls.{j}.id",
                    call_id,
                )

                tool_arguments = tool_call_dict.get(
                    "arguments", tool_call_dict.get("function", {}).get("arguments", "")
                )
                if isinstance(tool_arguments, str):
                    set_span_attribute(
                        span,
                        f"gen_ai.{attr_prefix}.{choice_index}.tool_calls.{j}.arguments",
                        tool_arguments,
                    )
                else:
                    set_span_attribute(
                        span,
                        f"gen_ai.{attr_prefix}.{choice_index}.tool_calls.{j}.arguments",
                        json.dumps(model_as_dict(tool_arguments)),
                    )

        def _process_response_choices(self, span, choices):
            """Process and set choice attributes on the span"""
            if not isinstance(choices, list):
                return

            for i, choice in enumerate(choices):
                choice_dict = model_as_dict(choice)
                message = choice_dict.get("message", choice_dict)

                role = message.get("role", "unknown")
                set_span_attribute(span, f"gen_ai.completion.{i}.role", role)

                tool_calls = message.get("tool_calls", [])
                self._process_tool_calls(span, tool_calls, i, is_response=True)

                content = message.get("content", "")
                if content is None:
                    continue
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    if isinstance(reasoning_content, str):
                        reasoning_content = [
                            {
                                "type": "text",
                                "text": reasoning_content,
                            }
                        ]
                    elif not isinstance(reasoning_content, list):
                        reasoning_content = [
                            {
                                "type": "text",
                                "text": str(reasoning_content),
                            }
                        ]
                else:
                    reasoning_content = []
                if isinstance(content, str):
                    if reasoning_content:
                        set_span_attribute(
                            span,
                            f"gen_ai.completion.{i}.content",
                            json.dumps(
                                reasoning_content
                                + [
                                    {
                                        "type": "text",
                                        "text": content,
                                    }
                                ]
                            ),
                        )
                    else:
                        set_span_attribute(
                            span,
                            f"gen_ai.completion.{i}.content",
                            content,
                        )
                elif isinstance(content, list):
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.{i}.content",
                        json.dumps(reasoning_content + content),
                    )
                else:
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.{i}.content",
                        json.dumps(reasoning_content + [model_as_dict(content)]),
                    )

        def _process_content_part(self, content_part: dict) -> dict:
            content_part_dict = model_as_dict(content_part)
            if content_part_dict.get("type") == "output_text":
                return {"type": "text", "text": content_part_dict.get("text")}
            return content_part_dict

        def _process_response_output(self, span, output):
            """Response of OpenAI Responses API"""
            if not isinstance(output, list):
                return
            set_span_attribute(span, "gen_ai.completion.0.role", "assistant")
            tool_call_index = 0
            for block in output:
                block_dict = model_as_dict(block)
                if block_dict.get("type") == "message":
                    content = block_dict.get("content")
                    if content is None:
                        continue
                    if isinstance(content, str):
                        set_span_attribute(span, "gen_ai.completion.0.content", content)
                    elif isinstance(content, list):
                        set_span_attribute(
                            span,
                            "gen_ai.completion.0.content",
                            json_dumps(
                                [self._process_content_part(part) for part in content]
                            ),
                        )
                if block_dict.get("type") == "function_call":
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                        block_dict.get("id"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                        block_dict.get("name"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.arguments",
                        block_dict.get("arguments"),
                    )
                    tool_call_index += 1
                elif block_dict.get("type") == "file_search_call":
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                        block_dict.get("id"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                        "file_search_call",
                    )
                    tool_call_index += 1
                elif block_dict.get("type") == "web_search_call":
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                        block_dict.get("id"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                        "web_search_call",
                    )
                    tool_call_index += 1
                elif block_dict.get("type") == "computer_call":
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.id",
                        block_dict.get("call_id"),
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.name",
                        "computer_call",
                    )
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.0.tool_calls.{tool_call_index}.arguments",
                        json_dumps(block_dict.get("action")),
                    )
                    tool_call_index += 1
                elif block_dict.get("type") == "reasoning":
                    reasoning_summary = block_dict.get("summary")
                    if reasoning_summary and isinstance(reasoning_summary, list):
                        processed_chunks = [
                            {"type": "text", "text": chunk.get("text")}
                            for chunk in reasoning_summary
                            if isinstance(chunk, dict)
                            and chunk.get("type") == "summary_text"
                        ]
                        set_span_attribute(
                            span,
                            "gen_ai.completion.0.reasoning",
                            json_dumps(processed_chunks),
                        )
                # TODO: handle other block types, in particular other calls

        def _process_success_response(self, span, response_obj):
            """Process successful response attributes"""
            response_dict = model_as_dict(response_obj)
            set_span_attribute(span, "gen_ai.response.id", response_dict.get("id"))
            set_span_attribute(
                span, "gen_ai.response.model", response_dict.get("model")
            )

            if getattr(response_obj, "usage", None):
                self._process_response_usage(span, getattr(response_obj, "usage", None))
            elif response_dict.get("usage"):
                self._process_response_usage(span, response_dict.get("usage"))

            if response_dict.get("cache_creation_input_tokens"):
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_creation_input_tokens",
                    response_dict.get("cache_creation_input_tokens"),
                )
            if response_dict.get("cache_read_input_tokens"):
                set_span_attribute(
                    span,
                    "gen_ai.usage.cache_read_input_tokens",
                    response_dict.get("cache_read_input_tokens"),
                )

            if response_dict.get("choices"):
                self._process_response_choices(span, response_dict.get("choices"))
            elif response_dict.get("output"):
                self._process_response_output(span, response_dict.get("output"))

except ImportError as e:
    logger.debug(f"LiteLLM callback unavailable: {e}")

    # Create a no-op logger when LiteLLM is not available
    class LaminarLiteLLMCallback:
        """No-op logger when LiteLLM is not available"""

        def __init__(self, **kwargs):
            logger.warning(
                "LiteLLM is not installed. Install with: pip install litellm"
            )

        def log_success_event(self, *args, **kwargs):
            pass

        def log_failure_event(self, *args, **kwargs):
            pass

        async def async_log_success_event(self, *args, **kwargs):
            pass

        async def async_log_failure_event(self, *args, **kwargs):
            pass

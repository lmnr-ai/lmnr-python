"""LiteLLM callback logger for Laminar"""

import json
from datetime import datetime

from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
from lmnr.opentelemetry_lib.litellm.utils import model_as_dict, set_span_attribute
from lmnr.opentelemetry_lib.tracing import TracerWrapper

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

SUPPORTED_CALL_TYPES = ["completion", "acompletion"]

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

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if not hasattr(TracerWrapper, "instance") or TracerWrapper.instance is None:
                raise ValueError("Laminar must be initialized before LiteLLM callback")

        def _get_tracer(self) -> Tracer:
            if not hasattr(TracerWrapper, "instance") or TracerWrapper.instance is None:
                raise ValueError("Laminar must be initialized before LiteLLM callback")
            return TracerWrapper().get_tracer()

        def log_success_event(
            self, kwargs, response_obj, start_time: datetime, end_time: datetime
        ):
            if kwargs.get("call_type") not in SUPPORTED_CALL_TYPES:
                return
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
            span_name = "litellm.completion"
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
                        span.record_exception(response_obj)

            except Exception as e:
                span.record_exception(e)
                logger.error(f"Error in Laminar LiteLLM instrumentation: {e}")
            finally:
                span.end(int(end_time.timestamp() * 1e9))

        def _process_input_messages(self, span, messages):
            """Process and set message attributes on the span"""
            if not isinstance(messages, list):
                return

            for i, message in enumerate(messages):
                message_dict = model_as_dict(message)
                role = message_dict.get("role", "unknown")
                set_span_attribute(span, f"gen_ai.prompt.{i}.role", role)

                tool_calls = message_dict.get("tool_calls", [])
                self._process_tool_calls(span, tool_calls, i, is_response=False)

                content = message_dict.get("content", "")
                if content is None:
                    continue
                if isinstance(content, str):
                    set_span_attribute(span, f"gen_ai.prompt.{i}.content", content)
                elif isinstance(content, list):
                    set_span_attribute(
                        span, f"gen_ai.prompt.{i}.content", json.dumps(content)
                    )
                else:
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{i}.content",
                        json.dumps(model_as_dict(content)),
                    )
                if role == "tool":
                    set_span_attribute(
                        span,
                        f"gen_ai.prompt.{i}.tool_call_id",
                        message_dict.get("tool_call_id"),
                    )

        def _process_request_tool_definitions(self, span, tools):
            """Process and set tool definitions attributes on the span"""
            if not isinstance(tools, list):
                return

            for i, tool in enumerate(tools):
                tool_dict = model_as_dict(tool)
                if tool_dict.get("type") != "function":
                    # TODO: parse other tool types
                    continue

                function_dict = tool_dict.get("function", {})
                function_name = function_dict.get("name", "")
                function_description = function_dict.get("description", "")
                function_parameters = function_dict.get("parameters", {})
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
            # TODO: add completion tokens details (reasoning tokens)

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
                if isinstance(content, str):
                    set_span_attribute(span, f"gen_ai.completion.{i}.content", content)
                elif isinstance(content, list):
                    set_span_attribute(
                        span, f"gen_ai.completion.{i}.content", json.dumps(content)
                    )
                else:
                    set_span_attribute(
                        span,
                        f"gen_ai.completion.{i}.content",
                        json.dumps(model_as_dict(content)),
                    )

        def _process_success_response(self, span, response_obj):
            """Process successful response attributes"""
            response_dict = model_as_dict(response_obj)
            set_span_attribute(span, "gen_ai.response.id", response_dict.get("id"))
            set_span_attribute(
                span, "gen_ai.response.model", response_dict.get("model")
            )

            if response_dict.get("usage"):
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

from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from lmnr.sdk.utils import json_dumps
from ..shared import _set_span_attribute
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.trace import Status, StatusCode
from typing_extensions import override

from openai import AssistantEventHandler


class EventHandlerWrapper(AssistantEventHandler):
    _prompt_tokens = 0
    _completion_tokens = 0

    def __init__(self, original_handler, span):
        super().__init__()
        self._original_handler = original_handler
        self._span = span
        self._output_messages = []

    @override
    def on_end(self):
        _set_span_attribute(
            self._span,
            SpanAttributes.LLM_USAGE_PROMPT_TOKENS,
            self._prompt_tokens,
        )
        _set_span_attribute(
            self._span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            self._completion_tokens,
        )
        _set_span_attribute(
            self._span,
            "gen_ai.output.messages",
            json_dumps(self._output_messages),
        )
        self._original_handler.on_end()
        self._span.end()

    @override
    def on_event(self, event):
        self._original_handler.on_event(event)

    @override
    def on_run_step_created(self, run_step):
        self._original_handler.on_run_step_created(run_step)

    @override
    def on_run_step_delta(self, delta, snapshot):
        self._original_handler.on_run_step_delta(delta, snapshot)

    @override
    def on_run_step_done(self, run_step):
        if run_step.usage:
            self._prompt_tokens += run_step.usage.prompt_tokens
            self._completion_tokens += run_step.usage.completion_tokens
        self._original_handler.on_run_step_done(run_step)

    @override
    def on_tool_call_created(self, tool_call):
        self._original_handler.on_tool_call_created(tool_call)

    @override
    def on_tool_call_delta(self, delta, snapshot):
        self._original_handler.on_tool_call_delta(delta, snapshot)

    @override
    def on_tool_call_done(self, tool_call):
        self._original_handler.on_tool_call_done(tool_call)

    @override
    def on_exception(self, exception: Exception):
        self._span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
        self._span.record_exception(
            exception, attributes=get_event_attributes_from_context()
        )
        self._span.set_status(Status(StatusCode.ERROR, str(exception)))
        self._original_handler.on_exception(exception)

    @override
    def on_timeout(self):
        self._original_handler.on_timeout()

    @override
    def on_message_created(self, message):
        self._original_handler.on_message_created(message)

    @override
    def on_message_delta(self, delta, snapshot):
        self._original_handler.on_message_delta(delta, snapshot)

    @override
    def on_message_done(self, message):
        if self._output_messages:
            self._output_messages[-1]["id"] = message.id
        self._original_handler.on_message_done(message)

    @override
    def on_text_created(self, text):
        self._original_handler.on_text_created(text)

    @override
    def on_text_delta(self, delta, snapshot):
        self._original_handler.on_text_delta(delta, snapshot)

    @override
    def on_text_done(self, text):
        self._original_handler.on_text_done(text)
        self._output_messages.append({"role": "assistant", "content": text.value})

    @override
    def on_image_file_done(self, image_file):
        self._original_handler.on_image_file_done(image_file)

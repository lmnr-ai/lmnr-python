from __future__ import annotations

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, Status, StatusCode
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    set_span_attribute,
)
from lmnr.opentelemetry_lib.tracing.context import get_event_attributes_from_context
from openai import AssistantEventHandler
from openai.types.beta import AssistantStreamEvent
from openai.types.beta.threads import ImageFile, Message, MessageDelta, Text
from openai.types.beta.threads.runs import (
    RunStep,
    RunStepDelta,
    ToolCall,
    ToolCallDelta,
)
from openai.types.beta.threads.text_delta import TextDelta


class EventHandlerWrapper(AssistantEventHandler):
    _current_text_index: int = 0
    _prompt_tokens: int = 0
    _completion_tokens: int = 0

    def __init__(
        self,
        original_handler: AssistantEventHandler,
        span: Span
    ):
        super().__init__()
        self._original_handler: AssistantEventHandler = original_handler
        self._span: Span = span

    @override
    def on_end(self):
        set_span_attribute(
            self._span,
            GEN_AI_USAGE_INPUT_TOKENS,
            self._prompt_tokens,
        )
        set_span_attribute(
            self._span,
            GEN_AI_USAGE_OUTPUT_TOKENS,
            self._completion_tokens,
        )
        self._original_handler.on_end()
        self._span.end()

    @override
    def on_event(self, event: AssistantStreamEvent):
        self._original_handler.on_event(event)

    @override
    def on_run_step_created(self, run_step: RunStep):
        self._original_handler.on_run_step_created(run_step)

    @override
    def on_run_step_delta(self, delta: RunStepDelta, snapshot: RunStep):
        self._original_handler.on_run_step_delta(delta, snapshot)

    @override
    def on_run_step_done(self, run_step: RunStep):
        if run_step.usage:
            self._prompt_tokens += run_step.usage.prompt_tokens
            self._completion_tokens += run_step.usage.completion_tokens
        self._original_handler.on_run_step_done(run_step)

    @override
    def on_tool_call_created(self, tool_call: ToolCall):
        self._original_handler.on_tool_call_created(tool_call)

    @override
    def on_tool_call_delta(self, delta: ToolCallDelta, snapshot: ToolCall):
        self._original_handler.on_tool_call_delta(delta, snapshot)

    @override
    def on_tool_call_done(self, tool_call: ToolCall):
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
    def on_message_created(self, message: Message):
        self._original_handler.on_message_created(message)

    @override
    def on_message_delta(self, delta: MessageDelta, snapshot: Message):
        self._original_handler.on_message_delta(delta, snapshot)

    @override
    def on_message_done(self, message: Message):
        set_span_attribute(
            self._span,
            f"gen_ai.response.{self._current_text_index}.id",
            message.id,
        )
        self._original_handler.on_message_done(message)
        self._current_text_index += 1

    @override
    def on_text_created(self, text: Text):
        self._original_handler.on_text_created(text)

    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text):
        self._original_handler.on_text_delta(delta, snapshot)

    @override
    def on_text_done(self, text: Text):
        self._original_handler.on_text_done(text)
        set_span_attribute(
            self._span,
            f"gen_ai.completion.{self._current_text_index}.role",
            "assistant",
        )
        set_span_attribute(
            self._span,
            f"gen_ai.completion.{self._current_text_index}.content",
            text.value,
        )

    @override
    def on_image_file_done(self, image_file: ImageFile):
        self._original_handler.on_image_file_done(image_file)

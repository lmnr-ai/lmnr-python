"""LangChain callback handler that records chain, LLM, tool and retriever runs as spans."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping, Sequence
from contextvars import Token
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry import context as context_api
from opentelemetry.context import _RUNTIME_CONTEXT, _SUPPRESS_INSTRUMENTATION_KEY, Context
from opentelemetry.trace import Span, Status, StatusCode, set_span_in_context
from opentelemetry.util.types import AttributeValue

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langchain.span_utils import (
    retriever_output,
    sanitize_metadata,
    set_chat_request,
    set_completion_request,
    set_request_attributes,
    set_response_attributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langchain.vendor_detection import (
    detect_vendor_from_class,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.context_keys import (
    LANGGRAPH_TOPOLOGY,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.utils import (
    dont_throw,
    safe_start_span,
    set_span_attribute,
    should_send_prompts,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    set_instrumentation_scope_attributes,
)
from lmnr.opentelemetry_lib.tracing.attributes import SPAN_INPUT, SPAN_OUTPUT
from lmnr.opentelemetry_lib.tracing.context import (
    attach_context,
    detach_context,
    get_current_context,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.types import LaminarSpanType
from lmnr.sdk.utils import json_dumps

logger = get_default_logger(__name__)

_INSTRUMENTATION_PACKAGE = "lmnr.opentelemetry_lib.opentelemetry.instrumentation"
_OPENAI_LLM_TYPES = frozenset({"openai-chat", "azure-openai-chat", "openai", "azure"})
# LangChain `_llm_type` (the `_type` invocation param) → (module, class) of the
# Laminar instrumentor that records the provider SDK call the model makes.
# Subclasses that override `_llm_type` (ChatDeepSeek, ChatXAI, ...) are not
# listed and keep the LangChain span.
_PROVIDER_INSTRUMENTORS: dict[str, tuple[str, str]] = {
    **{llm_type: ("openai", "OpenAIInstrumentor") for llm_type in _OPENAI_LLM_TYPES},
    "anthropic-chat": ("anthropic", "AnthropicInstrumentor"),
    "chat-google-generative-ai": ("google_genai", "GoogleGenAiSdkInstrumentor"),
    "groq-chat": ("groq", "GroqInstrumentor"),
}


@cache
def _google_genai_uses_sdk() -> bool:
    """langchain-google-genai >= 4 calls `google.genai`; earlier versions use a
    different client that the google_genai instrumentor does not wrap."""
    try:
        return int(version("langchain-google-genai").split(".")[0]) >= 4
    except (PackageNotFoundError, ValueError):
        return False


def _provider_records_call(kwargs: Mapping[str, Any]) -> bool:
    """True if a Laminar provider instrumentor will record this LLM run's SDK call."""
    params = kwargs.get("invocation_params")
    if not isinstance(params, Mapping):
        return False
    params = cast(Mapping[str, Any], params)
    llm_type = params.get("_type")
    target = _PROVIDER_INSTRUMENTORS.get(llm_type) if isinstance(llm_type, str) else None
    if target is None:
        return False
    # Structured output on the Responses API goes through `responses.parse`,
    # which the openai instrumentor does not wrap, and whether that API is
    # used is decided later, inside the model
    if llm_type in _OPENAI_LLM_TYPES and params.get("response_format") is not None:
        return False
    if llm_type == "chat-google-generative-ai" and not _google_genai_uses_sdk():
        return False
    # An instrumentor's module is always imported once it is instrumented;
    # reading it from sys.modules avoids importing provider SDKs here
    module = sys.modules.get(f"{_INSTRUMENTATION_PACKAGE}.{target[0]}")
    instrumentor = getattr(getattr(module, target[1], None), "_instance", None)
    return bool(instrumentor is not None and instrumentor.is_instrumented_by_opentelemetry)


class _SuppressWhileRunning:
    """`_SUPPRESS_INSTRUMENTATION_KEY` value that is truthy only while its run is open.

    Async LangChain fires `on_llm_end` from a different task than
    `on_chat_model_start` (`asyncio.gather` in `agenerate`, the producer task
    of `astream_v2`), where the start's context token cannot be reset. A plain
    `True` would then stay attached in the caller's context and silently
    suppress every later provider call; this flag stops suppressing once the
    run ends, wherever the token ended up.

    It also carries the LLM run's id, for code under the run that is not handed
    a `run_manager` (`BaseChatModel.stream` calls `_stream` without one).
    """

    __slots__: tuple[str, ...] = ("active", "run_id")

    def __init__(self, run_id: UUID) -> None:
        self.active: bool = True
        self.run_id: UUID = run_id

    def __bool__(self) -> bool:
        return self.active


@dataclass
class _Run:
    span: Span
    # Isolated context with this run's span current; child runs parent from it
    context: Context
    children: list[UUID] = field(default_factory=list)
    isolated_token: Token[Context] | None = None
    global_tokens: list[object] = field(default_factory=list)
    suppress_flag: _SuppressWhileRunning | None = None
    request_model: str | None = None


def _detach_global(token: object) -> None:
    # Callbacks of one run may fire in different asyncio tasks or threads, in
    # which case the token cannot be reset there. That is harmless — the span
    # itself is ended correctly — so detach through the runtime context
    # directly to skip the error log `context_api.detach` would emit.
    with contextlib.suppress(Exception):
        _RUNTIME_CONTEXT.detach(token)  # pyright: ignore[reportArgumentType]


def _run_name(serialized: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> str:
    serialized = serialized or {}
    serialized_kwargs = serialized.get("kwargs")
    if isinstance(serialized_kwargs, Mapping):
        name = cast(Mapping[str, Any], serialized_kwargs).get("name")
        if isinstance(name, str) and name:
            return name
    for name in (kwargs.get("name"), serialized.get("name")):
        if isinstance(name, str) and name:
            return name
    return _class_name(serialized) or "unknown"


def _class_name(serialized: Mapping[str, Any] | None) -> str:
    """Last segment of the serialized class path, e.g. `ChatOpenAI`."""
    class_id = (serialized or {}).get("id")
    if isinstance(class_id, list) and class_id:
        return str(cast(list[Any], class_id)[-1])
    return class_id if isinstance(class_id, str) else ""


def _thread_id(metadata: Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> str | None:
    """LangGraph's `config.configurable.thread_id`, the conversation id."""
    if metadata and (thread_id := metadata.get("thread_id")) is not None:
        return str(thread_id)
    config = kwargs.get("config")
    if not isinstance(config, Mapping):
        return None
    configurable = cast(Mapping[str, Any], config).get("configurable")
    if not isinstance(configurable, Mapping):
        return None
    thread_id = cast(Mapping[str, Any], configurable).get("thread_id")
    return str(thread_id) if thread_id is not None else None


class LaminarCallbackHandler(BaseCallbackHandler):
    """Records LangChain runs as Laminar spans.

    One handler instance is shared by every callback manager (it is injected in
    `BaseCallbackManager.__init__`), so all state is keyed by LangChain's
    `run_id`. Each run's span is attached to both the isolated Laminar context,
    so `observe` / `Laminar.start_span` inside a tool nest under it, and the
    global OTel context, so third-party OTel instrumentations do too.
    """

    def __init__(self, scope: LaminarInstrumentationScopeAttributes) -> None:
        super().__init__()
        self._scope: LaminarInstrumentationScopeAttributes = scope
        self._runs: dict[UUID, _Run] = {}
        # LLM runs left to the provider SDK instrumentation (no LangChain
        # span), mapped to the context their children parent from
        self._handed_off: dict[UUID, Context] = {}
        # Callbacks must run in the caller's task / thread, not an executor,
        # so the contexts they attach are visible to the code they wrap.
        self.run_inline: bool = True

    def get_span(self, run_id: UUID) -> Span | None:
        run = self._runs.get(run_id)
        return run.span if run is not None else None

    @staticmethod
    def current_llm_run_id() -> UUID | None:
        """Id of the LLM run open in the current context, if any."""
        flag = context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        if isinstance(flag, _SuppressWhileRunning) and flag.active:
            return flag.run_id
        return None

    def _start_run(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        span_type: LaminarSpanType = "DEFAULT",
        attributes: dict[str, AttributeValue] | None = None,
        tags: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        suppress_downstream: bool = False,
    ) -> Span | None:
        suppress = context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)
        # An open LLM run's own flag is aimed at provider SDK instrumentations;
        # LangChain runs started under it (a tool the model invokes, code
        # iterating the model's stream) are still recorded
        if suppress and not isinstance(suppress, _SuppressWhileRunning):
            return None

        parent = self._runs.get(parent_run_id) if parent_run_id is not None else None
        if parent is not None:
            parent_context = parent.context
        elif parent_run_id is not None and parent_run_id in self._handed_off:
            parent_context = self._handed_off[parent_run_id]
        else:
            parent_context = get_current_context()
        span = safe_start_span(
            name, context=parent_context, attributes=attributes, span_type=span_type
        )
        if span is None:
            return None
        set_instrumentation_scope_attributes(span, self._scope)
        if tags:
            set_span_attribute(span, "langchain.tags", tags)
        if sanitized := sanitize_metadata(metadata):
            set_span_attribute(span, "langchain.metadata", json_dumps(sanitized))

        run = _Run(span=span, context=set_span_in_context(span, parent_context))
        if parent is not None:
            parent.children.append(run_id)
        try:
            if suppress_downstream:
                # The LangChain LLM span already records this call; keep the
                # provider SDK instrumentation underneath from recording it
                # again. Nothing else needs the LLM span as the current span
                # (OpenAI trace headers find it by run id), so unlike other
                # runs it is not attached: start and end may run in different
                # tasks, and a span left attached would parent later spans.
                run.suppress_flag = _SuppressWhileRunning(run_id)
                run.global_tokens.append(
                    context_api.attach(
                        context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, run.suppress_flag)
                    )
                )
            else:
                run.isolated_token = attach_context(run.context)
                global_context = context_api.get_current()
                if suppress:
                    # Provider calls under this run are not the LLM run's own call
                    global_context = context_api.set_value(
                        _SUPPRESS_INSTRUMENTATION_KEY, None, global_context
                    )
                run.global_tokens.append(
                    context_api.attach(set_span_in_context(span, global_context))
                )
        except Exception:
            logger.debug("Failed to attach LangChain run context", exc_info=True)

        # A previous run with the same id (should not happen) would leak its span
        if (stale := self._runs.pop(run_id, None)) is not None:
            self._close(stale)
        self._runs[run_id] = run
        return span

    @staticmethod
    def _close(run: _Run) -> None:
        if run.suppress_flag is not None:
            run.suppress_flag.active = False
        for token in reversed(run.global_tokens):
            _detach_global(token)
        run.global_tokens.clear()
        if run.isolated_token is not None:
            with contextlib.suppress(Exception):
                detach_context(run.isolated_token)
            run.isolated_token = None
        if run.span.is_recording():
            run.span.end()

    def _end_run(self, run_id: UUID) -> None:
        _ = self._handed_off.pop(run_id, None)
        run = self._runs.pop(run_id, None)
        if run is None:
            return
        # Children still open when their parent finishes (e.g. a stream that
        # was abandoned) would otherwise never be exported
        for child_id in run.children:
            self._end_run(child_id)
        self._close(run)

    def _start_chain_like(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        name: str,
        tags: list[str] | None,
        metadata: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> Span | None:
        is_root = parent_run_id is None or (
            parent_run_id not in self._runs and parent_run_id not in self._handed_off
        )
        attributes: dict[str, AttributeValue] = {}
        if is_root:
            span_name = f"{name}.workflow"
            attributes["gen_ai.operation.name"] = "invoke_agent"
            attributes["gen_ai.agent.name"] = name
            attributes["gen_ai.agent.id"] = str(run_id)
        else:
            span_name = f"execute_task {name}"
            attributes["gen_ai.operation.name"] = "execute_task"
        if thread_id := _thread_id(metadata, kwargs):
            attributes["gen_ai.conversation.id"] = thread_id

        # Pregel.stream publishes the graph topology for the first chain run it
        # starts, which is the graph's own run. Consume it either way, so it does
        # not land on a nested run, but only record it on a root span: that is
        # where the trace view reads it from.
        topology = LANGGRAPH_TOPOLOGY.get()
        if topology:
            if is_root:
                for key, value in topology.items():
                    attributes[f"lmnr.association.properties.{key}"] = value
            topology.clear()

        return self._start_run(
            run_id, parent_run_id, span_name, attributes=attributes, tags=tags, metadata=metadata
        )

    def _start_llm(
        self,
        serialized: dict[str, Any],
        run_id: UUID,
        parent_run_id: UUID | None,
        operation: str,
        tags: list[str] | None,
        metadata: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> Span | None:
        if self.current_llm_run_id() is None and _provider_records_call(kwargs):
            # The provider SDK instrumentation records this call, with its
            # provider-specific attributes and debug replay, as the LLM span.
            # It parents from the current context, as runs started here do.
            parent = self._runs.get(parent_run_id) if parent_run_id is not None else None
            self._handed_off[run_id] = (
                parent.context if parent is not None else get_current_context()
            )
            if parent is not None:
                parent.children.append(run_id)
            return None

        name = _run_name(serialized, kwargs)
        vendor = detect_vendor_from_class(_class_name(serialized))
        span = self._start_run(
            run_id,
            parent_run_id,
            f"{name}.{'chat' if operation == 'chat' else 'completion'}",
            span_type="LLM",
            attributes={
                "gen_ai.operation.name": operation,
                "gen_ai.system": vendor,
                "gen_ai.provider.name": vendor,
            },
            tags=tags,
            metadata=metadata,
            suppress_downstream=True,
        )
        if span is not None:
            self._runs[run_id].request_model = set_request_attributes(
                span, serialized, metadata, kwargs
            )
        return span

    def _set_output(self, run_id: UUID, output: Any) -> None:
        run = self._runs.get(run_id)
        if run is not None and should_send_prompts():
            set_span_attribute(run.span, SPAN_OUTPUT, json_dumps(output))

    def _handle_error(self, error: BaseException, run_id: UUID) -> None:
        run = self._runs.get(run_id)
        if run is None:
            # A handed-off LLM run has no span (the provider span records the
            # error), but its `_handed_off` entry must still go: a failed root
            # call has no parent `_end_run` to clear it.
            _ = self._handed_off.pop(run_id, None)
            return
        try:
            run.span.set_attribute("error.type", type(error).__name__)
            run.span.record_exception(error)
            run.span.set_status(Status(StatusCode.ERROR, str(error)))
        finally:
            self._end_run(run_id)

    @dont_throw
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._start_chain_like(
            run_id, parent_run_id, _run_name(serialized, kwargs), tags, metadata, kwargs
        )
        if span is not None and should_send_prompts():
            set_span_attribute(span, SPAN_INPUT, json_dumps(inputs))

    @dont_throw
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._set_output(run_id, outputs)
        finally:
            self._end_run(run_id)

    @dont_throw
    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_error(error, run_id)

    @dont_throw
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._start_llm(serialized, run_id, parent_run_id, "chat", tags, metadata, kwargs)
        if span is not None:
            set_chat_request(span, messages)

    @dont_throw
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        span = self._start_llm(
            serialized, run_id, parent_run_id, "text_completion", tags, metadata, kwargs
        )
        if span is not None:
            set_completion_request(span, prompts)

    @dont_throw
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # End the run even if recording the response fails: an LLM run left
        # open keeps its suppression flag active
        try:
            run = self._runs.get(run_id)
            if run is not None:
                set_response_attributes(run.span, response, run.request_model)
        finally:
            self._end_run(run_id)

    @dont_throw
    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_error(error, run_id)

    @dont_throw
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _run_name(serialized, kwargs)
        attributes: dict[str, AttributeValue] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
        }
        description = (serialized or {}).get("description")
        if isinstance(description, str) and description:
            attributes["gen_ai.tool.description"] = description
        tool_call_id = kwargs.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            attributes["gen_ai.tool.call.id"] = tool_call_id
        span = self._start_run(
            run_id,
            parent_run_id,
            f"execute_tool {name}",
            span_type="TOOL",
            attributes=attributes,
            tags=tags,
            metadata=metadata,
        )
        if span is not None and should_send_prompts():
            set_span_attribute(
                span, SPAN_INPUT, json_dumps(inputs if inputs is not None else input_str)
            )

    @dont_throw
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._set_output(run_id, output)
        finally:
            self._end_run(run_id)

    @dont_throw
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_error(error, run_id)

    @dont_throw
    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = _run_name(serialized, kwargs)
        span = self._start_run(
            run_id,
            parent_run_id,
            f"vector_db_retrieve {name}",
            attributes={"gen_ai.operation.name": "vector_db_retrieve"},
            tags=tags,
            metadata=metadata,
        )
        if span is not None and should_send_prompts():
            set_span_attribute(span, SPAN_INPUT, json_dumps(query))

    @dont_throw
    def on_retriever_end(
        self,
        documents: Sequence[Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._set_output(run_id, retriever_output(documents))
        finally:
            self._end_run(run_id)

    @dont_throw
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle_error(error, run_id)

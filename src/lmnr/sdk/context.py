from .collector import ThreadManager
from .client import Laminar
from .providers import Provider
from .providers.fallback import FallbackProvider
from .tracing_types import EvaluateEvent, Event, Span, Trace
from .types import PipelineRunResponse
from .utils import PROVIDER_NAME_TO_OBJECT, is_async_iterator, is_iterator

from contextvars import ContextVar
from typing import Any, AsyncGenerator, Generator, Literal, Optional, Union
import atexit
import datetime
import dotenv
import inspect
import logging
import os
import pydantic
import uuid


_lmnr_stack_context: ContextVar[list[Union[Span, Trace]]] = ContextVar(
    "lmnr_stack_context", default=[]
)
_root_trace_id_context: ContextVar[Optional[str]] = ContextVar(
    "root_trace_id_context", default=None
)


class LaminarContextManager:
    _log = logging.getLogger("laminar.context_manager")

    def __init__(
        self,
        project_api_key: str = None,
        threads: int = 1,
        max_task_queue_size: int = 1000,
    ):
        self.project_api_key = project_api_key or os.environ.get("LMNR_PROJECT_API_KEY")
        if not self.project_api_key:
            dotenv_path = dotenv.find_dotenv(usecwd=True)
            self.project_api_key = dotenv.get_key(
                dotenv_path=dotenv_path, key_to_get="LMNR_PROJECT_API_KEY"
            )
        self.laminar = Laminar(project_api_key=self.project_api_key)
        self.thread_manager = ThreadManager(
            client=self.laminar,
            max_task_queue_size=max_task_queue_size,
            threads=threads,
        )
        # atexit executes functions last in first out, so we want to make sure
        # that we finalize the trace before thread manager is closed, so the updated
        # trace is sent to the server
        atexit.register(self._force_finalize_trace)

    def observe_start(
        self,
        # span attributes
        name: str,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        span_type: Literal["DEFAULT", "LLM"] = "DEFAULT",
        check_event_names: list[str] = None,
        # trace attributes
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
    ) -> Span:
        trace_id = _root_trace_id_context.get()
        if not trace_id:
            session_id = session_id or str(uuid.uuid4())
            trace_id = uuid.uuid4()
            trace = self.update_trace(
                id=trace_id,
                user_id=user_id,
                session_id=session_id,
                release=release,
                start_time=datetime.datetime.now(datetime.timezone.utc),
            )
            _root_trace_id_context.set(trace.id)
            _lmnr_stack_context.set([trace])

        parent = _lmnr_stack_context.get()[-1] if _lmnr_stack_context.get() else None
        parent_span_id = parent.id if isinstance(parent, Span) else None
        span = self.create_span(
            name=name,
            trace_id=trace_id,
            input=input,
            metadata=metadata,
            attributes=attributes,
            parent_span_id=parent_span_id,
            span_type=span_type,
            check_event_names=check_event_names,
        )
        stack = _lmnr_stack_context.get()
        _lmnr_stack_context.set(stack + [span])
        return span

    def observe_end(
        self,
        span: Span,
        provider_name: str = None,
        result: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> Any:
        stack = _lmnr_stack_context.get()
        if not stack:
            return
        provider = PROVIDER_NAME_TO_OBJECT.get(provider_name, FallbackProvider())
        new_stack = stack[:-1]
        _lmnr_stack_context.set(new_stack)

        if len(new_stack) == 1 and isinstance(stack[0], Trace):
            trace = stack[0]
            self.update_trace(
                id=trace.id,
                start_time=trace.startTime,
                end_time=datetime.datetime.now(datetime.timezone.utc),
                user_id=trace.userId,
                session_id=trace.sessionId,
                release=trace.release,
                metadata=metadata,
            )
            _root_trace_id_context.set(None)
            _lmnr_stack_context.set([])

        if error is not None:
            self.update_current_trace(
                success=False, end_time=datetime.datetime.now(datetime.timezone.utc)
            )

        if inspect.isgenerator(result) or is_iterator(result):
            return self._collect_generator_result(
                provider=provider,
                generator=result,
                span=span,
                metadata=metadata,
                attributes=attributes,
            )
        elif inspect.isasyncgen(result) or is_async_iterator(result):
            return self._collect_async_generator_result(
                provider=provider,
                generator=result,
                span=span,
                metadata=metadata,
                attributes=attributes,
            )
        if span.spanType == "LLM" and error is None:
            attributes = self._extract_llm_attributes_from_response(
                provider=provider, response=result
            )
        return self._finalize_span(
            span,
            provider=provider,
            result=error or result,
            metadata=metadata,
            attributes=attributes,
        )

    def update_current_span(
        self,
        metadata: Optional[dict[str, Any]] = None,
        check_event_names: list[str] = None,
        override: bool = False,
    ):
        stack = _lmnr_stack_context.get()
        if not stack:
            return
        span = stack[-1]
        new_metadata = (
            metadata if override else {**(span.metadata or {}), **(metadata or {})}
        )
        new_check_event_names = (
            check_event_names
            if override
            else span.evaluateEvents + (check_event_names or [])
        )
        self.update_span(
            span=span,
            metadata=new_metadata,
            evaluate_events=new_check_event_names,
        )

    def update_current_trace(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
        end_time: Optional[datetime.datetime] = None,
    ):
        existing_trace = (
            _lmnr_stack_context.get()[0] if _lmnr_stack_context.get() else None
        )
        if not existing_trace:
            return
        self.update_trace(
            id=existing_trace.id,
            start_time=existing_trace.startTime,
            end_time=end_time,
            user_id=user_id or existing_trace.userId,
            session_id=session_id or existing_trace.sessionId,
            release=release or existing_trace.release,
            metadata=metadata or existing_trace.metadata,
            success=success if success is not None else existing_trace.success,
        )

    def update_trace(
        self,
        id: uuid.UUID,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        release: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        success: bool = True,
    ) -> Trace:
        trace = Trace(
            start_time=start_time,
            end_time=end_time,
            id=id,
            user_id=user_id,
            session_id=session_id,
            release=release,
            metadata=metadata,
            success=success,
        )
        self._add_observation(trace)
        return trace

    def create_span(
        self,
        name: str,
        trace_id: uuid.UUID,
        start_time: Optional[datetime.datetime] = None,
        span_type: Literal["DEFAULT", "LLM"] = "DEFAULT",
        id: Optional[uuid.UUID] = None,
        parent_span_id: Optional[uuid.UUID] = None,
        input: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        check_event_names: list[str] = None,
    ) -> Span:
        span = Span(
            name=name,
            trace_id=trace_id,
            start_time=start_time or datetime.datetime.now(datetime.timezone.utc),
            id=id,
            parent_span_id=parent_span_id,
            input=input,
            metadata=metadata,
            attributes=attributes,
            span_type=span_type,
            evaluate_events=check_event_names or [],
        )
        return span

    def update_span(
        self,
        span: Span,
        finalize: bool = False,
        end_time: Optional[datetime.datetime] = None,
        output: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
        evaluate_events: Optional[list[EvaluateEvent]] = None,
    ) -> Span:
        span.update(
            end_time=end_time,
            output=output,
            metadata=metadata,
            attributes=attributes,
            evaluate_events=evaluate_events,
        )
        if finalize:
            self._add_observation(span)
        return span

    def event(
        self,
        name: str,
        value: Optional[Union[str, int]] = None,
        timestamp: Optional[datetime.datetime] = None,
    ):
        span = _lmnr_stack_context.get()[-1] if _lmnr_stack_context.get() else None
        if not span or not isinstance(span, Span):
            self._log.warning(f"No active span to send event. Ignoring event. {name}")
            return
        event = Event(
            name=name,
            span_id=span.id,
            timestamp=timestamp,
            value=value,
        )
        span.add_event(event)

    def evaluate_event(self, name: str, data: str):
        stack = _lmnr_stack_context.get()
        if not stack or not isinstance(stack[-1], Span):
            self._log.warning(
                f"No active span to add check event. Ignoring event. {name}"
            )
            return
        stack[-1].evaluateEvents.append(EvaluateEvent(name=name, data=data))

    def run_pipeline(
        self,
        pipeline: str,
        inputs: dict[str, Any],
        env: dict[str, str] = {},
        metadata: dict[str, str] = {},
    ) -> PipelineRunResponse:
        span = _lmnr_stack_context.get()[-1] if _lmnr_stack_context.get() else None
        span_id = span.id if isinstance(span, Span) else None
        trace = _lmnr_stack_context.get()[0] if _lmnr_stack_context.get() else None
        trace_id = trace.id if isinstance(trace, Trace) else None
        return self.laminar.run(
            pipeline=pipeline,
            inputs=inputs,
            env=env,
            metadata=metadata,
            parent_span_id=span_id,
            trace_id=trace_id,
        )

    def _force_finalize_trace(self):
        self.update_current_trace(end_time=datetime.datetime.now(datetime.timezone.utc))

    def _add_observation(self, observation: Union[Span, Trace]) -> bool:
        return self.thread_manager.add_task(observation)

    def _extract_llm_attributes_from_response(
        self,
        provider: Provider,
        response: Union[str, dict[str, Any], pydantic.BaseModel],
    ) -> dict[str, Any]:
        return provider.extract_llm_attributes_from_response(response)

    def _stream_list_to_dict(
        self, provider: Provider, response: list[Any]
    ) -> dict[str, Any]:
        return provider.stream_list_to_dict(response)

    def _extract_llm_output(
        self,
        provider: Provider,
        result: Union[dict[str, Any], pydantic.BaseModel],
    ) -> str:
        return provider.extract_llm_output(result)

    def _finalize_span(
        self,
        span: Span,
        provider: Provider = None,
        result: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Any:
        self.update_span(
            span=span,
            finalize=True,
            output=(
                result
                if span.spanType != "LLM"
                else self._extract_llm_output(provider, result)
            ),
            metadata=metadata,
            attributes=attributes,
        )
        return result

    def _collect_generator_result(
        self,
        generator: Generator,
        span: Span,
        provider: Provider = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> Generator:
        items = []
        try:
            for item in generator:
                items.append(item)
                yield item

        finally:
            output = items
            if all(isinstance(item, str) for item in items):
                output = "".join(items)
            if span.spanType == "LLM":
                collected = self._stream_list_to_dict(
                    provider=provider, response=output
                )
                attributes = self._extract_llm_attributes_from_response(
                    provider=provider, response=collected
                )
            self._finalize_span(
                span=span,
                provider=provider,
                result=collected,
                metadata=metadata,
                attributes=attributes,
            )

    async def _collect_async_generator_result(
        self,
        generator: AsyncGenerator,
        span: Span,
        provider: Provider = None,
        metadata: Optional[dict[str, Any]] = None,
        attributes: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator:
        items = []
        try:
            async for item in generator:
                items.append(item)
                yield item

        finally:
            output = items
            if all(isinstance(item, str) for item in items):
                output = "".join(items)
            if span.spanType == "LLM":
                collected = self._stream_list_to_dict(
                    provider=provider, response=output
                )
                attributes = self._extract_llm_attributes_from_response(
                    provider=provider, response=collected
                )
            self._finalize_span(
                span=span,
                provider=provider,
                result=collected,
                metadata=metadata,
                attributes=attributes,
            )


# TODO: add lock for thread safety
class LaminarSingleton:
    _instance = None
    _l: Optional[LaminarContextManager] = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LaminarSingleton, cls).__new__(cls)
        return cls._instance

    def get(cls, *args, **kwargs) -> LaminarContextManager:
        if not cls._l:
            cls._l = LaminarContextManager(*args, **kwargs)
        return cls._l

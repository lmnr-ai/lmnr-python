"""LangChain instrumentation.

Ported from OpenLLMetry's `opentelemetry-instrumentation-langchain`. A single
`LaminarCallbackHandler` is added to every LangChain callback manager, which
turns chain / LLM / tool / retriever runs into spans; LangGraph graph runs are
chain runs too, so they are covered by the same handler (see `langgraph/` for
the graph topology attributes).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from importlib.metadata import version
from typing import Any, Required

from opentelemetry.trace import set_span_in_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from typing_extensions import override

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.langchain.callback_handler import (
    LaminarCallbackHandler,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_instruments = ("langchain-core >= 0.1.0",)


class LangchainSpec(WrappedFunctionSpec, total=False):
    # The instrumentor's one callback handler: the init wrapper installs it,
    # the OpenAI wrappers look runs up in it
    handler: Required[LaminarCallbackHandler]


def wrap_callback_manager_init(
    to_wrap: LangchainSpec,
    wrapped: Callable[..., None],
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> None:
    wrapped(*args, **kwargs)
    try:
        # Child managers are built from the parent's inheritable handlers, so
        # the handler is usually there already
        if not any(isinstance(h, LaminarCallbackHandler) for h in instance.inheritable_handlers):
            instance.add_handler(to_wrap["handler"], True)
    except Exception:
        logger.debug("Failed to add the Laminar LangChain callback handler", exc_info=True)


def wrap_openai_call(
    to_wrap: LangchainSpec,
    wrapped: Callable[..., Any],
    _instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Send the LLM run's span as a `traceparent` header with the OpenAI request.

    The OpenAI SDK instrumentation is suppressed under a LangChain LLM run, so
    this is what lets a tracing-aware server (e.g. a proxy) link its spans to
    ours. Works for sync, async and streaming methods alike: only the kwargs
    are touched, before the call. `BaseChatModel.stream` / `astream` call
    `_stream` / `_astream` without a `run_manager`, so the run is then found
    through the context its LLM run attached.
    """
    try:
        handler = to_wrap["handler"]
        run_id = getattr(kwargs.get("run_manager"), "run_id", None)
        if run_id is None:
            run_id = handler.current_llm_run_id()
        span = handler.get_span(run_id) if run_id is not None else None
        if span is not None:
            extra_headers = dict(kwargs.get("extra_headers") or {})
            TraceContextTextMapPropagator().inject(extra_headers, context=set_span_in_context(span))
            kwargs["extra_headers"] = extra_headers
    except Exception:
        logger.debug("Failed to inject trace context into the OpenAI request", exc_info=True)
    return wrapped(*args, **kwargs)


# (module, class, methods) whose kwargs reach `openai`'s `create(...)` call
_OPENAI_CALL_TARGETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "langchain_community.llms.openai",
        "BaseOpenAI",
        ("_generate", "_agenerate", "_stream", "_astream"),
    ),
    (
        "langchain_openai.llms.base",
        "BaseOpenAI",
        ("_generate", "_agenerate", "_stream", "_astream"),
    ),
    (
        "langchain_openai.chat_models.base",
        "BaseChatOpenAI",
        # `ChatOpenAI._stream` / `_astream` delegate to these via `super()`;
        # the `_responses` ones serve the Responses API
        (
            "_generate",
            "_agenerate",
            "_stream",
            "_astream",
            "_stream_responses",
            "_astream_responses",
        ),
    ),
)


class LangchainInstrumentor(BaseLaminarInstrumentor):
    """An instrumentor for LangChain (and LangGraph runs, which are LangChain runs)."""

    _scope: LaminarInstrumentationScopeAttributes | None = None
    # `BaseInstrumentor` is a singleton but `__init__` runs on every
    # construction; keep the one handler that installed wrappers already use
    _handler: LaminarCallbackHandler | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                langchain_version = version("langchain-core")
            except Exception:
                logger.debug("Failed to get langchain-core version", exc_info=True)
                langchain_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="langchain",
                version=langchain_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        if self._handler is None:
            self._handler = LaminarCallbackHandler(self.instrumentation_scope())
        wrapped_functions: list[WrappedFunctionSpec] = [
            LangchainSpec(
                package_name="langchain_core.callbacks",
                object_name="BaseCallbackManager",
                method_name="__init__",
                is_async=False,
                is_streaming=False,
                # Spans are created by the callback handler this installs,
                # once per run, not per call of the wrapped method
                span_name=None,
                span_type=None,
                instrumentation_scope=self.instrumentation_scope(),
                wrapper_function=wrap_callback_manager_init,
                handler=self._handler,
            )
        ]
        for package_name, object_name, methods in _OPENAI_CALL_TARGETS:
            for method_name in methods:
                wrapped_functions.append(
                    LangchainSpec(
                        package_name=package_name,
                        object_name=object_name,
                        method_name=method_name,
                        is_async=method_name.startswith("_a"),
                        is_streaming="stream" in method_name,
                        span_name=None,
                        span_type=None,
                        instrumentation_scope=self.instrumentation_scope(),
                        wrapper_function=wrap_openai_call,
                        handler=self._handler,
                    )
                )
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=wrapped_functions
        )

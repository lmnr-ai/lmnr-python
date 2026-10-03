from collections.abc import Awaitable, Callable, Collection, Sequence
from importlib.metadata import version
from typing import Any, cast

import pydantic
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeVar, override

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.wrapper_helpers import (
    set_instrumentation_scope_attributes,
    stamp_instrumentation_scope,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import JsonValue, get_input_from_func_args, json_dumps

logger = get_default_logger(__name__)

try:
    from skyvern import Skyvern  # pyright: ignore[reportMissingImports]: TODO: Python 3.11 upgrade and install as a dev dep
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed " +
        "to patch Skyvern, which is not installed. Use `pip install skyvern` " +
        "to install Skyvern or remove this import."
    ) from e

_instruments = ("skyvern >= 0.1.0",)


T = TypeVar("T")

async def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped: Callable[..., Awaitable[T]],
    _instance: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    span_name = to_wrap.get("span_name") or "Skyvern.span"
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }
    attributes = {k: v for k,v in attributes.items() if v is not None}

    attributes["lmnr.span.input"] = json_dumps(
        get_input_from_func_args(wrapped, True, args, kwargs)
    )

    # `Laminar.start_as_current_span` rather than a per-library tracer: this
    # instrumentor no longer receives one. The attributes are passed through
    # verbatim so the emitted span is unchanged.
    with Laminar.start_as_current_span(span_name, attributes=cast(dict[str, AttributeValue], attributes)) as span:
        stamp_instrumentation_scope(span, to_wrap)
        try:
            result = await wrapped(*args, **kwargs)

            to_serialize = result
            serialized = (
                to_serialize.model_dump_json()
                if isinstance(to_serialize, pydantic.BaseModel)
                else json_dumps(cast(JsonValue, to_serialize))
            )
            span.set_attribute("lmnr.span.output", serialized)
            return result

        except Exception as e:
            span.record_exception(e)
            raise


def instrument_llm_handler(  # pyright: ignore[reportAny]
    scope: LaminarInstrumentationScopeAttributes | None = None,
) -> Any:  # pyright: ignore[reportExplicitAny]
    """Wrap skyvern's global LLM handler, returning the original for restoration.

    Reading `app.LLM_API_HANDLER` raises `RuntimeError` until skyvern's forge app
    has been started, which is the normal state at `Laminar.initialize()` time —
    hence the guard at the call site.
    """
    from skyvern.forge import app  # pyright: ignore[reportMissingImports]: TODO: Python 3.11 upgrade and install as a dev dep

    # Store the original handler
    original_handler = app.LLM_API_HANDLER  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    async def wrapped_llm_handler(*args: Any, **kwargs: Any) -> Any:   # pyright: ignore[reportAny, reportExplicitAny]

        prompt_name = kwargs.get("prompt_name", "")  # pyright: ignore[reportAny]

        if prompt_name:
            span_name = f"{prompt_name}"
        else:
            span_name = "app.LLM_API_HANDLER"

        attributes = {
            "lmnr.span.type": "DEFAULT",
        }

        with Laminar.start_as_current_span(span_name, attributes=cast(dict[str, AttributeValue], attributes)) as span:
            set_instrumentation_scope_attributes(span, scope)
            try:
                result = await original_handler(*args, **kwargs)  # pyright: ignore[reportUnknownVariableType]

                to_serialize = result# pyright: ignore[reportUnknownVariableType]
                serialized = (
                    to_serialize.model_dump_json()
                    if isinstance(to_serialize, pydantic.BaseModel)
                    else json_dumps(to_serialize)  # pyright: ignore[reportUnknownArgumentType]
                )
                span.set_attribute("lmnr.span.output", serialized)
                return result  # pyright: ignore[reportUnknownVariableType]
            except Exception as e:
                span.record_exception(e)
                raise

    # Replace the global handler
    app.LLM_API_HANDLER = wrapped_llm_handler
    return original_handler  # pyright: ignore[reportUnknownVariableType]


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    WrappedFunctionSpec(
        package_name="skyvern.library.skyvern",
        object_name="Skyvern",
        method_name="run_task",
        is_async=True,
        span_name="Skyvern.run_task",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.webeye.scraper.scraper",
        object_name=None,
        method_name="get_interactable_element_tree",
        is_async=True,
        span_name="get_interactable_element_tree",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.forge.agent",
        object_name="ForgeAgent",
        method_name="execute_step",
        is_async=True,
        span_name="ForgeAgent.execute_step",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.services.task_v2_service",
        object_name=None,
        method_name="initialize_task_v2",
        is_async=True,
        span_name="initialize_task_v2",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.services.task_v2_service",
        object_name=None,
        method_name="run_task_v2_helper",
        is_async=True,
        span_name="run_task_v2_helper",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.forge.sdk.workflow.models.block",
        object_name="Block",
        method_name="_generate_workflow_run_block_description",
        is_async=True,
        span_name="Block._generate_workflow_run_block_description",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
    WrappedFunctionSpec(
        package_name="skyvern.webeye.actions.handler",
        object_name=None,
        method_name="extract_information_for_navigation_goal",
        is_async=True,
        span_name="extract_information_for_navigation_goal",
        span_type="DEFAULT",
        wrapper_function=_wrap,
    ),
]


class SkyvernInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self):
        super().__init__()
        self._original_llm_handler: Callable[..., Any] | None = None  # pyright: ignore[reportExplicitAny]
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                skyvern_version = version("skyvern")
            except Exception:
                logger.debug("Failed to get skyvern version", exc_info=True)
                skyvern_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="skyvern",
                version=skyvern_version,
            )
        return self._scope

    @override
    def _instrument(self, **kwargs: Any):  # pyright: ignore[reportAny, reportExplicitAny]
        # Guarded: `app.LLM_API_HANDLER` raises RuntimeError until skyvern's
        # forge app is started, which is the normal state during
        # `Laminar.initialize()`. Unguarded, that exception propagated out of
        # `_instrument` before any method was wrapped, so a single uninitialized
        # global left ALL seven unwrapped — skyvern tracing silently did nothing.
        try:
            self._original_llm_handler = instrument_llm_handler(
                self.instrumentation_scope()
            )
        except Exception:
            logger.debug("Failed to instrument skyvern LLM_API_HANDLER", exc_info=True)

        super()._instrument(**kwargs)  # pyright: ignore[reportAny]

    @override
    def _uninstrument(self, **kwargs: Any):  # pyright: ignore[reportExplicitAny, reportAny]
        # `instrument_llm_handler` swaps a module-level global, which `unwrap`
        # cannot undo — without this the handler stayed wrapped forever and each
        # instrument/uninstrument cycle layered another wrapper on it.
        if self._original_llm_handler is not None:
            try:
                from skyvern.forge import app  # pyright: ignore[reportMissingImports]: TODO: Python 3.11 upgrade and install as a dev dep

                app.LLM_API_HANDLER = self._original_llm_handler
            except Exception:
                logger.debug("Failed to restore skyvern LLM_API_HANDLER", exc_info=True)
            self._original_llm_handler = None

        super()._uninstrument(**kwargs)  # pyright: ignore[reportAny]

from importlib.metadata import version
from typing import Any, Collection, Sequence

import pydantic

from lmnr import Laminar
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import get_input_from_func_args, json_dumps

logger = get_default_logger(__name__)

try:
    from skyvern import Skyvern
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Skyvern, which is not installed. Use `pip install skyvern` "
        "to install Skyvern or remove this import."
    ) from e

_instruments = ("skyvern >= 0.1.0",)



async def _wrap(
    to_wrap: WrappedFunctionSpec,
    wrapped,
    instance: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
):
    span_name = to_wrap.get("span_name")
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }

    attributes["lmnr.span.input"] = json_dumps(
        get_input_from_func_args(wrapped, True, args, kwargs)
    )

    # `Laminar.start_as_current_span` rather than a per-library tracer: this
    # instrumentor no longer receives one. The attributes are passed through
    # verbatim so the emitted span is unchanged.
    with Laminar.start_as_current_span(span_name, attributes=attributes) as span:
        try:
            result = await wrapped(*args, **kwargs)

            to_serialize = result
            serialized = (
                to_serialize.model_dump_json()
                if isinstance(to_serialize, pydantic.BaseModel)
                else json_dumps(to_serialize)
            )
            span.set_attribute("lmnr.span.output", serialized)
            return result

        except Exception as e:
            span.record_exception(e)
            raise


def instrument_llm_handler():
    """Wrap skyvern's global LLM handler, returning the original for restoration.

    Reading `app.LLM_API_HANDLER` raises `RuntimeError` until skyvern's forge app
    has been started, which is the normal state at `Laminar.initialize()` time —
    hence the guard at the call site.
    """
    from skyvern.forge import app

    # Store the original handler
    original_handler = app.LLM_API_HANDLER

    async def wrapped_llm_handler(*args, **kwargs):

        prompt_name = kwargs.get("prompt_name", "")

        if prompt_name:
            span_name = f"{prompt_name}"
        else:
            span_name = "app.LLM_API_HANDLER"

        attributes = {
            "lmnr.span.type": "DEFAULT",
        }

        with Laminar.start_as_current_span(span_name, attributes=attributes) as span:
            try:
                result = await original_handler(*args, **kwargs)

                to_serialize = result
                serialized = (
                    to_serialize.model_dump_json()
                    if isinstance(to_serialize, pydantic.BaseModel)
                    else json_dumps(to_serialize)
                )
                span.set_attribute("lmnr.span.output", serialized)
                return result
            except Exception as e:
                span.record_exception(e)
                raise

    # Replace the global handler
    app.LLM_API_HANDLER = wrapped_llm_handler
    return original_handler


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
        self._original_llm_handler = None
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
                skyvern_version = version("skyvern")
            except Exception as e:
                logger.debug(f"Failed to get skyvern version {e}")
                skyvern_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="skyvern",
                version=skyvern_version,
            )
        return self._scope

    def _instrument(self, **kwargs):
        # Guarded: `app.LLM_API_HANDLER` raises RuntimeError until skyvern's
        # forge app is started, which is the normal state during
        # `Laminar.initialize()`. Unguarded, that exception propagated out of
        # `_instrument` before any method was wrapped, so a single uninitialized
        # global left ALL seven unwrapped — skyvern tracing silently did nothing.
        try:
            self._original_llm_handler = instrument_llm_handler()
        except Exception as e:
            logger.debug(f"Failed to instrument skyvern LLM_API_HANDLER: {e}")

        super()._instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        # `instrument_llm_handler` swaps a module-level global, which `unwrap`
        # cannot undo — without this the handler stayed wrapped forever and each
        # instrument/uninstrument cycle layered another wrapper on it.
        if self._original_llm_handler is not None:
            try:
                from skyvern.forge import app

                app.LLM_API_HANDLER = self._original_llm_handler
            except Exception as e:
                logger.debug(f"Failed to restore skyvern LLM_API_HANDLER: {e}")
            self._original_llm_handler = None

        super()._uninstrument(**kwargs)

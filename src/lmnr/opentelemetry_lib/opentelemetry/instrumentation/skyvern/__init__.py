from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import with_tracer_wrapper
from lmnr.sdk.utils import get_input_from_func_args, json_dumps
from lmnr.version import __version__

logger = get_default_logger(__name__)

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from typing import Collection
from wrapt import wrap_function_wrapper
import pydantic

try:
    from skyvern import Skyvern
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed "
        "to patch Skyvern, which is not installed. Use `pip install skyvern` "
        "to install Skyvern or remove this import."
    ) from e

_instruments = ("skyvern >= 0.1.0",)

WRAPPED_METHODS = [
    {
        "package": "skyvern.library.skyvern",
        "object": "Skyvern",  # Class name
        "method": "run_task",  # Method name
        "span_name": "Skyvern.run_task",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.webeye.scraper.scraper",
        # No "object" field for module-level functions
        "method": "get_interactable_element_tree",  # Function name
        "span_name": "get_interactable_element_tree",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.forge.agent",
        "object": "ForgeAgent",
        "method": "execute_step",
        "span_name": "ForgeAgent.execute_step",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.services.task_v2_service",
        "method": "initialize_task_v2",
        "span_name": "initialize_task_v2",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.services.task_v2_service",
        "method": "run_task_v2_helper",
        "span_name": "run_task_v2_helper",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.forge.sdk.workflow.models.block",
        "object": "Block",
        "method": "_generate_workflow_run_block_description",
        "span_name": "Block._generate_workflow_run_block_description",
        "span_type": "DEFAULT",
    },
    {
        "package": "skyvern.webeye.actions.handler",
        "method": "extract_information_for_navigation_goal",
        "span_name": "extract_information_for_navigation_goal",
        "span_type": "DEFAULT",
    },
]


@with_tracer_wrapper
async def _wrap(tracer: Tracer, to_wrap, wrapped, instance, args, kwargs):
    span_name = to_wrap.get("span_name")
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }

    attributes["lmnr.span.input"] = json_dumps(
        get_input_from_func_args(wrapped, True, args, kwargs)
    )

    with tracer.start_as_current_span(span_name, attributes=attributes) as span:
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


def instrument_llm_handler(tracer: Tracer):
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

        with tracer.start_as_current_span(span_name, attributes=attributes) as span:
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


class SkyvernInstrumentor(BaseInstrumentor):
    def __init__(self):
        super().__init__()
        self._original_llm_handler = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):

        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        # Guarded, and deliberately BEFORE nothing: `app.LLM_API_HANDLER` raises
        # RuntimeError until skyvern's forge app is started, which is the normal
        # state during `Laminar.initialize()`. Unguarded, that exception
        # propagated out of `_instrument` before the loop below ran, so a single
        # uninitialized global left ALL seven methods unwrapped — i.e. skyvern
        # tracing silently did nothing.
        try:
            self._original_llm_handler = instrument_llm_handler(tracer)
        except Exception as e:
            logger.debug(f"Failed to instrument skyvern LLM_API_HANDLER: {e}")

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            # For class methods: "Class.method", for module functions: just "function_name"
            if wrap_object:
                target = f"{wrap_object}.{wrap_method}"
            else:
                target = wrap_method

            try:
                wrap_function_wrapper(
                    wrap_package,
                    target,
                    _wrap(
                        tracer,
                        wrapped_method,
                    ),
                )
            except ModuleNotFoundError:
                pass  # that's ok, we're not instrumenting everything

    def _uninstrument(self, **kwargs):

        # `instrument_llm_handler` swaps a module-level global, which `unwrap`
        # below cannot undo — without this the handler stayed wrapped forever
        # and each instrument/uninstrument cycle layered another wrapper on it.
        if self._original_llm_handler is not None:
            try:
                from skyvern.forge import app

                app.LLM_API_HANDLER = self._original_llm_handler
            except Exception as e:
                logger.debug(f"Failed to restore skyvern LLM_API_HANDLER: {e}")
            self._original_llm_handler = None

        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")

            # For class methods: "package.Class", for module functions: just "package"
            if wrap_object:
                module_path = f"{wrap_package}.{wrap_object}"
            else:
                module_path = wrap_package

            try:
                unwrap(module_path, wrap_method)
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                # Mirrors the per-wrap tolerance in `_instrument`. Several of
                # these targets live behind skyvern's own optional extras (the
                # scraper needs PIL, task_v2_service needs sqlalchemy), and
                # `unwrap` RAISES ImportError when it cannot resolve the holder
                # — so without this an install missing any one of them made
                # `uninstrument()` blow up rather than skip that target.
                logger.debug(f"Failed to uninstrument {module_path}.{wrap_method}: {e}")

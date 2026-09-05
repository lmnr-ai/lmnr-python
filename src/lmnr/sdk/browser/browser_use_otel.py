from collections.abc import Callable, Collection, Coroutine, Sequence
from importlib.metadata import version
from typing import Any, TypeVar

import pydantic
from typing_extensions import override

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


class BrowserUseSpec(WrappedFunctionSpec, total=False):
    """browser-use's per-method extras.

    Both flags are True on every current row, so the input/output recording
    below is effectively disabled — the wrapper reconstructs what it needs from
    the call arguments instead. They are kept because they are read per-row and
    a future row may want the default behaviour.
    """

    ignore_input: bool
    ignore_output: bool

try:
    # we could pull in browser-use as a dev dep, but (1) it's heavy, (2) it's Python 3.11+
    from browser_use import (  # pyright:ignore[reportMissingImports]
        AgentHistoryList,  # pyright:ignore[reportUnknownVariableType]
    )
except ImportError as e:
    raise ImportError(
        f"Attempted to import {__file__}, but it is designed " +
        "to patch Browser Use < 0.5.0, which is not installed. Use `pip install browser-use` " +
        "to install Browser Use or remove this import."
    ) from e

_instruments = ("browser-use < 0.5.0",)

#: `_wrap` returns exactly what `wrapped` returns, whatever that is per call
#: site — bounding it lets pyright track that identity instead of widening to
#: `Any`.
T = TypeVar("T")


async def _wrap(
    to_wrap: BrowserUseSpec,
    wrapped: Callable[..., Coroutine[Any, Any, T]],  # pyright: ignore[reportExplicitAny]
    instance: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    args: Sequence[Any],  # pyright: ignore[reportExplicitAny]
    kwargs: dict[str, Any],  # pyright: ignore[reportExplicitAny]
) -> T:
    span_name = to_wrap.get("span_name")
    attributes = {
        "lmnr.span.type": to_wrap.get("span_type"),
    }
    if to_wrap["method_name"] == "execute_action":
        span_name = args[0] if len(args) > 0 else kwargs.get("action_name", "action")
        attributes["lmnr.span.input"] = json_dumps(
            {
                "action": span_name,
                "params": args[1] if len(args) > 1 else kwargs.get("params", {}),
            }
        )
    else:
        if not to_wrap.get("ignore_input"):
            inp_dict = get_input_from_func_args(wrapped, True, args, kwargs)
            # Add task to the `agent.run` span input
            if to_wrap["method_name"] == "run" and hasattr(instance, "task"):
                inp_dict["task"] = instance.task
            attributes["lmnr.span.input"] = json_dumps(inp_dict)
    if to_wrap["method_name"] == "step" and to_wrap.get("object_name") == "Agent":
        # Add step number to the `agent.step` span name
        step_info = kwargs.get("step_info", args[0] if len(args) > 0 else None)
        if step_info and hasattr(step_info, "step_number"):
            span_name = f"agent.step.{step_info.step_number}"

    with Laminar.start_as_current_span(str(span_name)) as span:
        result = await wrapped(*args, **kwargs)
        if not to_wrap.get("ignore_output"):
            # A fresh `Any`-typed alias, not `result` itself: `result` carries
            # the bound `T` (so the function can return it unchanged below),
            # but the isinstance narrowing here is only for serialization and
            # must not leak back into `T`.
            to_serialize: Any = result
            if isinstance(to_serialize, AgentHistoryList):
                to_serialize = to_serialize.final_result()
            serialized = (
                to_serialize.model_dump_json()
                if isinstance(to_serialize, pydantic.BaseModel)
                else json_dumps(to_serialize)
            )
            span.set_attribute("lmnr.span.output", serialized)
        return result


WRAPPED_FUNCTIONS: list[BrowserUseSpec] = [
    BrowserUseSpec(
        package_name="browser_use.agent.service",
        object_name="Agent",
        method_name="run",
        is_async=True,
        span_name="agent.run",
        span_type="DEFAULT",
        ignore_input=True,
        ignore_output=True,
        wrapper_function=_wrap,
    ),
    BrowserUseSpec(
        package_name="browser_use.agent.service",
        object_name="Agent",
        method_name="step",
        is_async=True,
        span_name="agent.step",
        span_type="DEFAULT",
        ignore_input=True,
        ignore_output=True,
        wrapper_function=_wrap,
    ),
    BrowserUseSpec(
        package_name="browser_use.controller.service",
        object_name="Controller",
        method_name="act",
        is_async=True,
        span_name="controller.act",
        span_type="DEFAULT",
        ignore_input=True,
        ignore_output=True,
        wrapper_function=_wrap,
    ),
    BrowserUseSpec(
        package_name="browser_use.controller.registry.service",
        object_name="Registry",
        method_name="execute_action",
        is_async=True,
        span_type="TOOL",
        ignore_input=True,
        ignore_output=True,
        wrapper_function=_wrap,
    ),
]


class BrowserUseLegacyInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    @override
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    @override
    def instrumentation_scope(self) -> LaminarInstrumentationScopeAttributes:
        if self._scope is None:
            try:
                bu_version = version("browser-use")
            except Exception as e:
                logger.debug(f"Failed to get browser-use version {e}")
                bu_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="browser-use",
                version=bu_version,
            )
        return self._scope

    def __init__(self):
        super().__init__()
        self.instrumentor_config: LaminarInstrumentorConfig = LaminarInstrumentorConfig(
            wrapped_functions=[
                {**spec, "instrumentation_scope": self.instrumentation_scope()}
                for spec in WRAPPED_FUNCTIONS
            ]
        )

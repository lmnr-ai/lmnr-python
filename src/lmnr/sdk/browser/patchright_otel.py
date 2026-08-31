from importlib.metadata import version
from typing import Any, Collection

from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.base_instrumentor import (
    BaseLaminarInstrumentor,
)
from lmnr.opentelemetry_lib.opentelemetry.instrumentation.shared.types import (
    LaminarInstrumentationScopeAttributes,
    LaminarInstrumentorConfig,
    WrappedFunctionSpec,
)
from lmnr.sdk.browser.playwright_otel import WRAPPED_FUNCTIONS as PLAYWRIGHT_FUNCTIONS
from lmnr.sdk.client.asynchronous.async_client import AsyncLaminarClient
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_instruments = ("patchright >= 1.9.0",)


def _to_patchright(spec: WrappedFunctionSpec) -> WrappedFunctionSpec:
    """Retarget a playwright spec at the patchright package.

    patchright is a drop-in fork, so the two tables differ only in the package
    name. They used to be maintained as two hand-written copies, which drifted:
    patchright was missing both `Browser.new_page` rows (and did not even import
    their wrapper), so patchright users silently lost session recording for
    pages opened that way. Deriving the table removes the class of bug.
    """
    return {
        **spec,
        "package_name": spec["package_name"].replace("playwright.", "patchright.", 1),
    }


WRAPPED_FUNCTIONS: list[WrappedFunctionSpec] = [
    _to_patchright(spec) for spec in PLAYWRIGHT_FUNCTIONS
]


class PatchrightInstrumentor(BaseLaminarInstrumentor):
    _scope: LaminarInstrumentationScopeAttributes | None = None

    def __init__(self, async_client: AsyncLaminarClient):
        super().__init__()
        self.async_client = async_client
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
                pr_version = version("patchright")
            except Exception as e:
                logger.debug(f"Failed to get patchright version {e}")
                pr_version = "unknown"
            self._scope = LaminarInstrumentationScopeAttributes(
                name="patchright",
                version=pr_version,
            )
        return self._scope

    def wrapper_kwargs(self) -> dict[str, Any]:
        # See PlaywrightInstrumentor: sync wrappers also get the async client.
        return {"client": self.async_client}

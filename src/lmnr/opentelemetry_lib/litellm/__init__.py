"""LiteLLM callback logger for Laminar (deprecated no-op)"""

from lmnr.opentelemetry_lib.utils.package_check import is_package_installed
from lmnr.sdk.log import get_default_logger

logger = get_default_logger(__name__)

_DEPRECATION_MESSAGE = (
    "Laminar LiteLLM callback is deprecated. "
    + "LiteLLM is already instrumented by Laminar. This callback will not have any effect. "
    + "You can safely remove the callback from your code."
)

if is_package_installed("litellm"):
    from litellm.integrations.custom_logger import CustomLogger as _Base
else:
    _Base = object


class LaminarLiteLLMCallback(_Base):  # pyright: ignore[reportGeneralTypeIssues]
    """Deprecated no-op. LiteLLM is instrumented automatically by Laminar."""

    def __init__(self, **kwargs):
        if _Base is object:
            super().__init__()
        else:
            super().__init__(**kwargs)
        logger.warning(_DEPRECATION_MESSAGE)

    def log_success_event(self, *args, **kwargs):
        pass

    def log_failure_event(self, *args, **kwargs):
        pass

    async def async_log_success_event(self, *args, **kwargs):
        pass

    async def async_log_failure_event(self, *args, **kwargs):
        pass

import os
from typing import TypeVar, cast

from opentelemetry import context as context_api


LMNR_TRACE_CONTENT = "LMNR_TRACE_CONTENT"


T = TypeVar("T")


def should_send_prompts():
    return (
        os.getenv(LMNR_TRACE_CONTENT) or "true"
    ).lower() == "true" or cast(bool, context_api.get_value("override_enable_content_tracing"))



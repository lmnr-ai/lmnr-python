from typing import Callable, Optional

from opentelemetry._events import EventLogger


class Config:
    enrich_token_usage = False
    enrich_assistant = False
    exception_logger = None
    get_common_metrics_attributes: Callable[[], dict] = lambda: {}
    enable_trace_context_propagation: bool = True
    use_legacy_attributes = True
    event_logger: Optional[EventLogger] = None

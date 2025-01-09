import sys

from typing import Optional, Set
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.util.re import parse_env_headers

from lmnr.openllmetry_sdk.instruments import Instruments
from lmnr.openllmetry_sdk.config import (
    is_content_tracing_enabled,
    is_tracing_enabled,
)
from lmnr.openllmetry_sdk.tracing.tracing import TracerWrapper
from typing import Dict


class Traceloop:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(
        app_name: Optional[str] = sys.argv[0],
        api_endpoint: str = "https://api.lmnr.ai",
        api_key: Optional[str] = None,
        headers: Dict[str, str] = {},
        disable_batch=False,
        exporter: Optional[SpanExporter] = None,
        processor: Optional[SpanProcessor] = None,
        propagator: Optional[TextMapPropagator] = None,
        should_enrich_metrics: bool = False,
        resource_attributes: dict = {},
        instruments: Optional[Set[Instruments]] = None,
    ) -> None:
        if not is_tracing_enabled():
            return

        enable_content_tracing = is_content_tracing_enabled()

        if isinstance(headers, str):
            headers = parse_env_headers(headers)

        if (
            not exporter
            and not processor
            and api_endpoint == "https://api.lmnr.ai"
            and not api_key
        ):
            print(
                "Set the LMNR_PROJECT_API_KEY environment variable to your project API key"
            )
            return

        if api_key and not exporter and not processor and not headers:
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

        # Tracer init
        resource_attributes.update({SERVICE_NAME: app_name})
        TracerWrapper.set_static_params(
            resource_attributes, enable_content_tracing, api_endpoint, headers
        )
        Traceloop.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch,
            processor=processor,
            propagator=propagator,
            exporter=exporter,
            should_enrich_metrics=should_enrich_metrics,
            instruments=instruments,
        )

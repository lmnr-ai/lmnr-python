import grpc
import re
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http import Compression as HTTPCompression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)

from lmnr.sdk.log import get_default_logger
from lmnr.sdk.utils import from_env, get_otel_env_var, parse_otel_headers

logger = get_default_logger(__name__)


class LaminarSpanExporter(SpanExporter):
    instance: OTLPSpanExporter | HTTPOTLPSpanExporter
    endpoint: str
    headers: dict[str, str]
    timeout: float
    force_http: bool

    def __init__(
        self,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        force_http: bool = False,
    ):
        url = base_url or from_env("LMNR_BASE_URL") or "https://api.lmnr.ai"
        url = url.rstrip("/")
        if match := re.search(r":(\d{1,5})$", url):
            url = url[: -len(match.group(0))]
            if port is None:
                port = int(match.group(1))
        if port is None:
            port = 443 if force_http else 8443
        final_url = f"{url}:{port or 443}"
        api_key = api_key or from_env("LMNR_PROJECT_API_KEY")
        self.endpoint = final_url
        if api_key:
            self.headers = (
                {"Authorization": f"Bearer {api_key}"}
                if force_http
                else {"authorization": f"Bearer {api_key}"}
            )
        elif get_otel_env_var("HEADERS"):
            self.headers = parse_otel_headers(get_otel_env_var("HEADERS"))
        else:
            self.headers = {}
        self.timeout = timeout_seconds
        self.force_http = force_http
        if get_otel_env_var("ENDPOINT"):
            if not base_url:
                self.endpoint = get_otel_env_var("ENDPOINT")
            else:
                logger.warning(
                    "OTEL_ENDPOINT is set, but Laminar base URL is also set. Ignoring OTEL_ENDPOINT."
                )
            protocol = get_otel_env_var("PROTOCOL") or "grpc/protobuf"
            exporter_type = from_env("OTEL_EXPORTER") or "otlp_grpc"
            self.force_http = (
                protocol in ("http/protobuf", "http/json")
                or exporter_type == "otlp_http"
            )
        if not self.endpoint:
            raise ValueError(
                "Laminar base URL is not set and OTEL_ENDPOINT is not set. Please either\n"
                "- set the LMNR_BASE_URL environment variable\n"
                "- set the OTEL_ENDPOINT environment variable\n"
                "- pass the base_url parameter to Laminar.initialize"
            )

        if self.force_http:
            self.instance = HTTPOTLPSpanExporter(
                endpoint=self.endpoint,
                headers=self.headers,
                compression=HTTPCompression.Gzip,
                timeout=self.timeout,
            )
        else:
            self.instance = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers=self.headers,
                timeout=self.timeout,
                compression=grpc.Compression.Gzip,
            )

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        return self.instance.export(spans)

    def shutdown(self) -> None:
        return self.instance.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.instance.force_flush(timeout_millis)

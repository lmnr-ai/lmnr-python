import os
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

from lmnr.sdk.utils import from_env, get_otel_env_var, parse_otel_headers


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
        use_otel_config: bool = False,
    ):
        if use_otel_config:
            self._init_from_otel_config(timeout_seconds)
        else:
            self._init_from_laminar_config(
                base_url, port, api_key, timeout_seconds, force_http
            )
        self._init_instance()

    def _init_from_otel_config(self, timeout_seconds: int):
        endpoint = get_otel_env_var("ENDPOINT")
        if not endpoint:
            raise ValueError("OTEL endpoint not configured")

        self.endpoint = endpoint

        headers_str = get_otel_env_var("HEADERS")
        self.headers = parse_otel_headers(headers_str)

        timeout_str = get_otel_env_var("TIMEOUT")
        if timeout_str:
            try:
                timeout_seconds = int(timeout_str.rstrip("s"))
            except ValueError:
                pass
        self.timeout = timeout_seconds

        protocol = get_otel_env_var("PROTOCOL") or "grpc/protobuf"
        exporter_type = from_env("OTEL_EXPORTER") or "otlp_grpc"
        self.force_http = (
            protocol in ("http/protobuf", "http/json") or exporter_type == "otlp_http"
        )

    def _init_from_laminar_config(
        self,
        base_url: str | None,
        port: int | None,
        api_key: str | None,
        timeout_seconds: int,
        force_http: bool,
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
        self.headers = (
            {"Authorization": f"Bearer {api_key}"}
            if force_http
            else {"authorization": f"Bearer {api_key}"}
        )
        self.timeout = timeout_seconds
        self.force_http = force_http

    def _init_instance(self):
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

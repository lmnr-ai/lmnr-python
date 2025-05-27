import grpc
import re
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)

from lmnr.sdk.utils import from_env


class LaminarSpanExporter(SpanExporter):
    instance: OTLPSpanExporter | HTTPOTLPSpanExporter

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
        if force_http:
            self.instance = HTTPOTLPSpanExporter(
                endpoint=f"{final_url}/v1/traces",
                headers={"Authorization": f"Bearer {api_key}"},
                compression=Compression.Gzip,
                timeout=timeout_seconds,
            )
        else:
            self.instance = OTLPSpanExporter(
                endpoint=final_url,
                headers={"authorization": f"Bearer {api_key}"},
                compression=grpc.Compression.Gzip,
                timeout=timeout_seconds,
            )

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        return self.instance.export(spans)

    def shutdown(self) -> None:
        return self.instance.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.instance.force_flush(timeout_millis)

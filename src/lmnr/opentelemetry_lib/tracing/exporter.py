import grpc
import re
import threading
from urllib.parse import urlparse, urlunparse
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
    _instance_lock: threading.RLock

    def __init__(
        self,
        base_url: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 30,
        force_http: bool = False,
    ):
        self._instance_lock = threading.RLock()
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
        self._init_instance()

    def _normalize_http_endpoint(self, endpoint: str) -> str:
        """
        Normalize HTTP endpoint URL by adding /v1/traces path if no path is present.

        Args:
            endpoint: The endpoint URL to normalize

        Returns:
            The normalized endpoint URL with /v1/traces path if needed
        """
        try:
            parsed = urlparse(endpoint)
            # Check if there's no path or only a trailing slash
            if not parsed.path or parsed.path == "/":
                # Add /v1/traces to the endpoint
                new_parsed = parsed._replace(path="/v1/traces")
                normalized_url = urlunparse(new_parsed)
                logger.info(
                    f"No path found in HTTP endpoint URL. "
                    f"Adding default path /v1/traces: {endpoint} -> {normalized_url}"
                )
                return normalized_url
            return endpoint
        except Exception as e:
            logger.warning(
                f"Failed to parse endpoint URL '{endpoint}': {e}. Using as-is."
            )
            return endpoint

    def _init_instance(self):
        # Create new instance first (outside critical section for performance)
        if self.force_http:
            # Normalize HTTP endpoint to ensure it has a path
            http_endpoint = self._normalize_http_endpoint(self.endpoint)
            new_instance = HTTPOTLPSpanExporter(
                endpoint=http_endpoint,
                headers=self.headers,
                compression=HTTPCompression.Gzip,
                timeout=self.timeout,
            )
        else:
            new_instance = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers=self.headers,
                timeout=self.timeout,
                compression=grpc.Compression.Gzip,
            )

        # Atomic swap with proper cleanup
        with self._instance_lock:
            old_instance: OTLPSpanExporter | HTTPOTLPSpanExporter | None = getattr(
                self, "instance", None
            )
            if old_instance is not None:
                try:
                    old_instance.shutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down old exporter instance: {e}")
            self.instance = new_instance

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        with self._instance_lock:
            return self.instance.export(spans)

    def shutdown(self) -> None:
        with self._instance_lock:
            return self.instance.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._instance_lock:
            return self.instance.force_flush(timeout_millis)
